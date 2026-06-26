// Package router fronts the rig with one OpenAI-compatible endpoint (default :8090,
// a drop-in for FlexLLama). It reads the `model` field, finds the slot serving it, and
// reverse-proxies — streaming (SSE) included.
package router

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/cpuchip/llama-chip/internal/backends"
	"github.com/cpuchip/llama-chip/internal/config"
	"github.com/cpuchip/llama-chip/internal/fed"
	"github.com/cpuchip/llama-chip/internal/gguf"
	"github.com/cpuchip/llama-chip/internal/gpu"
	"github.com/cpuchip/llama-chip/internal/models"
	"github.com/cpuchip/llama-chip/internal/rig"
)

//go:embed static/*
var staticFS embed.FS

// Router serves the OpenAI API + a small management API over a Rig. When fed is non-nil it
// also federates: requests for a model not served locally route to a peer that serves it.
type Router struct {
	rig *rig.Rig
	fed *fed.Federation // may be nil (standalone node)
	log *log.Logger
}

func New(r *rig.Rig, f *fed.Federation, logger *log.Logger) *Router {
	return &Router{rig: r, fed: f, log: logger}
}

// Handler returns the http.Handler (mount the UI on top in Phase 3).
func (rt *Router) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", rt.proxyByModel)
	mux.HandleFunc("/v1/completions", rt.proxyByModel)
	mux.HandleFunc("/v1/embeddings", rt.proxyByModel)
	mux.HandleFunc("/v1/models", rt.listModels)
	mux.HandleFunc("/api/status", rt.status)
	mux.HandleFunc("/api/gpu", rt.gpuStatus)
	mux.HandleFunc("/api/models", rt.availableModels)
	mux.HandleFunc("/api/backends", rt.availableBackends)
	mux.HandleFunc("/api/load", rt.load)
	mux.HandleFunc("/api/unload", rt.unload)
	mux.HandleFunc("/api/unload-all", rt.unloadAll)
	mux.HandleFunc("/api/profiles", rt.profiles)
	mux.HandleFunc("/api/profile", rt.applyProfile)
	mux.HandleFunc("/api/ensure", rt.ensure)
	mux.HandleFunc("/api/guess-context", rt.guessContext)
	mux.HandleFunc("/api/live", rt.live)          // live per-slot activity (in-flight requests, tokens)
	mux.HandleFunc("/api/pool", rt.pool)          // federated pool: every node's GPUs + models
	mux.HandleFunc("/api/remote", rt.remote)      // manage a peer over the mesh (load/unload/profile + reads)
	mux.HandleFunc("/api/fed/local", rt.fedLocal) // gossip: LOCAL-only model list (peers poll this)
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, 200, map[string]string{"status": "ok"})
	})
	// the loader UI at /
	sub, _ := fs.Sub(staticFS, "static")
	mux.Handle("/", http.FileServer(http.FS(sub)))
	return rt.withAuth(mux)
}

// withAuth gates non-loopback requests with the federation bearer token when one is set.
// Loopback (the local substrate, a local browser) is always allowed, so turning on the token
// secures the mesh-exposed surface without changing how same-box clients connect. /health is
// always open (liveness probes). When no token is set, this is a pass-through.
func (rt *Router) withAuth(next http.Handler) http.Handler {
	token := rt.fed.Token()
	if token == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/health" || isLoopback(req.RemoteAddr) {
			next.ServeHTTP(w, req)
			return
		}
		if req.Header.Get("Authorization") != "Bearer "+token {
			writeErr(w, 401, "federation token required")
			return
		}
		next.ServeHTTP(w, req)
	})
}

func isLoopback(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// fedLocal reports this node's LOCAL-only model list (slot names) + identity. Peers poll it to
// learn what this node can serve; keeping it local-only (never the aggregated view) prevents
// two mutually-peered nodes from looping.
func (rt *Router) fedLocal(w http.ResponseWriter, _ *http.Request) {
	var local []string
	for _, s := range rt.rig.Snapshot() {
		local = append(local, s.Name)
	}
	writeJSON(w, 200, map[string]any{
		"node":      rt.fed.NodeName(),
		"advertise": rt.fed.Advertise(),
		"models":    local,
	})
}

// SlotLive is one backend slot's live state (a request in flight, or idle).
type SlotLive struct {
	ID              int  `json:"id"`
	Task            int  `json:"id_task"`
	Processing      bool `json:"processing"`
	PromptTokens    int  `json:"prompt_tokens"`    // size of this slot's prompt
	PromptProcessed int  `json:"prompt_processed"` // prompt tokens evaluated so far
	PromptCache     int  `json:"prompt_cache"`     // prompt tokens reused from cache
	Decoded         int  `json:"decoded"`          // output tokens generated so far
	Remain          int  `json:"n_remain"`         // output tokens left in the budget
	NCtx            int  `json:"n_ctx"`            // this slot's context window
}

// ModelLive is a loaded model's live activity across its parallel slots.
type ModelLive struct {
	Name       string     `json:"name"`
	Model      string     `json:"model"`
	GPUs       []int      `json:"gpus"`
	Parallel   int        `json:"parallel"`
	CtxPerSlot int        `json:"ctx_per_slot"`
	Active     int        `json:"active"` // slots currently processing a request
	Slots      []SlotLive `json:"slots"`
	Err        string     `json:"err,omitempty"`
}

// parseSlots maps a llama-server /slots response into SlotLive entries. parallel = slot count,
// ctxPerSlot from the first slot's n_ctx (the per-request window), active = slots processing.
func parseSlots(body []byte) (slots []SlotLive, parallel, ctxPerSlot, active int) {
	var raw []struct {
		ID                     int  `json:"id"`
		IDTask                 int  `json:"id_task"`
		NCtx                   int  `json:"n_ctx"`
		IsProcessing           bool `json:"is_processing"`
		NPromptTokens          int  `json:"n_prompt_tokens"`
		NPromptTokensProcessed int  `json:"n_prompt_tokens_processed"`
		NPromptTokensCache     int  `json:"n_prompt_tokens_cache"`
		NextToken              []struct {
			NDecoded int `json:"n_decoded"`
			NRemain  int `json:"n_remain"`
		} `json:"next_token"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, 0, 0, 0
	}
	for _, rs := range raw {
		s := SlotLive{
			ID: rs.ID, Task: rs.IDTask, Processing: rs.IsProcessing, NCtx: rs.NCtx,
			PromptTokens: rs.NPromptTokens, PromptProcessed: rs.NPromptTokensProcessed, PromptCache: rs.NPromptTokensCache,
		}
		if len(rs.NextToken) > 0 {
			s.Decoded = rs.NextToken[0].NDecoded
			s.Remain = rs.NextToken[0].NRemain
		}
		if rs.IsProcessing {
			active++
		}
		slots = append(slots, s)
	}
	parallel = len(raw)
	if parallel > 0 {
		ctxPerSlot = raw[0].NCtx
	}
	return slots, parallel, ctxPerSlot, active
}

// live reports per-slot activity for every healthy model by reading each backend's /slots —
// requests in flight, prompt/output tokens, and per-slot context use. The UI derives tok/s from
// successive polls (no Prometheus / --metrics needed).
func (rt *Router) live(w http.ResponseWriter, _ *http.Request) {
	cl := &http.Client{Timeout: time.Second}
	var out []ModelLive
	for _, s := range rt.rig.Snapshot() {
		if s.State != rig.Healthy {
			continue
		}
		m := ModelLive{Name: s.Name, Model: s.Model, GPUs: s.GPUs, Parallel: s.Parallel}
		if s.Parallel > 0 {
			m.CtxPerSlot = s.Ctx / s.Parallel
		}
		resp, err := cl.Get(fmt.Sprintf("http://127.0.0.1:%d/slots", s.Port))
		if err != nil {
			m.Err = "slots unavailable"
			out = append(out, m)
			continue
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		slots, parallel, ctxPerSlot, active := parseSlots(body)
		m.Slots, m.Active = slots, active
		if parallel > 0 { // the backend is authoritative for live slot count + per-slot ctx
			m.Parallel, m.CtxPerSlot = parallel, ctxPerSlot
		}
		out = append(out, m)
	}
	writeJSON(w, 200, map[string]any{"models": out, "at_ms": time.Now().UnixMilli()})
}

// poolNode is one node in the federated pool view (this node + every peer).
type poolNode struct {
	Name     string          `json:"name"`
	MeshAddr string          `json:"mesh_addr,omitempty"`
	IsSelf   bool            `json:"is_self"`
	Models   []string        `json:"models"`
	GPUs     []fed.RosterGPU `json:"gpus"`
}

// pool reports every node in the federation — its GPUs + models — so a single node's UI can show
// the whole pool. In hub mode it's the hub roster (self freshened from the local box); in
// static-peer mode it's local + peers (peers carry models only). Standalone → just this node.
func (rt *Router) pool(w http.ResponseWriter, _ *http.Request) {
	roster, self := rt.fed.Roster()
	if self == "" {
		self = rt.fed.NodeName()
	}
	if self == "" {
		self = "this-node"
	}
	var nodes []poolNode
	if len(roster) > 0 { // hub mode: the roster already has every node's GPUs + models
		for _, e := range roster {
			n := poolNode{Name: e.Name, MeshAddr: e.MeshAddr, IsSelf: strings.EqualFold(e.Name, self), Models: e.Models, GPUs: e.GPUs}
			if n.IsSelf { // freshen self from the local box (the roster's self entry is a few s stale)
				n.Models, n.GPUs = rt.localModelNames(), rt.localGPUs()
			}
			nodes = append(nodes, n)
		}
	} else { // static-peer or standalone: this node + any peers (peers carry models only)
		nodes = append(nodes, poolNode{Name: self, IsSelf: true, Models: rt.localModelNames(), GPUs: rt.localGPUs()})
		for _, p := range rt.fed.Peers() {
			nodes = append(nodes, poolNode{Name: p.Name, MeshAddr: p.URL, Models: p.Models})
		}
	}
	writeJSON(w, 200, map[string]any{"self": self, "nodes": nodes})
}

func (rt *Router) localModelNames() []string {
	var out []string
	for _, s := range rt.rig.Snapshot() {
		if s.State == rig.Healthy {
			out = append(out, s.Name)
		}
	}
	return out
}

func (rt *Router) localGPUs() []fed.RosterGPU {
	gs, err := gpu.Query()
	if err != nil {
		return nil
	}
	out := make([]fed.RosterGPU, 0, len(gs))
	for _, g := range gs {
		out = append(out, fed.RosterGPU{Index: g.Index, Name: g.Name, MemUsed: g.MemUsed, MemTotal: g.MemTotal, Util: g.Util})
	}
	return out
}

// remoteOps whitelists which management endpoints /api/remote may proxy to a peer, with the HTTP
// method each uses. Read ops (GET) populate a remote node's management forms (its GGUFs, GPUs,
// profiles, live state); write ops (POST) load/unload/switch on the peer over the mesh. The op
// name doubles as the local path suffix (op "load" → /api/load), so the UI can call the same op
// against self (locally) or a peer (via this proxy) with one code path.
var remoteOps = map[string]struct {
	path   string
	method string
}{
	"models":        {"/api/models", http.MethodGet},
	"backends":      {"/api/backends", http.MethodGet},
	"profiles":      {"/api/profiles", http.MethodGet},
	"gpu":           {"/api/gpu", http.MethodGet},
	"status":        {"/api/status", http.MethodGet},
	"live":          {"/api/live", http.MethodGet},
	"guess-context": {"/api/guess-context", http.MethodGet},
	"load":          {"/api/load", http.MethodPost},
	"unload":        {"/api/unload", http.MethodPost},
	"unload-all":    {"/api/unload-all", http.MethodPost},
	"profile":       {"/api/profile", http.MethodPost},
	"ensure":        {"/api/ensure", http.MethodPost},
}

// remote proxies a single management call to a peer over the mesh: GET/POST /api/remote?node=<name>
// &op=<op>[&...]. The named node is resolved to its mesh address from the federation (hub roster or
// static peers); the call is forwarded to that node's matching management endpoint with the
// federation bearer attached, and the peer's response is streamed straight back. This is how one
// node's loader UI manages every other node in the pool. (A peer's management endpoints are gated
// by ITS federation token over the mesh — the same token we forward, so a shared-secret cluster
// authenticates and an open mesh-only cluster needs none.)
func (rt *Router) remote(w http.ResponseWriter, req *http.Request) {
	if !rt.fed.Enabled() {
		writeErr(w, 400, "federation not enabled — no peers to manage")
		return
	}
	q := req.URL.Query()
	node, op := q.Get("node"), q.Get("op")
	spec, known := remoteOps[op]
	if node == "" || !known {
		writeErr(w, 400, "remote needs node=<name> and a known op (got op="+op+")")
		return
	}
	base, ok := rt.fed.NodeAddr(node)
	if !ok || base == "" {
		writeErr(w, 404, fmt.Sprintf("no mesh address for node %q (see /api/pool)", node))
		return
	}
	q.Del("node")
	q.Del("op")
	target := base + spec.path
	if rest := q.Encode(); rest != "" { // pass through any remaining params (e.g. guess-context gpus/kv)
		target += "?" + rest
	}

	var bodyR io.Reader
	if spec.method == http.MethodPost {
		b, _ := io.ReadAll(io.LimitReader(req.Body, 1<<20))
		bodyR = bytes.NewReader(b)
	}
	pr, err := http.NewRequestWithContext(req.Context(), spec.method, target, bodyR)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if spec.method == http.MethodPost {
		pr.Header.Set("Content-Type", "application/json")
	}
	if tok := rt.fed.Token(); tok != "" {
		pr.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(pr)
	if err != nil {
		writeErr(w, 502, fmt.Sprintf("remote %s %s: %v", node, op, err))
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(resp.Body, 8<<20))
}

// availableModels lists the GGUFs that could be loaded (for the UI picker).
func (rt *Router) availableModels(w http.ResponseWriter, _ *http.Request) {
	ms, err := models.Discover()
	if err != nil {
		writeJSON(w, 200, map[string]any{"models": []any{}, "error": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"models": ms})
}

// availableBackends lists the llama.cpp builds (for the UI dropdown).
func (rt *Router) availableBackends(w http.ResponseWriter, _ *http.Request) {
	bs, err := backends.Discover()
	if err != nil {
		writeJSON(w, 200, map[string]any{"backends": []any{}, "error": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"backends": bs})
}

// load adds a slot at runtime: {"model","alias","gpus":[0],"ctx_size","kv_cache","split_mode"}.
func (rt *Router) load(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeErr(w, 405, "POST only")
		return
	}
	var s config.Slot
	if err := json.NewDecoder(req.Body).Decode(&s); err != nil {
		writeErr(w, 400, "bad slot json: "+err.Error())
		return
	}
	if err := rt.rig.Load(s); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 202, map[string]string{"status": "loading", "slot": s.Name()})
}

// unload removes a slot: {"name":"..."}.
func (rt *Router) unload(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeErr(w, 405, "POST only")
		return
	}
	var body struct {
		Name string `json:"name"`
	}
	_ = json.NewDecoder(req.Body).Decode(&body)
	if err := rt.rig.Unload(body.Name); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 200, map[string]string{"status": "unloaded", "slot": body.Name})
}

// unloadAll unloads every slot, freeing all VRAM (the "free GPUs" control).
func (rt *Router) unloadAll(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeErr(w, 405, "POST only")
		return
	}
	names := rt.rig.UnloadAll()
	writeJSON(w, 200, map[string]any{"status": "unloaded", "slots": names})
}

// profiles lists the named GPU layouts.
func (rt *Router) profiles(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{"profiles": rt.rig.ProfileNames()})
}

// applyProfile switches the whole layout: {"name":"diverse"}.
func (rt *Router) applyProfile(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeErr(w, 405, "POST only")
		return
	}
	var body struct {
		Name string `json:"name"`
	}
	_ = json.NewDecoder(req.Body).Decode(&body)
	if err := rt.rig.ApplyProfile(body.Name); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 202, map[string]string{"status": "applying", "profile": body.Name})
}

// ensure is the substrate's load-by-need call: {"model","gpus":[0],"ctx_size",...}. Loads
// the model if no healthy slot serves it (evicting overlapping-GPU slots first).
func (rt *Router) ensure(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeErr(w, 405, "POST only")
		return
	}
	var s config.Slot
	if err := json.NewDecoder(req.Body).Decode(&s); err != nil {
		writeErr(w, 400, "bad slot json: "+err.Error())
		return
	}
	name, action, err := rt.rig.Ensure(s)
	if err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 200, map[string]string{"slot": name, "action": action})
}

// proxyByModel buffers the request, reads `model`, and reverse-proxies to whatever serves it:
// a LOCAL slot first, then — if federation is on — a PEER that serves it over the mesh.
func (rt *Router) proxyByModel(w http.ResponseWriter, req *http.Request) {
	body, err := io.ReadAll(io.LimitReader(req.Body, 256<<20))
	if err != nil {
		writeErr(w, 400, "read body: "+err.Error())
		return
	}
	var probe struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal(body, &probe)
	if probe.Model == "" {
		writeErr(w, 400, "missing `model`")
		return
	}

	var target *url.URL
	var label, bearer string
	// ?node=<name> pins the request to a SPECIFIC node over the mesh — the chat UI uses it to test
	// a chosen remote even when the local rig serves a model of the same name. Self / unknown falls
	// through to normal local-first resolution. The param is stripped before forwarding so the peer
	// resolves the model locally instead of trying to re-route (no double-hop, no loop).
	if pin := req.URL.Query().Get("node"); pin != "" && rt.fed.Enabled() && !rt.fed.IsSelf(pin) {
		base, ok := rt.fed.NodeAddr(pin)
		if !ok || base == "" {
			writeErr(w, 404, fmt.Sprintf("node %q not in the pool (see /api/pool)", pin))
			return
		}
		target, _ = url.Parse(base)
		label = fmt.Sprintf("node %q (%s)", pin, base)
		bearer = rt.fed.Token()
		stripQueryParam(req, "node")
	} else if in, ok := rt.rig.Resolve(probe.Model); ok {
		target, _ = url.Parse(fmt.Sprintf("http://127.0.0.1:%d", in.Port))
		label = fmt.Sprintf("local slot %q (:%d)", in.Slot.Name(), in.Port)
	} else if route, ok := rt.fed.Resolve(probe.Model); ok {
		target, _ = url.Parse(route.PeerURL)
		label = fmt.Sprintf("peer %q (%s)", route.PeerName, route.PeerURL)
		bearer = rt.fed.Token()
	} else {
		writeErr(w, 404, fmt.Sprintf("no local slot or reachable peer serves model %q (see /v1/models)", probe.Model))
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = -1 // flush immediately — keep SSE streaming responsive
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, e error) {
		writeErr(w, 502, fmt.Sprintf("%s error: %v", label, e))
	}
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	req.Host = target.Host
	if bearer != "" { // authenticate to the peer's federation endpoint (forwarded by the proxy)
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	proxy.ServeHTTP(w, req)
}

// listModels reports the models THIS endpoint can serve in OpenAI /v1/models shape — local
// slots plus, when federated, every model reachable on a peer (local wins a name clash).
// This is the CLIENT view; peers poll /api/fed/local (local-only) instead, so the aggregation
// never feeds back into the gossip and two mutually-peered nodes can't loop.
func (rt *Router) listModels(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	now := time.Now().Unix()
	seen := map[string]bool{}
	var data []model
	for _, s := range rt.rig.Snapshot() {
		seen[strings.ToLower(s.Name)] = true
		data = append(data, model{ID: s.Name, Object: "model", Created: now, OwnedBy: "llama-chip"})
	}
	for _, r := range rt.fed.RemoteModels() {
		if seen[strings.ToLower(r.Model)] {
			continue // local serves it — local wins
		}
		seen[strings.ToLower(r.Model)] = true
		data = append(data, model{ID: r.Model, Object: "model", Created: now, OwnedBy: "peer:" + r.PeerName})
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func (rt *Router) status(w http.ResponseWriter, _ *http.Request) {
	out := map[string]any{"slots": rt.rig.Snapshot()}
	if rt.fed.Enabled() {
		out["federation"] = map[string]any{
			"node":  rt.fed.NodeName(),
			"peers": rt.fed.Peers(),
		}
	}
	writeJSON(w, 200, out)
}

func (rt *Router) gpuStatus(w http.ResponseWriter, _ *http.Request) {
	gs, err := gpu.Query()
	if err != nil {
		writeJSON(w, 200, map[string]any{"gpus": []any{}, "error": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"gpus": gs})
}

// guessContext estimates the largest context window that fits a model on the chosen
// GPU(s), from current free VRAM, the model file size, and a KV-cache estimate that scales
// with model size. A rough GUESS (KV/token is architecture-dependent) — biased conservative.
//
//	GET /api/guess-context?model=<handle>&gpus=0,1&kv=q8_0
func (rt *Router) guessContext(w http.ResponseWriter, req *http.Request) {
	q := req.URL.Query()
	m, ok := models.Find(q.Get("model"))
	if !ok {
		writeErr(w, 400, "no single model matches "+q.Get("model"))
		return
	}
	gs, err := gpu.Query()
	if err != nil {
		writeErr(w, 400, "nvidia-smi: "+err.Error())
		return
	}
	byIdx := map[int]gpu.GPU{}
	for _, g := range gs {
		byIdx[g.Index] = g
	}
	freeMiB := 0
	for _, s := range strings.Split(q.Get("gpus"), ",") {
		var idx int
		fmt.Sscanf(strings.TrimSpace(s), "%d", &idx)
		if g, ok := byIdx[idx]; ok {
			freeMiB += g.MemTotal - g.MemUsed
		}
	}
	modelMiB := float64(m.Bytes) / (1 << 20)
	availMiB := float64(freeMiB) - modelMiB - 700 // 700 MiB compute/overhead headroom

	bytesPerElem := 1.0625 // q8_0 KV element (~34 bytes / 32)
	switch q.Get("kv") {
	case "f16", "":
		bytesPerElem = 2.0
	case "q4_0":
		bytesPerElem = 0.5625
	}
	// Accurate KV/token from the model's real attention dims (layers x kv-heads x head-dim).
	var kvPerTokMiB float64
	trainCtx := 0
	note := "estimate from GGUF dims — verify by loading"
	if p, err := gguf.Read(m.Path); err == nil {
		if b := p.KVBytesPerToken(bytesPerElem); b > 0 {
			kvPerTokMiB = b / (1 << 20)
			trainCtx = int(p.TrainCtx)
		}
	}
	if kvPerTokMiB == 0 { // fallback if the GGUF couldn't be parsed
		perGB := 0.005
		if bytesPerElem >= 2 {
			perGB = 0.010
		}
		kvPerTokMiB = perGB * (modelMiB / 1024)
		note = "rough estimate (GGUF attention dims unavailable)"
	}
	maxCtx := 0
	if kvPerTokMiB > 0 && availMiB > 0 {
		maxCtx = int(availMiB * 0.9 / kvPerTokMiB) // 0.9 safety margin
		maxCtx = (maxCtx / 4096) * 4096            // round down to 4k
	}
	if trainCtx > 0 && maxCtx > trainCtx { // don't exceed what the model was trained for
		maxCtx = trainCtx
	}
	writeJSON(w, 200, map[string]any{
		"max_ctx": maxCtx, "free_mib": freeMiB, "model_mib": int(modelMiB),
		"train_ctx": trainCtx, "kv_mib_per_1k_tok": int(kvPerTokMiB * 1000), "note": note,
	})
}

// stripQueryParam removes one query parameter from the request URL in place (so a forwarded
// reverse-proxy request doesn't carry our routing hint downstream).
func stripQueryParam(req *http.Request, key string) {
	q := req.URL.Query()
	q.Del(key)
	req.URL.RawQuery = q.Encode()
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": map[string]string{"message": strings.TrimSpace(msg)}})
}
