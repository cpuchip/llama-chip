package router

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cpuchip/llama-chip/internal/fed"
	"github.com/cpuchip/llama-chip/internal/rig"
)

// fakeBackend stands in for a peer node's llama-chip: it answers /api/fed/local (so the
// federation learns its model), /v1/chat/completions (the inference endpoint the router forwards
// to), and a few management endpoints (/api/status, /api/unload, /api/models) so the remote-
// management proxy can be exercised. It records the Authorization header it saw and the last
// unload it was asked to do.
type fakeBackend struct {
	srv       *httptest.Server
	name      string
	model     string
	wantTok   string
	gotAuth   string
	gotPath   string
	gotQuery  string
	gotUnload string
}

func newFakeBackend(model string) *fakeBackend {
	b := &fakeBackend{name: "peer", model: model}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/fed/local", func(w http.ResponseWriter, r *http.Request) {
		if !b.auth(w, r) {
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"node": b.name, "models": []string{b.model}})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		b.record(r)
		if !b.auth(w, r) {
			return
		}
		io.Copy(io.Discard, r.Body)
		json.NewEncoder(w).Encode(map[string]any{"served_by": b.name, "model": b.model})
	})
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		b.record(r)
		if !b.auth(w, r) {
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"slots": []map[string]any{{"name": b.model, "state": "healthy"}}})
	})
	mux.HandleFunc("/api/unload", func(w http.ResponseWriter, r *http.Request) {
		b.record(r)
		if !b.auth(w, r) {
			return
		}
		var body struct {
			Name string `json:"name"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		b.gotUnload = body.Name
		json.NewEncoder(w).Encode(map[string]any{"status": "unloaded", "slot": body.Name})
	})
	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		b.record(r)
		if !b.auth(w, r) {
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"models": []map[string]any{{"id": b.model, "bytes": 1234}}})
	})
	b.srv = httptest.NewServer(mux)
	return b
}

func (b *fakeBackend) record(r *http.Request) {
	b.gotAuth = r.Header.Get("Authorization")
	b.gotPath = r.URL.Path
	b.gotQuery = r.URL.RawQuery
}

func (b *fakeBackend) auth(w http.ResponseWriter, r *http.Request) bool {
	if b.wantTok != "" && r.Header.Get("Authorization") != "Bearer "+b.wantTok {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	return true
}

func (b *fakeBackend) close() { b.srv.Close() }

// newRouter builds a router over an EMPTY local rig (so every model resolves remotely) plus
// the given federation. A zero rig.Rig safely resolves nothing.
func newRouter(t *testing.T, f *fed.Federation) *Router {
	t.Helper()
	return New(&rig.Rig{}, f, log.New(io.Discard, "", 0))
}

func chatReq(t *testing.T, base, model string) *http.Response {
	t.Helper()
	body := strings.NewReader(`{"model":"` + model + `","messages":[]}`)
	resp, err := http.Post(base+"/v1/chat/completions", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestProxyRoutesToRemotePeer(t *testing.T) {
	peer := newFakeBackend("qwen3.6-35b-a3b")
	defer peer.close()

	f := fed.New(fed.Config{NodeName: "laptop", Peers: []fed.PeerConfig{{Name: "home", URL: peer.srv.URL}}}, log.New(io.Discard, "", 0))
	f.Refresh(context.Background())

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()

	resp := chatReq(t, front.URL, "qwen3.6-35b-a3b")
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, body %s", resp.StatusCode, b)
	}
	var got map[string]any
	json.NewDecoder(resp.Body).Decode(&got)
	if got["served_by"] != "peer" {
		t.Fatalf("expected the peer to serve the request, got %+v", got)
	}
}

func TestProxyForwardsTokenToPeer(t *testing.T) {
	peer := newFakeBackend("m1")
	peer.wantTok = "s3cret"
	defer peer.close()

	f := fed.New(fed.Config{Token: "s3cret", Peers: []fed.PeerConfig{{URL: peer.srv.URL}}}, log.New(io.Discard, "", 0))
	f.Refresh(context.Background())

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()

	// the incoming client request is loopback → exempt from auth; the router must still inject
	// the bearer token on the OUTBOUND call to the peer.
	resp := chatReq(t, front.URL, "m1")
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, body %s", resp.StatusCode, b)
	}
	if peer.gotAuth != "Bearer s3cret" {
		t.Fatalf("peer saw Authorization %q, want %q", peer.gotAuth, "Bearer s3cret")
	}
}

// TestProxyPeerSpecificTokenWinsOverNodeToken: a per-peer token (federation.peer_tokens[name])
// is attached to that peer's proxied request, OVERRIDING the node's own federation.token. This is
// how a keyless mesh node authenticates to a peer that runs its own distinct token (a public,
// bearer-gated NOCIX node). The node token is deliberately "wrong" for the peer — only the
// peer-specific token should get the request through.
func TestProxyPeerSpecificTokenWinsOverNodeToken(t *testing.T) {
	peer := newFakeBackend("m1")
	peer.name = "nocix"
	peer.wantTok = "peercred"
	defer peer.close()

	// hub-managed mode (fermion's real posture): the roster supplies the peer's model + mesh addr,
	// so the peer is reachable without polling it. node token would 401 at the peer if it leaked.
	f := fed.New(fed.Config{
		NodeName:   "fermion",
		HubURL:     "http://hub",
		Token:      "node-cluster-token",
		PeerTokens: map[string]string{"nocix": "peercred"},
	}, log.New(io.Discard, "", 0))
	f.ApplyRoster([]fed.RosterEntry{{Name: "nocix", MeshAddr: peer.srv.URL, Models: []string{"m1"}}}, "fermion")

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()

	resp := chatReq(t, front.URL, "m1")
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, body %s", resp.StatusCode, b)
	}
	if peer.gotAuth != "Bearer peercred" {
		t.Fatalf("peer saw Authorization %q, want Bearer peercred (per-peer token must win over node token)", peer.gotAuth)
	}
}

// TestProxyNodePinAttachesPeerToken: the ?node=<name> pin path also honors the per-peer token.
func TestProxyNodePinAttachesPeerToken(t *testing.T) {
	peer := newFakeBackend("shared")
	peer.name = "beta"
	peer.wantTok = "betacred"
	defer peer.close()

	f := fed.New(fed.Config{NodeName: "head", HubURL: "http://hub", PeerTokens: map[string]string{"beta": "betacred"}}, log.New(io.Discard, "", 0))
	f.ApplyRoster([]fed.RosterEntry{{Name: "beta", MeshAddr: peer.srv.URL, Models: []string{"shared"}}}, "head")

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()

	resp, err := http.Post(front.URL+"/v1/chat/completions?node=beta", "application/json",
		strings.NewReader(`{"model":"shared","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, body %s", resp.StatusCode, b)
	}
	if peer.gotAuth != "Bearer betacred" {
		t.Fatalf("pinned peer saw Authorization %q, want Bearer betacred", peer.gotAuth)
	}
}

// TestProxyForwardsIncomingBearerWhenNoTokens: with neither a node token nor a per-peer token,
// the caller's own incoming Authorization header is forwarded unchanged (the legacy keyless
// passthrough) — documenting the precedence tail: per-peer token > node token > incoming bearer.
func TestProxyForwardsIncomingBearerWhenNoTokens(t *testing.T) {
	peer := newFakeBackend("m1") // wantTok "" — accepts anything, records what it saw
	defer peer.close()

	f := fed.New(fed.Config{Peers: []fed.PeerConfig{{Name: "home", URL: peer.srv.URL}}}, log.New(io.Discard, "", 0))
	f.Refresh(context.Background())

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()

	req, _ := http.NewRequest(http.MethodPost, front.URL+"/v1/chat/completions", strings.NewReader(`{"model":"m1","messages":[]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer incoming-xyz")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if peer.gotAuth != "Bearer incoming-xyz" {
		t.Fatalf("peer saw Authorization %q, want the caller's incoming bearer forwarded unchanged", peer.gotAuth)
	}
}

func TestProxy404WhenNobodyServes(t *testing.T) {
	f := fed.New(fed.Config{Peers: []fed.PeerConfig{{URL: "http://127.0.0.1:1"}}}, log.New(io.Discard, "", 0))
	f.Refresh(context.Background()) // peer unreachable → no routes

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()
	resp := chatReq(t, front.URL, "nobody-serves-this")
	defer resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Fatalf("expected 404 when no local slot or peer serves the model, got %d", resp.StatusCode)
	}
}

func TestV1ModelsAggregatesRemote(t *testing.T) {
	peer := newFakeBackend("remote-only")
	defer peer.close()
	f := fed.New(fed.Config{Peers: []fed.PeerConfig{{Name: "home", URL: peer.srv.URL}}}, log.New(io.Discard, "", 0))
	f.Refresh(context.Background())

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()
	resp, err := http.Get(front.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Data []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	var found bool
	for _, d := range out.Data {
		if d.ID == "remote-only" && d.OwnedBy == "peer:home" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected remote-only owned_by peer:home in /v1/models, got %+v", out.Data)
	}
}

func TestAuthMiddlewareGatesNonLoopback(t *testing.T) {
	// a federation with a token enabled (one configured peer so it's "enabled")
	f := fed.New(fed.Config{Token: "tok", Peers: []fed.PeerConfig{{URL: "http://127.0.0.1:1"}}}, log.New(io.Discard, "", 0))
	h := newRouter(t, f).Handler()

	// non-loopback caller, no token → 401
	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.RemoteAddr = "10.4.0.7:5555"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 401 {
		t.Fatalf("expected 401 for non-loopback without token, got %d", rec.Code)
	}

	// non-loopback caller WITH token → allowed
	req = httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.RemoteAddr = "10.4.0.7:5555"
	req.Header.Set("Authorization", "Bearer tok")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("expected 200 for non-loopback with token, got %d", rec.Code)
	}

	// /health is always open, even non-loopback without token
	req = httptest.NewRequest(http.MethodGet, "/health", nil)
	req.RemoteAddr = "10.4.0.7:5555"
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("expected /health open, got %d", rec.Code)
	}
}

func TestAuthAllowsLoopbackWithoutToken(t *testing.T) {
	f := fed.New(fed.Config{Token: "tok", Peers: []fed.PeerConfig{{URL: "http://127.0.0.1:1"}}}, log.New(io.Discard, "", 0))
	h := newRouter(t, f).Handler()
	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.RemoteAddr = "127.0.0.1:5555" // the local substrate / browser
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("loopback should never need the token, got %d", rec.Code)
	}
}

func TestStandaloneNodeUnaffected(t *testing.T) {
	// fed nil (no peers) → no auth, /api/status has no federation section, unknown model 404s.
	h := newRouter(t, nil).Handler()
	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.RemoteAddr = "10.0.0.9:1" // even a non-loopback caller is fine when no token is set
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("standalone status should be open, got %d", rec.Code)
	}
	var out map[string]any
	json.Unmarshal(rec.Body.Bytes(), &out)
	if _, ok := out["federation"]; ok {
		t.Fatal("standalone node should not report a federation section")
	}
}

func TestIsLoopback(t *testing.T) {
	cases := map[string]bool{
		"127.0.0.1:8090": true,
		"[::1]:8090":     true,
		"10.4.0.7:5555":  false,
		"100.84.0.3:80":  false,
	}
	for addr, want := range cases {
		if got := isLoopback(addr); got != want {
			t.Errorf("isLoopback(%q)=%v want %v", addr, got, want)
		}
	}
}

func TestPoolEndpoint(t *testing.T) {
	f := fed.New(fed.Config{NodeName: "beta", HubURL: "http://hub"}, log.New(io.Discard, "", 0))
	f.ApplyRoster([]fed.RosterEntry{
		{Name: "alpha", MeshAddr: "http://100.84.0.2:8090", Models: []string{"qwen3.6-35b-a3b"},
			GPUs: []fed.RosterGPU{{Index: 0, Name: "RTX 4090", MemTotal: 24564, MemUsed: 21000}}},
		{Name: "beta", MeshAddr: "http://100.84.0.3:8090"},
	}, "beta")

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()
	resp, err := http.Get(front.URL + "/api/pool")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Self  string `json:"self"`
		Nodes []struct {
			Name   string `json:"name"`
			IsSelf bool   `json:"is_self"`
			GPUs   []struct {
				Name     string `json:"name"`
				MemTotal int    `json:"mem_total"`
			} `json:"gpus"`
		} `json:"nodes"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if out.Self != "beta" || len(out.Nodes) != 2 {
		t.Fatalf("pool: self=%q nodes=%d, want beta/2 (%+v)", out.Self, len(out.Nodes), out.Nodes)
	}
	// the peer (alpha) carries its GPUs from the roster, flagged not-self
	var ok bool
	for _, n := range out.Nodes {
		if n.Name == "alpha" && !n.IsSelf && len(n.GPUs) == 1 && n.GPUs[0].Name == "RTX 4090" {
			ok = true
		}
	}
	if !ok {
		t.Fatalf("expected alpha's GPUs in the pool, got %+v", out.Nodes)
	}
}

// hubFed builds a hub-managed federation whose roster places `peerName` at the given mesh address
// (the fakeBackend URL), with `self` as this node — so NodeAddr resolves the peer for /api/remote.
func hubFed(t *testing.T, self, peerName, meshAddr, token string) *fed.Federation {
	t.Helper()
	f := fed.New(fed.Config{NodeName: self, HubURL: "http://hub", Token: token}, log.New(io.Discard, "", 0))
	f.ApplyRoster([]fed.RosterEntry{{Name: peerName, MeshAddr: meshAddr, Models: []string{"m1"}}}, self)
	return f
}

func TestRemoteProxiesGET(t *testing.T) {
	peer := newFakeBackend("m1")
	defer peer.close()
	f := hubFed(t, "head", "alpha", peer.srv.URL, "")

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()
	resp, err := http.Get(front.URL + "/api/remote?node=alpha&op=status")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, body %s", resp.StatusCode, b)
	}
	var out struct {
		Slots []map[string]any `json:"slots"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Slots) != 1 || out.Slots[0]["name"] != "m1" {
		t.Fatalf("expected the peer's status slots proxied back, got %+v", out.Slots)
	}
	if peer.gotPath != "/api/status" {
		t.Fatalf("peer saw path %q, want /api/status", peer.gotPath)
	}
}

func TestRemoteProxiesPOSTUnload(t *testing.T) {
	peer := newFakeBackend("m1")
	defer peer.close()
	f := hubFed(t, "head", "alpha", peer.srv.URL, "")

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()
	resp, err := http.Post(front.URL+"/api/remote?node=alpha&op=unload", "application/json",
		strings.NewReader(`{"name":"qwen"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, body %s", resp.StatusCode, b)
	}
	if peer.gotUnload != "qwen" {
		t.Fatalf("peer was asked to unload %q, want qwen", peer.gotUnload)
	}
}

func TestRemoteForwardsToken(t *testing.T) {
	peer := newFakeBackend("m1")
	peer.wantTok = "s3cret"
	defer peer.close()
	f := hubFed(t, "head", "alpha", peer.srv.URL, "s3cret")

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()
	resp, err := http.Get(front.URL + "/api/remote?node=alpha&op=models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected the token to authenticate the remote read, got %d", resp.StatusCode)
	}
	if peer.gotAuth != "Bearer s3cret" {
		t.Fatalf("peer saw Authorization %q, want Bearer s3cret", peer.gotAuth)
	}
}

func TestRemoteUnknownNodeAndOp(t *testing.T) {
	peer := newFakeBackend("m1")
	defer peer.close()
	f := hubFed(t, "head", "alpha", peer.srv.URL, "")
	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()

	// unknown op → 400
	if r, _ := http.Get(front.URL + "/api/remote?node=alpha&op=nuke"); r != nil && r.StatusCode != 400 {
		t.Fatalf("unknown op should 400, got %d", r.StatusCode)
	}
	// unknown node → 404
	if r, _ := http.Get(front.URL + "/api/remote?node=ghost&op=status"); r != nil && r.StatusCode != 404 {
		t.Fatalf("unknown node should 404, got %d", r.StatusCode)
	}
}

// TestProxyNodeOverridePicksNamedPeer: two peers serve the SAME model. Without ?node= the first
// wins; with ?node=<second> the request is pinned to that node — the mechanism that lets the chat
// UI test a chosen remote even on a name clash. Also guards that the node param is stripped (the
// chosen peer must serve locally, not re-route).
func TestProxyNodeOverridePicksNamedPeer(t *testing.T) {
	alpha := newFakeBackend("shared")
	alpha.name = "alpha"
	beta := newFakeBackend("shared")
	beta.name = "beta"
	defer alpha.close()
	defer beta.close()

	f := fed.New(fed.Config{NodeName: "head", HubURL: "http://hub"}, log.New(io.Discard, "", 0))
	f.ApplyRoster([]fed.RosterEntry{
		{Name: "alpha", MeshAddr: alpha.srv.URL, Models: []string{"shared"}},
		{Name: "beta", MeshAddr: beta.srv.URL, Models: []string{"shared"}},
	}, "head")

	front := httptest.NewServer(newRouter(t, f).Handler())
	defer front.Close()

	// default resolution → first-listed peer (alpha)
	resp := chatReq(t, front.URL, "shared")
	var got map[string]any
	json.NewDecoder(resp.Body).Decode(&got)
	resp.Body.Close()
	if got["served_by"] != "alpha" {
		t.Fatalf("default route should hit the first peer, got %+v", got)
	}

	// ?node=beta → pinned to beta despite the name clash
	resp2, err := http.Post(front.URL+"/v1/chat/completions?node=beta", "application/json",
		strings.NewReader(`{"model":"shared","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	var got2 map[string]any
	json.NewDecoder(resp2.Body).Decode(&got2)
	if got2["served_by"] != "beta" {
		t.Fatalf("?node=beta should pin to beta, got %+v", got2)
	}
	if beta.gotQuery != "" { // node param must be stripped before forwarding (no double-hop)
		t.Fatalf("forwarded request should carry no query, got %q", beta.gotQuery)
	}
}
