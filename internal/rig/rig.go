// Package rig supervises llama-server processes: one per model slot, pinned to its
// GPU(s) via CUDA_VISIBLE_DEVICES, health-checked, and AUTO-RESTARTED on crash — the
// thing the dockerized wrapper lacked (its runner died and stayed dead).
package rig

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cpuchip/llama-chip/internal/backends"
	"github.com/cpuchip/llama-chip/internal/config"
	"github.com/cpuchip/llama-chip/internal/gguf"
	"github.com/cpuchip/llama-chip/internal/gpu"
	"github.com/cpuchip/llama-chip/internal/models"
)

type State string

const (
	Starting State = "starting"
	Healthy  State = "healthy"
	Crashed  State = "crashed"
	Stopped  State = "stopped"
)

// Instance is one supervised llama-server.
type Instance struct {
	Slot    config.Slot
	Model   models.Model
	Backend backends.Backend
	Port    int
	// External is set for a slot the rig does not launch: the upstream's root URL. The router
	// proxies there (with the slot's bearer) and the supervisor only polls its /health.
	External *url.URL

	mu       sync.Mutex
	state    State
	restarts int
	lastErr  string
	tail     *ring
	cmd      *exec.Cmd
	exitCh   chan error // the current process's Wait result (recreated each launch)
	stopping bool
	pid      int         // the launched llama-server's PID (0 when not running / external)
	baseline map[int]int // external slots: each pinned card's mem_used (MiB) when the upstream first went healthy
	docker   string      // container slots: the docker container name the rig launched (removed on unload)

	vramOnce sync.Once
	vramEst  int // MiB the slot is expected to hold: weights + KV at its context + overhead (0 = unknown)
}

// VRAMEstimate is the slot's expected VRAM in MiB: the model file plus the KV cache at the slot's
// context (from the GGUF's real attention dims, the same math as /api/guess-context) plus a fixed
// compute overhead. 0 for external slots and unparseable models. Computed once per instance.
func (in *Instance) VRAMEstimate() int {
	in.vramOnce.Do(func() {
		if in.External != nil || in.Model.Bytes == 0 {
			return
		}
		est := float64(in.Model.Bytes)/(1<<20) + 700
		if in.Slot.CtxSize > 0 {
			bytesPerElem := 1.0625 // q8_0
			switch in.Slot.KVCache {
			case "f16", "":
				bytesPerElem = 2.0
			case "q4_0":
				bytesPerElem = 0.5625
			}
			if p, err := gguf.Read(in.Model.Path); err == nil {
				if b := p.KVBytesPerToken(bytesPerElem); b > 0 {
					est += b * float64(in.Slot.CtxSize) / (1 << 20)
				}
			}
		}
		in.vramEst = int(est)
	})
	return in.vramEst
}

func (in *Instance) isStopping() bool {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.stopping
}

// Status is a snapshot for the API/UI.
type Status struct {
	Name     string  `json:"name"`
	Model    string  `json:"model"`
	GPUs     []int   `json:"gpus"`
	Port     int     `json:"port"`
	Ctx      int     `json:"ctx_size"` // TOTAL context across slots; per-request = Ctx / Parallel
	Parallel int     `json:"parallel"` // concurrent request slots (--parallel); 1 = single
	State    State   `json:"state"`
	Restarts int     `json:"restarts"`
	LastErr  string  `json:"last_err,omitempty"`
	SizeGB   float64 `json:"size_gb"`
	External string  `json:"external,omitempty"` // the upstream root URL for a slot the rig does not launch

	// The slot contract (fleet): what a peer or a scheduler needs to place work.
	Kind        string      `json:"kind"`                   // "llama-server" | "external"
	Loaded      bool        `json:"loaded"`                 // healthy and answering
	PID         int         `json:"pid,omitempty"`          // the llama-server process (attribution of VRAM)
	VRAMEstMiB  int         `json:"vram_est_mib,omitempty"` // expected VRAM: weights + KV at ctx + overhead
	BaselineMiB map[int]int `json:"baseline_mib,omitempty"` // external: card -> mem_used when first healthy
}

func (in *Instance) snapshot() Status {
	in.mu.Lock()
	defer in.mu.Unlock()
	par := in.Slot.Parallel
	if par < 1 {
		par = 1
	}
	st := Status{
		Name: in.Slot.Name(), Model: in.Model.ID, GPUs: in.Slot.GPUs, Port: in.Port,
		Ctx: in.Slot.CtxSize, Parallel: par, State: in.state, Restarts: in.restarts, LastErr: in.lastErr,
		SizeGB: float64(in.Model.Bytes) / (1 << 30),
		Kind:   "llama-server", Loaded: in.state == Healthy, PID: in.pid,
	}
	if in.External != nil {
		st.External = in.External.String()
		st.Kind = "external"
		if in.docker != "" {
			st.Kind = "container"
		}
		if len(in.baseline) > 0 {
			st.BaselineMiB = make(map[int]int, len(in.baseline))
			for k, v := range in.baseline {
				st.BaselineMiB[k] = v
			}
		}
	} else {
		st.VRAMEstMiB = in.vramEst // filled by VRAMEstimate(); read under the lock without recomputing
	}
	return st
}

func (in *Instance) set(s State, errMsg string) {
	in.mu.Lock()
	in.state = s
	if errMsg != "" {
		in.lastErr = errMsg
	}
	in.mu.Unlock()
}

// Rig is the supervised set of instances.
type Rig struct {
	cfg       *config.Config
	backend   backends.Backend
	basePort  int
	mu        sync.RWMutex
	instances []*Instance
	byName    map[string]*Instance
	log       *log.Logger
}

// Load adds a slot at runtime and starts supervising it (the dynamic-management primitive).
// An empty GPUs ([] or omitted) is a CPU-only slot: CUDA_VISIBLE_DEVICES="" hides every card and
// the backend runs on CPU (the mode the GPU-less NOCIX node uses to keep the federation alive).
func (r *Rig) Load(s config.Slot) error {
	if s.Container != nil {
		return r.loadContainer(s)
	}
	if s.External != "" {
		return r.loadExternal(s)
	}
	if s.Model == "" {
		return fmt.Errorf("load needs a model")
	}
	if s.KVCache == "" {
		s.KVCache = "q8_0"
	}
	if s.SplitMode == "" {
		s.SplitMode = "layer"
	}
	m, ok := models.Find(s.Model)
	if !ok {
		return fmt.Errorf("no single model matches %q", s.Model)
	}
	be, err := r.backendFor(s)
	if err != nil {
		return fmt.Errorf("slot %q: %w", s.Name(), err)
	}
	r.mu.Lock()
	key := strings.ToLower(s.Name())
	if _, exists := r.byName[key]; exists {
		r.mu.Unlock()
		return fmt.Errorf("slot %q already loaded", s.Name())
	}
	if s.Port == 0 {
		s.Port = r.nextPortLocked()
	}
	in := &Instance{Slot: s, Model: m, Backend: be, Port: s.Port, state: Stopped, tail: newRing(60)}
	r.instances = append(r.instances, in)
	r.byName[key] = in
	r.mu.Unlock()
	r.log.Printf("[%s] loading %s on gpus %v", s.Name(), m.ID, s.GPUs)
	go r.supervise(in)
	return nil
}

// Unload stops and removes a slot at runtime.
func (r *Rig) Unload(name string) error {
	r.mu.Lock()
	key := strings.ToLower(name)
	in, ok := r.byName[key]
	if !ok {
		r.mu.Unlock()
		return fmt.Errorf("no slot %q loaded", name)
	}
	delete(r.byName, key)
	for i, x := range r.instances {
		if x == in {
			r.instances = append(r.instances[:i], r.instances[i+1:]...)
			break
		}
	}
	r.mu.Unlock()
	in.mu.Lock()
	in.stopping = true
	cmd := in.cmd
	container := in.docker
	in.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
	if container != "" {
		r.removeContainer(in, container)
	}
	in.set(Stopped, "")
	r.log.Printf("[%s] unloaded", name)
	return nil
}

// UnloadAll unloads every slot, freeing all VRAM. Returns the names unloaded.
// Used by the "free GPUs" control so the box can be handed to other work (games).
func (r *Rig) UnloadAll() []string {
	r.mu.Lock()
	names := make([]string, 0, len(r.instances))
	for _, in := range r.instances {
		names = append(names, in.Slot.Name())
	}
	r.mu.Unlock()
	for _, n := range names {
		_ = r.Unload(n)
	}
	return names
}

// nextPortLocked returns a free backend port (caller holds r.mu).
func (r *Rig) nextPortLocked() int {
	p := r.basePort
	used := map[int]bool{}
	for _, in := range r.instances {
		used[in.Port] = true
	}
	for used[p] {
		p++
	}
	return p
}

// ProfileNames lists the configured profiles (named GPU layouts).
func (r *Rig) ProfileNames() []string {
	names := make([]string, 0, len(r.cfg.Profiles))
	for n := range r.cfg.Profiles {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// ApplyProfile unloads everything and loads the named profile's slots — an atomic-ish
// layout switch (e.g. one big split model -> two diverse models, one per card).
func (r *Rig) ApplyProfile(name string) error {
	slots, ok := r.cfg.Profiles[name]
	if !ok {
		return fmt.Errorf("no profile %q (have: %s)", name, strings.Join(r.ProfileNames(), ", "))
	}
	r.mu.RLock()
	current := make([]string, 0, len(r.instances))
	for _, in := range r.instances {
		current = append(current, in.Slot.Name())
	}
	r.mu.RUnlock()
	for _, n := range current {
		_ = r.Unload(n)
	}
	var errs []string
	for _, s := range slots {
		if err := r.Load(s); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("profile %q loaded with errors: %s", name, strings.Join(errs, "; "))
	}
	r.log.Printf("applied profile %q (%d slots)", name, len(slots))
	return nil
}

// Ensure loads a slot if no healthy slot already serves the model — the substrate's
// load-by-need primitive. It first unloads any slot pinned to an overlapping GPU, so a new
// model swaps in cleanly for a work phase. Returns (slotName, action).
func (r *Rig) Ensure(s config.Slot) (string, string, error) {
	if in, ok := r.Resolve(s.Model); ok && in.snapshot().State == Healthy {
		return in.Slot.Name(), "already-loaded", nil
	}
	want := map[int]bool{}
	for _, g := range s.GPUs {
		want[g] = true
	}
	r.mu.RLock()
	var evict []string
	for _, in := range r.instances {
		for _, g := range in.Slot.GPUs {
			if want[g] {
				evict = append(evict, in.Slot.Name())
				break
			}
		}
	}
	r.mu.RUnlock()
	for _, n := range evict {
		_ = r.Unload(n)
	}
	if err := r.Load(s); err != nil {
		return "", "", err
	}
	return s.Name(), "loaded", nil
}

// New resolves the backend + every slot's model and prepares (does not start) the rig.
// DefaultMaxTokens is the config cap the router applies to capless completions (0 = none).
func (r *Rig) DefaultMaxTokens() int {
	if r == nil || r.cfg == nil {
		return 0
	}
	return r.cfg.DefaultMaxTokens
}

func New(cfg *config.Config, logger *log.Logger) (*Rig, error) {
	if logger == nil {
		logger = log.New(os.Stderr, "", log.LstdFlags)
	}
	be, err := resolveBackend(cfg.Backend)
	if err != nil {
		return nil, err
	}
	logger.Printf("backend: %s %s (%s)", be.Variant, be.Version, be.Server)
	r := &Rig{cfg: cfg, backend: be, basePort: cfg.BasePort, byName: map[string]*Instance{}, log: logger}
	for _, s := range cfg.Slots {
		if s.External != "" {
			if _, err := r.addExternal(s, Stopped); err != nil {
				return nil, fmt.Errorf("slot %q: %w", s.Name(), err)
			}
			continue
		}
		m, ok := models.Find(s.Model)
		if !ok {
			return nil, fmt.Errorf("slot %q: no single model matches %q (run `llama-chip models`)", s.Name(), s.Model)
		}
		sbe, err := r.backendFor(s)
		if err != nil {
			return nil, fmt.Errorf("slot %q: %w", s.Name(), err)
		}
		if s.Backend != "" {
			logger.Printf("slot %q backend override: %s (%s)", s.Name(), sbe.Variant, sbe.Server)
		}
		in := &Instance{Slot: s, Model: m, Backend: sbe, Port: s.Port, state: Stopped, tail: newRing(60)}
		r.instances = append(r.instances, in)
		r.byName[strings.ToLower(s.Name())] = in
	}
	return r, nil
}

// backendFor resolves a slot's backend: its per-slot override (a variant name or an
// explicit dir holding a self-managed llama-server) if set, else the rig-global backend.
// This is what lets the stable models stay on LM Studio's vetted build while one slot
// runs a fresh ggml-org llama-server for a bleeding-edge arch.
func (r *Rig) backendFor(s config.Slot) (backends.Backend, error) {
	if s.Backend == "" {
		return r.backend, nil
	}
	return resolveBackend(s.Backend)
}

func resolveBackend(spec string) (backends.Backend, error) {
	// a self-managed ggml-org build llama-chip pulled itself: "ggml@latest" | "ggml@bNNNN"
	if rest, ok := strings.CutPrefix(spec, "ggml@"); ok {
		return backends.ResolveGGML(rest)
	}
	// explicit directory containing the llama-server binary
	if strings.ContainsAny(spec, "/\\") {
		srv := spec + string(os.PathSeparator) + backends.ServerBinName()
		if _, err := os.Stat(srv); err == nil {
			return backends.Backend{Variant: backends.Variant("custom"), Dir: spec, Server: srv, Complete: true}, nil
		}
		return backends.Backend{}, fmt.Errorf("no %s in %s", backends.ServerBinName(), spec)
	}
	if b, ok := backends.Latest(backends.Variant(spec)); ok {
		return b, nil
	}
	return backends.Backend{}, fmt.Errorf("no runnable %q backend found — open LM Studio and let it download one, or set an explicit dir", spec)
}

// Start launches every slot and begins supervising.
func (r *Rig) Start() {
	for _, in := range r.instances {
		if in.External != nil {
			go r.superviseExternal(in)
			continue
		}
		go r.supervise(in)
	}
}

// StopAll terminates every instance (no restart).
func (r *Rig) StopAll() {
	for _, in := range r.instances {
		in.mu.Lock()
		in.stopping = true
		cmd := in.cmd
		container := in.docker
		in.mu.Unlock()
		if cmd != nil && cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		if container != "" { // a lab container must not outlive the rig holding its card
			r.removeContainer(in, container)
		}
		in.set(Stopped, "")
	}
}

// Resolve finds the instance serving a model handle (by slot name or model id/substring).
func (r *Rig) Resolve(model string) (*Instance, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if in, ok := r.byName[strings.ToLower(model)]; ok {
		return in, true
	}
	for _, in := range r.instances { // fall back to model id / substring
		if strings.EqualFold(in.Model.ID, model) || strings.Contains(strings.ToLower(in.Model.ID), strings.ToLower(model)) {
			return in, true
		}
	}
	return nil, false
}

// Snapshot returns the status of every instance.
func (r *Rig) Snapshot() []Status {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Status, 0, len(r.instances))
	for _, in := range r.instances {
		out = append(out, in.snapshot())
	}
	return out
}

// supervise runs an instance's launch→watch→restart loop until StopAll.
func (r *Rig) supervise(in *Instance) {
	backoff := time.Second
	for {
		if in.isStopping() {
			return
		}
		if err := r.launch(in); err != nil {
			in.set(Crashed, err.Error())
			r.log.Printf("[%s] not ready: %v (retry in %s)", in.Slot.Name(), err, backoff)
		} else {
			backoff = time.Second // a clean start resets backoff
			in.set(Healthy, "")
			r.log.Printf("[%s] healthy on :%d (gpus %v)", in.Slot.Name(), in.Port, in.Slot.GPUs)
			err := <-in.exitCh // block until the process exits
			if in.isStopping() {
				return
			}
			in.restarts++
			in.set(Crashed, fmt.Sprintf("exited: %v — %s", err, in.tail.last()))
			r.log.Printf("[%s] CRASHED (%v) — %s — restarting (#%d) in %s", in.Slot.Name(), err, in.tail.last(), in.restarts, backoff)
		}
		if in.isStopping() {
			return
		}
		time.Sleep(backoff)
		if backoff < 30*time.Second {
			backoff *= 2
		}
	}
}

// launch starts the llama-server process and waits for /health (or an early exit).
func (r *Rig) launch(in *Instance) error {
	in.set(Starting, "")
	// Fail fast if something else already holds the slot's port — otherwise we pay a
	// full model load just to die on bind, and the crash loop looks like a model/GPU
	// problem. (A stale Flutter dart.exe squatting :9100 cost hours this way.)
	if ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", in.Port)); err != nil {
		return fmt.Errorf("port %d already in use by another process — llama-server cannot bind (find the holder before retrying)", in.Port)
	} else {
		ln.Close()
	}
	cmd := exec.Command(in.Backend.Server, r.args(in)...)
	libPath := strings.Join(in.Backend.DLLDirs(), string(os.PathListSeparator))
	// The runtime libs (ggml-cuda, cudart/cublas) are found differently per OS: Windows
	// resolves DLLs via PATH; Linux/macOS resolve .so/.dylib via LD_LIBRARY_PATH /
	// DYLD_LIBRARY_PATH (sibling libs also via the binary's $ORIGIN rpath, hence cmd.Dir).
	cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES="+gpuList(in.Slot.GPUs))
	sep := string(os.PathListSeparator)
	switch runtime.GOOS {
	case "windows":
		cmd.Env = append(cmd.Env, "PATH="+libPath+sep+os.Getenv("PATH"))
	case "darwin":
		cmd.Env = append(cmd.Env, "DYLD_LIBRARY_PATH="+libPath+sep+os.Getenv("DYLD_LIBRARY_PATH"))
	default:
		cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+libPath+sep+os.Getenv("LD_LIBRARY_PATH"))
	}
	cmd.Dir = in.Backend.Dir // so the loader also finds sibling libs next to the binary
	// Reap this child if the supervisor dies abruptly (SIGKILL / crash) — belt to the
	// signal-handler's braces, so a dead llama-chip never orphans a llama-server holding
	// GPU VRAM + its backend port. (Linux: PR_SET_PDEATHSIG; no-op elsewhere.)
	setChildReaping(cmd)
	stderr, _ := cmd.StderrPipe()
	stdout, _ := cmd.StdoutPipe()
	if err := cmd.Start(); err != nil {
		return err
	}
	exitCh := make(chan error, 1)
	in.mu.Lock()
	in.cmd = cmd
	in.exitCh = exitCh
	in.pid = cmd.Process.Pid
	in.mu.Unlock()
	in.VRAMEstimate() // parse the GGUF once, off the snapshot path
	go in.drain(stderr)
	go in.drain(stdout)
	go func() { exitCh <- cmd.Wait() }() // the single Wait — populates exit + lets us notice a crash

	deadline := time.Now().Add(4 * time.Minute) // big-context model loads can be slow
	url := fmt.Sprintf("http://127.0.0.1:%d/health", in.Port)
	for time.Now().Before(deadline) {
		if in.isStopping() {
			return fmt.Errorf("stopping")
		}
		select {
		case err := <-exitCh: // died before becoming healthy — surface its output
			return fmt.Errorf("exited during load (%v) — %s", err, in.tail.last())
		default:
		}
		if resp, err := http.Get(url); err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
		time.Sleep(time.Second)
	}
	return fmt.Errorf("health timeout — %s", in.tail.last())
}

// args builds the llama-server command line for a slot.
func (r *Rig) args(in *Instance) []string {
	s := in.Slot
	// CPU-only slot (no GPUs pinned): offload zero layers to the GPU. Belt-and-braces with
	// CUDA_VISIBLE_DEVICES="" (which already hides every card) — and correct on a CPU-only
	// backend build that has no GPU offload at all.
	ngl := "999"
	if len(s.GPUs) == 0 {
		ngl = "0"
	}
	a := []string{
		"--model", in.Model.Path,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(in.Port),
		"--alias", s.Name(),
		"--n-gpu-layers", ngl,
	}
	// Multimodal: load the vision/audio projector so the model can accept images.
	// An explicit slot.MMProj wins; otherwise auto-detect the co-located mmproj-*.gguf
	// next to the model GGUF, unless the slot opted out with no_mmproj.
	if s.MMProj != "" {
		a = append(a, "--mmproj", s.MMProj)
	} else if !s.NoMMProj {
		if mp, ok := models.FindMMProj(in.Model.Path); ok {
			a = append(a, "--mmproj", mp)
		}
	}
	if s.CtxSize > 0 {
		// --ctx-size is the TOTAL context across all parallel slots (standard llama.cpp);
		// per-request context = CtxSize / Parallel. (Verified empirically 2026-06-21 by VRAM.)
		a = append(a, "--ctx-size", strconv.Itoa(s.CtxSize))
	}
	par := s.Parallel
	if par < 1 {
		par = 1 // default to 1 (this build otherwise defaults --parallel to 4)
	}
	a = append(a, "--parallel", strconv.Itoa(par))
	if s.NoFlash {
		a = append(a, "--flash-attn", "off")
	} else {
		a = append(a, "--flash-attn", "on") // value form; required, and KV-cache quantization needs FA on
	}
	if s.KVCache != "" && s.KVCache != "f16" {
		a = append(a, "--cache-type-k", s.KVCache, "--cache-type-v", s.KVCache)
	}
	if len(s.GPUs) > 1 {
		a = append(a, "--split-mode", s.SplitMode)
	} else {
		a = append(a, "--split-mode", "none")
	}
	a = append(a, "--jinja")   // OpenAI tool-call / chat-template support
	a = append(a, "--metrics") // /metrics: requests processing + deferred, for the slot contract's queue depth
	a = append(a, s.ExtraArgs...)
	return a
}

func (in *Instance) drain(rc io.Reader) {
	sc := bufio.NewScanner(rc)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		in.tail.push(line)
	}
}

func gpuList(gpus []int) string {
	parts := make([]string, len(gpus))
	for i, g := range gpus {
		parts[i] = strconv.Itoa(g)
	}
	return strings.Join(parts, ",")
}

// ring is a tiny fixed-size line buffer for an instance's recent output.
type ring struct {
	mu    sync.Mutex
	lines []string
	n     int
}

func newRing(n int) *ring { return &ring{lines: make([]string, 0, n), n: n} }
func (r *ring) push(s string) {
	r.mu.Lock()
	if len(r.lines) >= r.n {
		r.lines = r.lines[1:]
	}
	r.lines = append(r.lines, s)
	r.mu.Unlock()
}
func (r *ring) last() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.lines) == 0 {
		return ""
	}
	from := len(r.lines) - 3
	if from < 0 {
		from = 0
	}
	return strings.Join(r.lines[from:], " | ")
}

// Tail returns the recent output lines (for the UI).
func (in *Instance) Tail() []string {
	in.tail.mu.Lock()
	defer in.tail.mu.Unlock()
	out := make([]string, len(in.tail.lines))
	copy(out, in.tail.lines)
	return out
}

// ---- external slots: an OpenAI-compatible server the rig does not launch (a vLLM container, say) ----

// addExternal registers an external slot without starting anything. Alias is the model name the
// upstream serves and is required (there is no GGUF to name it after).
func (r *Rig) addExternal(s config.Slot, st State) (*Instance, error) {
	if s.Alias == "" {
		return nil, fmt.Errorf("external slot needs an alias (the model name the upstream serves)")
	}
	u, err := url.Parse(strings.TrimRight(s.External, "/"))
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("external %q is not a URL like http://127.0.0.1:18020", s.External)
	}
	port, _ := strconv.Atoi(u.Port())
	r.mu.Lock()
	defer r.mu.Unlock()
	key := strings.ToLower(s.Name())
	if _, exists := r.byName[key]; exists {
		return nil, fmt.Errorf("slot %q already loaded", s.Name())
	}
	in := &Instance{Slot: s, Model: models.Model{ID: s.Alias, Name: s.Alias}, External: u, Port: port, state: st, tail: newRing(8)}
	r.instances = append(r.instances, in)
	r.byName[key] = in
	return in, nil
}

// loadExternal is Load for an external slot: register it and start the health poller.
func (r *Rig) loadExternal(s config.Slot) error {
	in, err := r.addExternal(s, Starting)
	if err != nil {
		return err
	}
	r.log.Printf("[%s] external upstream %s (not launched here; health-polled)", s.Name(), in.External)
	go r.superviseExternal(in)
	return nil
}

// Bearer is the Authorization token the router sends to an external upstream: the env var named
// by api_key_env when set, else api_key. Empty for a slot the rig launches itself.
func (in *Instance) Bearer() string {
	if in.Slot.APIKeyEnv != "" {
		if v := os.Getenv(in.Slot.APIKeyEnv); v != "" {
			return v
		}
	}
	return in.Slot.APIKey
}

// superviseExternal polls the upstream's /health every few seconds and mirrors it into the slot
// state: 200 = healthy, anything else = crashed (with the reason), until the slot is unloaded.
// There is nothing to restart: the server belongs to whoever launched it.
func (r *Rig) superviseExternal(in *Instance) {
	cl := &http.Client{Timeout: 3 * time.Second}
	target := in.External.String() + "/health"
	last := State("")
	for {
		if in.isStopping() {
			return
		}
		req, _ := http.NewRequest(http.MethodGet, target, nil)
		if b := in.Bearer(); b != "" {
			req.Header.Set("Authorization", "Bearer "+b)
		}
		resp, err := cl.Do(req)
		var st State
		var why string
		if err != nil {
			st, why = Crashed, "upstream unreachable: "+err.Error()
		} else {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				st = Healthy
			} else {
				st, why = Crashed, fmt.Sprintf("upstream /health returned %d", resp.StatusCode)
			}
		}
		if st != last {
			in.set(st, why)
			if st == Healthy {
				r.log.Printf("[%s] external upstream healthy (%s)", in.Slot.Name(), in.External)
				in.captureBaseline()
			} else {
				r.log.Printf("[%s] external upstream not ready: %s", in.Slot.Name(), why)
			}
			last = st
		}
		for i := 0; i < 50; i++ { // 5 s between polls, but leave promptly on unload
			if in.isStopping() {
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}
}

// NewEmptyForTest builds a rig with no backend and no slots, for tests that only need external
// slots (nothing is ever launched).
// captureBaseline records, for an external slot, how much of each pinned card is in use the
// first time the upstream answers healthy. The rig did not launch that process (a vLLM container
// is a WSL2 VM's memory on Windows) so it cannot attribute VRAM by PID; the card's usage at that
// moment stands in for "ours" and anything above it later counts as foreign. Taken once.
func (in *Instance) captureBaseline() {
	in.mu.Lock()
	have := in.baseline != nil
	in.mu.Unlock()
	if have {
		return
	}
	gs, err := gpu.Query()
	if err != nil {
		return
	}
	base := map[int]int{}
	for _, g := range gs {
		for _, idx := range in.Slot.GPUs {
			if g.Index == idx {
				base[idx] = g.MemUsed
			}
		}
	}
	in.mu.Lock()
	in.baseline = base
	in.mu.Unlock()
}

func NewEmptyForTest(logger *log.Logger) *Rig {
	if logger == nil {
		logger = log.New(io.Discard, "", 0)
	}
	return &Rig{byName: map[string]*Instance{}, log: logger}
}
