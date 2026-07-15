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

	mu       sync.Mutex
	state    State
	restarts int
	lastErr  string
	tail     *ring
	cmd      *exec.Cmd
	exitCh   chan error // the current process's Wait result (recreated each launch)
	stopping bool
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
}

func (in *Instance) snapshot() Status {
	in.mu.Lock()
	defer in.mu.Unlock()
	par := in.Slot.Parallel
	if par < 1 {
		par = 1
	}
	return Status{
		Name: in.Slot.Name(), Model: in.Model.ID, GPUs: in.Slot.GPUs, Port: in.Port,
		Ctx: in.Slot.CtxSize, Parallel: par, State: in.state, Restarts: in.restarts, LastErr: in.lastErr,
		SizeGB: float64(in.Model.Bytes) / (1 << 30),
	}
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
	in.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
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
		go r.supervise(in)
	}
}

// StopAll terminates every instance (no restart).
func (r *Rig) StopAll() {
	for _, in := range r.instances {
		in.mu.Lock()
		in.stopping = true
		cmd := in.cmd
		in.mu.Unlock()
		if cmd != nil && cmd.Process != nil {
			_ = cmd.Process.Kill()
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
	in.mu.Unlock()
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
	a = append(a, "--jinja") // OpenAI tool-call / chat-template support
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
