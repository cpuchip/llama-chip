package rig

import (
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cpuchip/llama-chip/internal/config"
	"github.com/cpuchip/llama-chip/internal/models"
)

// argFor returns the value that follows flag in the arg list (or "" if absent).
func argFor(args []string, flag string) string {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

func TestGpuListEmptyIsCPU(t *testing.T) {
	// gpuList([]) must be "" so CUDA_VISIBLE_DEVICES="" hides every card (CPU semantics).
	if got := gpuList(nil); got != "" {
		t.Fatalf("gpuList(nil)=%q want empty (CPU: hide all GPUs)", got)
	}
	if got := gpuList([]int{}); got != "" {
		t.Fatalf("gpuList([])=%q want empty", got)
	}
	if got := gpuList([]int{0, 1}); got != "0,1" {
		t.Fatalf("gpuList([0,1])=%q want 0,1", got)
	}
}

func TestArgsCPUSlotOffloadsZeroLayers(t *testing.T) {
	r := &Rig{}
	in := &Instance{
		Slot:  config.Slot{Model: "m", Alias: "cpu-model", GPUs: nil, NoMMProj: true, KVCache: "f16"},
		Model: models.Model{ID: "m", Path: "/nonexistent/m.gguf"},
		Port:  9100,
	}
	args := r.args(in)
	if got := argFor(args, "--n-gpu-layers"); got != "0" {
		t.Fatalf("CPU slot --n-gpu-layers=%q want 0 (no GPU offload); args=%v", got, args)
	}
	if got := argFor(args, "--split-mode"); got != "none" {
		t.Fatalf("CPU slot --split-mode=%q want none", got)
	}
}

func TestArgsGPUSlotOffloadsAllLayers(t *testing.T) {
	r := &Rig{}
	in := &Instance{
		Slot:  config.Slot{Model: "m", Alias: "gpu-model", GPUs: []int{0}, NoMMProj: true, KVCache: "f16"},
		Model: models.Model{ID: "m", Path: "/nonexistent/m.gguf"},
		Port:  9100,
	}
	args := r.args(in)
	if got := argFor(args, "--n-gpu-layers"); got != "999" {
		t.Fatalf("GPU slot --n-gpu-layers=%q want 999; args=%v", got, args)
	}
	// a single-GPU slot uses split-mode none too; sanity-check the model flag is present
	if !strings.Contains(strings.Join(args, " "), "--model") {
		t.Fatalf("args missing --model: %v", args)
	}
}

// An external slot is a server the rig does not launch: it is registered under its alias, its
// /health is mirrored into the slot state, its bearer comes from api_key_env over api_key, and
// unloading it stops the poller without touching the upstream.
func TestExternalSlotHealthAndBearer(t *testing.T) {
	healthy := true
	var mu sync.Mutex
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		ok := healthy
		mu.Unlock()
		if r.URL.Path != "/health" || !ok {
			http.Error(w, "nope", 503)
			return
		}
		w.WriteHeader(200)
	}))
	defer up.Close()

	r := &Rig{byName: map[string]*Instance{}, log: log.New(io.Discard, "", 0)}
	if err := r.Load(config.Slot{External: up.URL, GPUs: []int{0}}); err == nil {
		t.Fatal("an external slot without an alias must be refused")
	}
	t.Setenv("LLAMACHIP_TEST_KEY", "from-env")
	if err := r.Load(config.Slot{Alias: "ext-model", External: up.URL + "/", GPUs: []int{0}, APIKey: "from-file", APIKeyEnv: "LLAMACHIP_TEST_KEY"}); err != nil {
		t.Fatal(err)
	}
	in, ok := r.Resolve("ext-model")
	if !ok || in.External == nil || in.External.String() != up.URL {
		t.Fatalf("Resolve: ok=%v external=%v want %s", ok, in.External, up.URL)
	}
	if got := in.Bearer(); got != "from-env" {
		t.Fatalf("Bearer()=%q want the env var over api_key", got)
	}
	waitState := func(want State) {
		t.Helper()
		deadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(deadline) {
			if in.snapshot().State == want {
				return
			}
			time.Sleep(20 * time.Millisecond)
		}
		t.Fatalf("slot never reached %s (state %s, err %q)", want, in.snapshot().State, in.snapshot().LastErr)
	}
	waitState(Healthy)
	if st := in.snapshot(); st.External != up.URL || st.Model != "ext-model" {
		t.Fatalf("snapshot %+v", st)
	}
	mu.Lock()
	healthy = false
	mu.Unlock()
	// the poller runs every 5 s; nudge it by waiting for the next poll
	deadline := time.Now().Add(7 * time.Second)
	for time.Now().Before(deadline) && in.snapshot().State != Crashed {
		time.Sleep(50 * time.Millisecond)
	}
	if st := in.snapshot(); st.State != Crashed || !strings.Contains(st.LastErr, "503") {
		t.Fatalf("after the upstream failed: state %s err %q", st.State, st.LastErr)
	}
	if err := r.Unload("ext-model"); err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Resolve("ext-model"); ok {
		t.Fatal("still resolvable after unload")
	}
}
