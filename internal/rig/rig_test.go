package rig

import (
	"strings"
	"testing"

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
