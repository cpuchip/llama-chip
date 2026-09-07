package telemetry

import (
	"testing"
	"time"

	"github.com/cpuchip/llama-chip/internal/gpu"
	"github.com/cpuchip/llama-chip/internal/rig"
)

func TestComputeOursForeign(t *testing.T) {
	gpus := []gpu.GPU{
		{Index: 0, UUID: "GPU-a", MemUsed: 23664, MemTotal: 24564},
		{Index: 1, UUID: "GPU-b", MemUsed: 21000, MemTotal: 24564},
	}
	procs := []gpu.ProcUsage{
		{PID: 4242, GPUIndex: 1, MiB: 15100}, // our llama-server on card 1
		{PID: 9999, GPUIndex: 1, MiB: 5800},  // a game on card 1
		{PID: 42832, GPUIndex: 0, MiB: 23664}, // the WSL VM holding vLLM (not our PID)
	}
	statuses := []rig.Status{
		{Name: "gguf", Kind: "llama-server", Loaded: true, PID: 4242, GPUs: []int{1}},
		{Name: "vllm", Kind: "external", Loaded: true, GPUs: []int{0}, BaselineMiB: map[int]int{0: 23600}},
	}
	views := compute(gpus, procs, statuses, map[int]bool{1: true})
	if len(views) != 2 {
		t.Fatalf("views %d", len(views))
	}
	c0, c1 := views[0], views[1]
	if c0.OursMiB != 23600 || c0.ForeignMiB != 64 || len(c0.ForeignPIDs) != 1 {
		t.Fatalf("card 0 = %+v", c0)
	}
	if c1.OursMiB != 15100 || c1.ForeignMiB != 5900 || len(c1.ForeignPIDs) != 1 || c1.ForeignPIDs[0] != 9999 || !c1.Yielding {
		t.Fatalf("card 1 = %+v", c1)
	}
	if c1.MemFree != 24564-21000 {
		t.Fatalf("free derived wrong: %d", c1.MemFree)
	}
}

func TestExternalCardUnknownUntilBaseline(t *testing.T) {
	// A card carrying an external slot with no baseline yet must report foreign 0 (unknown),
	// not the whole card as foreign — the startup race that tripped a spurious yield on card 0.
	gpus := []gpu.GPU{{Index: 0, UUID: "GPU-a", MemUsed: 23664, MemTotal: 24564}}
	statuses := []rig.Status{{Name: "vllm", Kind: "external", Loaded: true, GPUs: []int{0}}} // BaselineMiB nil
	v := compute(gpus, nil, statuses, nil)[0]
	if v.ForeignMiB != 0 || v.OursMiB != 23664 {
		t.Fatalf("pre-baseline external card should read foreign 0 / ours all: %+v", v)
	}
	// Once the baseline is captured, attribution is normal again.
	statuses[0].BaselineMiB = map[int]int{0: 23600}
	v = compute(gpus, nil, statuses, nil)[0]
	if v.OursMiB != 23600 || v.ForeignMiB != 64 {
		t.Fatalf("post-baseline attribution wrong: %+v", v)
	}
}

func TestStatsView(t *testing.T) {
	s := NewStats()
	s.Observe("x", Obs{TTFT: 100 * time.Millisecond, Dur: 1100 * time.Millisecond, PromptTok: 1000, ComplTok: 100, CachedTok: 800, Status: 200})
	s.Observe("x", Obs{TTFT: 300 * time.Millisecond, Dur: 1300 * time.Millisecond, PromptTok: 1000, ComplTok: 200, CachedTok: 0, Status: 200})
	s.Observe("x", Obs{Status: 502})
	req, errs, hit, tokS, ttft := s.View("x")
	if req != 3 || errs != 1 {
		t.Fatalf("req %d errs %d", req, errs)
	}
	if hit < 0.39 || hit > 0.41 {
		t.Fatalf("cache hit %.3f", hit)
	}
	if tokS < 149 || tokS > 151 { // (100/1.0 + 200/1.0) / 2
		t.Fatalf("tok/s %.1f", tokS)
	}
	if ttft != 200 {
		t.Fatalf("ttft %.0f", ttft)
	}
	if r, _, _, _, _ := s.View("nope"); r != 0 {
		t.Fatal("unknown slot must be empty")
	}
}

func TestParseUsageTail(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":25,\"completion_tokens\":50,\"prompt_tokens_details\":{\"cached_tokens\":20}}}\n\ndata: [DONE]\n"
	u, ok := ParseUsageTail([]byte(sse))
	if !ok || u.PromptTok != 25 || u.ComplTok != 50 || u.CachedTok != 20 {
		t.Fatalf("sse usage = %+v ok=%v", u, ok)
	}
	llama := `{"choices":[{"message":{"content":"x"}}],"timings":{"prompt_n":10,"predicted_n":30,"cache_n":90,"predicted_per_second":41.2}}`
	u, ok = ParseUsageTail([]byte(llama))
	if !ok || u.PromptTok != 100 || u.ComplTok != 30 || u.CachedTok != 90 {
		t.Fatalf("llama usage = %+v ok=%v", u, ok)
	}
	anthropic := `{"type":"message","usage":{"input_tokens":40,"output_tokens":7,"cache_read_input_tokens":32}}`
	u, ok = ParseUsageTail([]byte(anthropic))
	if !ok || u.PromptTok != 40 || u.ComplTok != 7 || u.CachedTok != 32 {
		t.Fatalf("anthropic usage = %+v ok=%v", u, ok)
	}
	// A tail that lost its head still yields the usage block.
	torn := `"content":"...long body..."}}],"usage":{"prompt_tokens":5,"completion_tokens":6}}`
	u, ok = ParseUsageTail([]byte(torn))
	if !ok || u.PromptTok != 5 || u.ComplTok != 6 {
		t.Fatalf("torn usage = %+v ok=%v", u, ok)
	}
	if _, ok := ParseUsageTail([]byte("not json at all")); ok {
		t.Fatal("garbage must not parse")
	}
}

func TestParseSlotsAndDeferred(t *testing.T) {
	inflight, cache, err := parseSlots([]byte(`[{"id":0,"is_processing":true,"n_prompt_tokens_cache":120},{"id":1,"is_processing":false,"n_prompt_tokens_cache":9}]`))
	if err != nil || inflight != 1 || cache != 120 {
		t.Fatalf("slots: %d %d %v", inflight, cache, err)
	}
	if q := parseDeferred("# HELP x\nllamacpp:requests_processing 1\nllamacpp:requests_deferred 3\n"); q != 3 {
		t.Fatalf("deferred %d", q)
	}
}
