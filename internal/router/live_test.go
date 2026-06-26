package router

import "testing"

// the exact /slots shape probed from a live llama-server (qwen3.6-35b-a3b, dance-moe):
// one idle slot that just finished a request, one actively generating.
const sampleSlots = `[
  {"id":0,"id_task":684035,"is_processing":false,"n_ctx":122880,
   "n_prompt_tokens":26471,"n_prompt_tokens_cache":0,"n_prompt_tokens_processed":26329,
   "next_token":[{"has_new_line":false,"has_next_token":false,"n_decoded":143,"n_remain":5939}],"speculative":false},
  {"id":1,"id_task":684040,"is_processing":true,"n_ctx":122880,
   "n_prompt_tokens":51200,"n_prompt_tokens_cache":1024,"n_prompt_tokens_processed":51200,
   "next_token":[{"has_new_line":true,"has_next_token":true,"n_decoded":812,"n_remain":15188}],"speculative":false}
]`

func TestParseSlots(t *testing.T) {
	slots, parallel, ctxPerSlot, active := parseSlots([]byte(sampleSlots))
	if parallel != 2 {
		t.Fatalf("parallel: want 2, got %d", parallel)
	}
	if ctxPerSlot != 122880 {
		t.Fatalf("ctxPerSlot: want 122880, got %d", ctxPerSlot)
	}
	if active != 1 {
		t.Fatalf("active: want 1 (slot 1 processing), got %d", active)
	}
	// slot 0 — idle, finished: prompt 26471, generated 143
	if s := slots[0]; s.Processing || s.PromptTokens != 26471 || s.Decoded != 143 || s.Remain != 5939 {
		t.Fatalf("slot0 mismatch: %+v", s)
	}
	// slot 1 — busy: prompt 51200 (1024 cached), generating, 812 out so far
	if s := slots[1]; !s.Processing || s.PromptTokens != 51200 || s.PromptCache != 1024 || s.Decoded != 812 || s.Task != 684040 {
		t.Fatalf("slot1 mismatch: %+v", s)
	}
}

func TestParseSlotsEmptyOrGarbage(t *testing.T) {
	if s, p, c, a := parseSlots([]byte(`[]`)); len(s) != 0 || p != 0 || c != 0 || a != 0 {
		t.Fatalf("empty slots should parse to zeros, got %d/%d/%d/%d", len(s), p, c, a)
	}
	if s, _, _, _ := parseSlots([]byte(`not json`)); s != nil {
		t.Fatalf("garbage should yield nil slots, got %+v", s)
	}
}
