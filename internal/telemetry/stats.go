package telemetry

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"time"
)

// Obs is one proxied request as the router saw it.
type Obs struct {
	At        time.Time
	TTFT      time.Duration // first response byte
	Dur       time.Duration // whole response
	PromptTok int
	ComplTok  int
	CachedTok int // prompt tokens the backend reported as served from cache
	Status    int
}

const window = 50

type ring struct {
	obs      []Obs
	next     int
	requests int
	errors   int
}

// Stats keeps a rolling window of observations per slot.
type Stats struct {
	mu  sync.Mutex
	per map[string]*ring
}

func NewStats() *Stats { return &Stats{per: map[string]*ring{}} }

// Observe records one request against a slot name.
func (s *Stats) Observe(slot string, o Obs) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.per[slot]
	if r == nil {
		r = &ring{}
		s.per[slot] = r
	}
	r.requests++
	if o.Status >= 400 {
		r.errors++
	}
	if len(r.obs) < window {
		r.obs = append(r.obs, o)
	} else {
		r.obs[r.next] = o
	}
	r.next = (r.next + 1) % window
}

// View summarizes a slot's window: totals, cache hit rate, tokens per second, time to first token.
func (s *Stats) View(slot string) (requests, errors int, cacheHit, tokS, ttftMs float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.per[slot]
	if r == nil {
		return 0, 0, 0, 0, 0
	}
	var prompt, cached, nTok, nTTFT int
	var tokSum, ttftSum float64
	for _, o := range r.obs {
		if o.Status >= 400 {
			continue
		}
		prompt += o.PromptTok
		cached += o.CachedTok
		if o.ComplTok > 0 && o.Dur > o.TTFT {
			tokSum += float64(o.ComplTok) / (o.Dur - o.TTFT).Seconds()
			nTok++
		}
		if o.TTFT > 0 {
			ttftSum += float64(o.TTFT.Milliseconds())
			nTTFT++
		}
	}
	if prompt > 0 {
		cacheHit = float64(cached) / float64(prompt)
	}
	if nTok > 0 {
		tokS = tokSum / float64(nTok)
	}
	if nTTFT > 0 {
		ttftMs = ttftSum / float64(nTTFT)
	}
	return r.requests, r.errors, cacheHit, tokS, ttftMs
}

// Usage is what a response said about its own token accounting, in whichever dialect it spoke.
type Usage struct {
	PromptTok int
	ComplTok  int
	CachedTok int
}

// ParseUsageTail reads token usage from the end of a response body. Streams (SSE) carry it in
// the last data event that has one; JSON bodies carry it at the top level. Dialects: OpenAI
// (usage.prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens), llama-server
// (timings.prompt_n, predicted_n, cache_n; or tokens_evaluated, tokens_predicted, tokens_cached),
// Anthropic (usage.input_tokens, output_tokens, cache_read_input_tokens).
func ParseUsageTail(tail []byte) (Usage, bool) {
	trimmed := bytes.TrimSpace(tail)
	if bytes.Contains(trimmed, []byte("data:")) {
		var last Usage
		found := false
		for _, line := range bytes.Split(trimmed, []byte("\n")) {
			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, []byte("data:")) {
				continue
			}
			payload := bytes.TrimSpace(line[len("data:"):])
			if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
				continue
			}
			if u, ok := usageFromJSON(payload); ok {
				last, found = u, true
			}
		}
		return last, found
	}
	// A JSON body: the tail may have lost its head; find the last top-level object start if the
	// whole thing does not parse.
	if u, ok := usageFromJSON(trimmed); ok {
		return u, true
	}
	if i := bytes.LastIndex(trimmed, []byte(`"usage"`)); i >= 0 {
		// Rebuild a minimal object around the usage block.
		frag := trimmed[i:]
		if j := bytes.Index(frag, []byte("}")); j >= 0 {
			obj := append([]byte("{"), frag[:j+1]...)
			obj = append(obj, '}')
			if u, ok := usageFromJSON(obj); ok {
				return u, true
			}
		}
	}
	return Usage{}, false
}

func usageFromJSON(b []byte) (Usage, bool) {
	var doc struct {
		Usage *struct {
			PromptTokens    int `json:"prompt_tokens"`
			CompletionTok   int `json:"completion_tokens"`
			InputTokens     int `json:"input_tokens"`
			OutputTokens    int `json:"output_tokens"`
			CacheReadInput  int `json:"cache_read_input_tokens"`
			PromptTokDetail *struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
		Timings *struct {
			PromptN    int `json:"prompt_n"`
			PredictedN int `json:"predicted_n"`
			CacheN     int `json:"cache_n"`
		} `json:"timings"`
		TokensEvaluated int `json:"tokens_evaluated"`
		TokensPredicted int `json:"tokens_predicted"`
		TokensCached    int `json:"tokens_cached"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return Usage{}, false
	}
	var u Usage
	found := false
	if doc.Usage != nil {
		u.PromptTok = doc.Usage.PromptTokens + doc.Usage.InputTokens
		u.ComplTok = doc.Usage.CompletionTok + doc.Usage.OutputTokens
		u.CachedTok = doc.Usage.CacheReadInput
		if doc.Usage.PromptTokDetail != nil {
			u.CachedTok += doc.Usage.PromptTokDetail.CachedTokens
		}
		found = u.PromptTok > 0 || u.ComplTok > 0
	}
	if doc.Timings != nil && (doc.Timings.PromptN > 0 || doc.Timings.PredictedN > 0) {
		if u.PromptTok == 0 {
			u.PromptTok = doc.Timings.PromptN + doc.Timings.CacheN
		}
		if u.ComplTok == 0 {
			u.ComplTok = doc.Timings.PredictedN
		}
		if u.CachedTok == 0 {
			u.CachedTok = doc.Timings.CacheN
		}
		found = true
	}
	if !found && (doc.TokensEvaluated > 0 || doc.TokensPredicted > 0) {
		u.PromptTok = doc.TokensEvaluated + doc.TokensCached
		u.ComplTok = doc.TokensPredicted
		u.CachedTok = doc.TokensCached
		found = true
	}
	return u, found
}

// IsSSE reports whether a response content type is a server-sent event stream.
func IsSSE(contentType string) bool {
	return strings.HasPrefix(strings.ToLower(contentType), "text/event-stream")
}
