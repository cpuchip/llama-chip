// Package telemetry measures what the fleet decides on: per-GPU ours/foreign VRAM (who holds a
// card), per-slot in-flight and queued requests, and per-slot request statistics (cache hit
// rate, tokens per second, time to first token). Numbers are measured, never configured.
package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cpuchip/llama-chip/internal/gpu"
	"github.com/cpuchip/llama-chip/internal/rig"
)

// GPUView is one card as the fleet sees it.
type GPUView struct {
	Index       int    `json:"index"`
	UUID        string `json:"uuid,omitempty"`
	Name        string `json:"name"`
	MemTotal    int    `json:"mem_total_mib"`
	MemUsed     int    `json:"mem_used_mib"`
	MemFree     int    `json:"mem_free_mib"`
	Util        int    `json:"util_pct"`
	OursMiB     int    `json:"ours_mib"`    // VRAM held by this rig's slots (by PID, or an external slot's baseline)
	ForeignMiB  int    `json:"foreign_mib"` // used minus ours: a game, another server, anything not ours
	ForeignPIDs []int  `json:"foreign_pids,omitempty"`
	Yielding    bool   `json:"yielding"` // the yield controller has marked this card as given up
}

// SlotView is one slot's live load and rolling request statistics.
type SlotView struct {
	Name         string  `json:"name"`
	Inflight     int     `json:"inflight"`             // requests being processed right now
	Queued       int     `json:"queued"`               // requests waiting for a free backend slot
	PromptCache  int     `json:"prompt_cache_tokens"`  // prompt tokens the backend reused from cache on its current requests
	Requests     int     `json:"requests"`             // total routed through this rig since start
	Errors       int     `json:"errors"`               // responses with status >= 400
	CacheHitRate float64 `json:"cache_hit_rate"`       // cached prompt tokens / prompt tokens over the last window
	TokS         float64 `json:"tok_s"`                // completion tokens per second after first token, rolling mean
	TTFTms       float64 `json:"ttft_ms"`              // time to first byte of the response, rolling mean
	SampleErr    string  `json:"sample_err,omitempty"` // why live load could not be read this sample
}

// Snapshot is one sample of the whole node.
type Snapshot struct {
	At    time.Time           `json:"at"`
	GPUs  []GPUView           `json:"gpus"`
	Slots map[string]SlotView `json:"slots"`
	Err   string              `json:"err,omitempty"`
}

// Sampler takes snapshots on a cadence and holds the latest one.
type Sampler struct {
	rig   *rig.Rig
	log   *log.Logger
	stats *Stats

	mu       sync.RWMutex
	snap     Snapshot
	yielding map[int]bool

	// Seams for tests; production values read the platform.
	queryGPUs func() ([]gpu.GPU, error)
	procUsage func([]gpu.GPU) ([]gpu.ProcUsage, error)
	slotsLive func(port int) (inflight, queued, cache int, err error)
}

func New(r *rig.Rig, logger *log.Logger) *Sampler {
	if logger == nil {
		logger = log.Default()
	}
	return &Sampler{
		rig:       r,
		log:       logger,
		stats:     NewStats(),
		yielding:  map[int]bool{},
		queryGPUs: gpu.Query,
		procUsage: gpu.ProcessUsage,
		slotsLive: slotsLive,
	}
}

// Stats is where the router records each request it proxied.
func (s *Sampler) Stats() *Stats { return s.stats }

// Run samples every interval until ctx ends.
func (s *Sampler) Run(ctx context.Context, every time.Duration) {
	s.Sample()
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.Sample()
		}
	}
}

// Snapshot returns the latest sample (zero value before the first).
func (s *Sampler) Snapshot() Snapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snap
}

// SetYielding marks a card as yielded (or restored); the next sample carries it.
func (s *Sampler) SetYielding(index int, on bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if on {
		s.yielding[index] = true
	} else {
		delete(s.yielding, index)
	}
}

// Yielding reports whether a card is marked yielded.
func (s *Sampler) Yielding(index int) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.yielding[index]
}

// Sample takes one snapshot now and stores it.
func (s *Sampler) Sample() Snapshot {
	snap := Snapshot{At: time.Now(), Slots: map[string]SlotView{}}
	statuses := s.rig.Snapshot()

	gpus, err := s.queryGPUs()
	if err != nil {
		snap.Err = "nvidia-smi: " + err.Error()
	}
	var procs []gpu.ProcUsage
	if err == nil && len(gpus) > 0 {
		procs, err = s.procUsage(gpus)
		if err != nil {
			snap.Err = "process usage: " + err.Error()
		}
	}
	// A running llama-server pinned to one card names that card's adapter, whatever the usage
	// levels say (Windows learns LUIDs from this; Linux already knows UUIDs).
	for _, st := range statuses {
		if st.PID == 0 || len(st.GPUs) != 1 {
			continue
		}
		for _, p := range procs {
			if p.PID == st.PID && p.GPUIndex < 0 && p.LUID != "" {
				gpu.LearnLUID(p.LUID, st.GPUs[0])
			}
		}
	}
	if len(procs) > 0 {
		procs, _ = s.procUsage(gpus) // re-stamp with anything just learned (cheap on the second call: cached counters are not, so tolerate one extra read)
	}
	s.mu.RLock()
	yielding := make(map[int]bool, len(s.yielding))
	for k, v := range s.yielding {
		yielding[k] = v
	}
	s.mu.RUnlock()
	snap.GPUs = compute(gpus, procs, statuses, yielding)

	for _, st := range statuses {
		v := SlotView{Name: st.Name}
		if st.Loaded && st.Kind == "llama-server" {
			inflight, queued, cache, lerr := s.slotsLive(st.Port)
			if lerr != nil {
				v.SampleErr = lerr.Error()
			} else {
				v.Inflight, v.Queued, v.PromptCache = inflight, queued, cache
			}
		}
		v.Requests, v.Errors, v.CacheHitRate, v.TokS, v.TTFTms = s.stats.View(st.Name)
		snap.Slots[st.Name] = v
	}

	s.mu.Lock()
	s.snap = snap
	s.mu.Unlock()
	return snap
}

// compute attributes each card's used VRAM between ours and foreign. Pure, for tests.
func compute(gpus []gpu.GPU, procs []gpu.ProcUsage, statuses []rig.Status, yielding map[int]bool) []GPUView {
	ours := map[int]bool{}
	for _, st := range statuses {
		if st.PID > 0 {
			ours[st.PID] = true
		}
	}
	oursMiB := map[int]int{}
	for _, p := range procs {
		if p.GPUIndex >= 0 && ours[p.PID] {
			oursMiB[p.GPUIndex] += p.MiB
		}
	}
	for _, st := range statuses {
		if st.Kind == "external" && st.Loaded {
			for idx, base := range st.BaselineMiB {
				oursMiB[idx] += base
			}
		}
	}
	out := make([]GPUView, 0, len(gpus))
	for _, g := range gpus {
		v := GPUView{
			Index: g.Index, UUID: g.UUID, Name: g.Name,
			MemTotal: g.MemTotal, MemUsed: g.MemUsed, MemFree: g.MemFree, Util: g.Util,
			OursMiB: oursMiB[g.Index], Yielding: yielding[g.Index],
		}
		if v.MemFree == 0 && v.MemTotal > 0 {
			v.MemFree = v.MemTotal - v.MemUsed
		}
		if v.OursMiB > v.MemUsed {
			v.OursMiB = v.MemUsed
		}
		v.ForeignMiB = v.MemUsed - v.OursMiB
		if v.ForeignMiB < 0 {
			v.ForeignMiB = 0
		}
		for _, p := range procs {
			if p.GPUIndex == g.Index && !ours[p.PID] && p.PID > 0 && p.MiB >= 64 {
				v.ForeignPIDs = append(v.ForeignPIDs, p.PID)
			}
		}
		sort.Ints(v.ForeignPIDs)
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Index < out[j].Index })
	return out
}

var liveClient = &http.Client{Timeout: 1500 * time.Millisecond}

// slotsLive reads a llama-server's /slots (in-flight, cache reuse) and /metrics (deferred =
// queued). /metrics needs the server launched with --metrics; without it queued stays 0.
func slotsLive(port int) (inflight, queued, cache int, err error) {
	resp, err := liveClient.Get(fmt.Sprintf("http://127.0.0.1:%d/slots", port))
	if err != nil {
		return 0, 0, 0, err
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	resp.Body.Close()
	inflight, cache, err = parseSlots(body)
	if err != nil {
		return 0, 0, 0, err
	}
	if mresp, merr := liveClient.Get(fmt.Sprintf("http://127.0.0.1:%d/metrics", port)); merr == nil {
		mbody, _ := io.ReadAll(io.LimitReader(mresp.Body, 1<<20))
		mresp.Body.Close()
		if mresp.StatusCode == 200 {
			queued = parseDeferred(string(mbody))
		}
	}
	return inflight, queued, cache, nil
}

func parseSlots(body []byte) (inflight, cache int, err error) {
	var raw []struct {
		IsProcessing       bool `json:"is_processing"`
		NPromptTokensCache int  `json:"n_prompt_tokens_cache"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return 0, 0, fmt.Errorf("slots: %w", err)
	}
	for _, r := range raw {
		if r.IsProcessing {
			inflight++
			cache += r.NPromptTokensCache
		}
	}
	return inflight, cache, nil
}

// parseDeferred pulls llamacpp:requests_deferred out of a Prometheus exposition.
func parseDeferred(text string) int {
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "llamacpp:requests_deferred") {
			f := strings.Fields(line)
			if len(f) >= 2 {
				if v, err := strconv.ParseFloat(f[len(f)-1], 64); err == nil {
					return int(v)
				}
			}
		}
	}
	return 0
}
