package yield

import (
	"sync"
	"testing"
	"time"

	"github.com/cpuchip/llama-chip/internal/config"
	"github.com/cpuchip/llama-chip/internal/rig"
	"github.com/cpuchip/llama-chip/internal/telemetry"
)

type fakeRig struct {
	mu      sync.Mutex
	loaded  map[string]rig.Status
	unloads []string
	loads   []string
}

func newFakeRig(slots ...rig.Status) *fakeRig {
	m := map[string]rig.Status{}
	for _, s := range slots {
		m[s.Name] = s
	}
	return &fakeRig{loaded: m}
}

func (f *fakeRig) Snapshot() []rig.Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]rig.Status, 0, len(f.loaded))
	for _, s := range f.loaded {
		out = append(out, s)
	}
	return out
}

func (f *fakeRig) Load(s config.Slot) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.loads = append(f.loads, s.Name())
	f.loaded[s.Name()] = rig.Status{Name: s.Name(), Kind: "llama-server", GPUs: s.GPUs, Loaded: true, State: rig.Healthy}
	return nil
}

func (f *fakeRig) Unload(name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.unloads = append(f.unloads, name)
	delete(f.loaded, name)
	return nil
}

func (f *fakeRig) names() map[string]bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	m := map[string]bool{}
	for n := range f.loaded {
		m[n] = true
	}
	return m
}

type fakeSampler struct {
	mu       sync.Mutex
	snap     telemetry.Snapshot
	yielding map[int]bool
}

func (s *fakeSampler) set(gpus ...telemetry.GPUView) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snap = telemetry.Snapshot{GPUs: gpus}
}

func (s *fakeSampler) Snapshot() telemetry.Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.snap
}

func (s *fakeSampler) SetYielding(index int, on bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.yielding == nil {
		s.yielding = map[int]bool{}
	}
	if on {
		s.yielding[index] = true
	} else {
		delete(s.yielding, index)
	}
}

func TestYieldThenRestoreOnForeign(t *testing.T) {
	r := newFakeRig(rig.Status{Name: "m", Kind: "llama-server", GPUs: []int{1}, Loaded: true, State: rig.Healthy})
	s := &fakeSampler{}
	slots := []config.Slot{{Alias: "m", Model: "m.gguf", GPUs: []int{1}, CtxSize: 32768}}
	c := New(r, s, nil, slots, &config.Yield{Enabled: true, ForeignMiB: 1024, HoldSec: 5, RestoreSec: 30}, nil)

	base := time.Unix(1_000_000, 0)

	// A game appears on card 1. Below hold: nothing yet.
	s.set(telemetry.GPUView{Index: 1, ForeignMiB: 6000})
	c.evaluate(base)
	if len(r.unloads) != 0 {
		t.Fatalf("yielded before hold elapsed: %v", r.unloads)
	}
	// Hold elapsed: the slot is unloaded and the card marked yielding.
	c.evaluate(base.Add(6 * time.Second))
	if r.names()["m"] || len(r.unloads) != 1 {
		t.Fatalf("expected unload after hold; unloads=%v loaded=%v", r.unloads, r.names())
	}
	if !s.yielding[1] {
		t.Fatal("card 1 should be marked yielding")
	}
	// Game still running: no restore.
	c.evaluate(base.Add(20 * time.Second))
	if len(r.loads) != 0 {
		t.Fatalf("restored while still hot: %v", r.loads)
	}
	// Game exits (foreign gone). Below restore delay: still yielded.
	s.set(telemetry.GPUView{Index: 1, ForeignMiB: 0})
	c.evaluate(base.Add(40 * time.Second))
	if len(r.loads) != 0 {
		t.Fatalf("restored before restore delay: %v", r.loads)
	}
	// Restore delay elapsed: the slot comes back and the mark clears.
	c.evaluate(base.Add(75 * time.Second))
	if !r.names()["m"] || len(r.loads) != 1 {
		t.Fatalf("expected restore; loads=%v loaded=%v", r.loads, r.names())
	}
	if s.yielding[1] {
		t.Fatal("card 1 should no longer be yielding")
	}
}

func TestDisabledPolicyDoesNotYield(t *testing.T) {
	r := newFakeRig(rig.Status{Name: "m", Kind: "llama-server", GPUs: []int{1}, Loaded: true})
	s := &fakeSampler{}
	s.set(telemetry.GPUView{Index: 1, ForeignMiB: 9000})
	c := New(r, s, nil, nil, &config.Yield{Enabled: false}, nil)
	c.evaluate(time.Unix(1, 0))
	c.evaluate(time.Unix(100, 0))
	if len(r.unloads) != 0 {
		t.Fatalf("disabled policy yielded: %v", r.unloads)
	}
}

func TestManualYieldIgnoresThresholdAndHold(t *testing.T) {
	r := newFakeRig(rig.Status{Name: "m", Kind: "llama-server", GPUs: []int{0}, Loaded: true})
	s := &fakeSampler{}
	s.set(telemetry.GPUView{Index: 0, ForeignMiB: 0}) // no game at all
	c := New(r, s, nil, nil, &config.Yield{Enabled: false}, nil)
	c.SetManual(0, true) // manual works even with the policy disabled and no foreign use
	if r.names()["m"] {
		t.Fatalf("manual yield should have unloaded immediately: %v", r.names())
	}
	// Releasing the manual hold, with no foreign use, restores after the delay.
	c.SetManual(0, false)
	c.evaluate(time.Unix(1, 0))
	c.evaluate(time.Unix(1000, 0))
	if !r.names()["m"] {
		t.Fatalf("release with no game should restore: loads=%v", r.loads)
	}
}

func TestExternalSlotNeverYielded(t *testing.T) {
	r := newFakeRig(rig.Status{Name: "vllm", Kind: "external", GPUs: []int{0}, Loaded: true})
	s := &fakeSampler{}
	s.set(telemetry.GPUView{Index: 0, ForeignMiB: 9000})
	c := New(r, s, nil, nil, &config.Yield{Enabled: true, ForeignMiB: 1024, HoldSec: 0}, nil)
	c.evaluate(time.Unix(1, 0))
	c.evaluate(time.Unix(10, 0))
	if len(r.unloads) != 0 {
		t.Fatalf("external slot must never be unloaded: %v", r.unloads)
	}
	if !s.yielding[0] {
		t.Fatal("card still marks yielding so placement avoids it, even with nothing to unload")
	}
}
