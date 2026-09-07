// Package yield gives a GPU up to the foreground. When something not ours takes VRAM on a card
// (a game starting), the controller drains and unloads the rig's slots on that card so the work
// fails over to a fleet peer, and reloads them when the foreign use is gone. This is the piece
// that lets one box be both a workstation and a fleet node: the desktop wins, the fleet adapts.
//
// It acts only on llama-server slots (the rig launched them, so it can stop and restart them).
// An external slot (a vLLM container) is left alone: the rig did not launch it and cannot move
// it, so a foreign hog on that card is logged, not yielded.
package yield

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/cpuchip/llama-chip/internal/config"
	"github.com/cpuchip/llama-chip/internal/rig"
	"github.com/cpuchip/llama-chip/internal/telemetry"
)

// Rig is the slice of *rig.Rig the controller drives (an interface so tests can fake it).
type Rig interface {
	Snapshot() []rig.Status
	Load(config.Slot) error
	Unload(name string) error
}

// Sampler is the telemetry the controller reads and writes yield marks on.
type Sampler interface {
	Snapshot() telemetry.Snapshot
	SetYielding(index int, on bool)
}

// Peer is one federation peer the controller may pre-warm before it yields.
type Peer struct {
	Name string
	URL  string
}

// Fed is the federation view the controller uses to pick a warm-up target.
type Fed interface {
	Peers() []Peer // online peers, for a best-effort pre-warm
}

// Controller watches the sampler and yields or restores cards.
type Controller struct {
	rig    Rig
	tel    Sampler
	fed    Fed
	slots  map[string]config.Slot // by name: how to reload a yielded slot
	policy *config.Yield
	log    *log.Logger
	client *http.Client
	tick   time.Duration

	mu       sync.Mutex
	overHot  map[int]time.Time   // gpu -> when foreign first exceeded the threshold (zero = below)
	overCold map[int]time.Time   // gpu -> when foreign dropped back below (zero = above)
	yielded  map[int][]config.Slot // gpu -> slots unloaded for the yield, to reload on restore
	manual   map[int]bool        // gpu -> a human forced this card to yield
}

// New builds a controller. slots is the rig's configured slot list (to reload after a yield).
func New(r Rig, tel Sampler, fed Fed, slots []config.Slot, policy *config.Yield, logger *log.Logger) *Controller {
	if logger == nil {
		logger = log.Default()
	}
	byName := make(map[string]config.Slot, len(slots))
	for _, s := range slots {
		byName[s.Name()] = s
	}
	return &Controller{
		rig: r, tel: tel, fed: fed, slots: byName, policy: policy, log: logger,
		client:   &http.Client{Timeout: 10 * time.Second},
		tick:     2 * time.Second,
		overHot:  map[int]time.Time{},
		overCold: map[int]time.Time{},
		yielded:  map[int][]config.Slot{},
		manual:   map[int]bool{},
	}
}

// Run evaluates on a ticker until ctx ends. A nil or disabled policy still serves manual yields.
func (c *Controller) Run(ctx context.Context) {
	t := time.NewTicker(c.tick)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			c.evaluate(time.Now())
		}
	}
}

// SetManual forces a card to yield (on=true) or releases the manual hold (on=false). Manual yield
// acts at once and ignores the foreign threshold and hold delay. Manual release only clears the
// hold: the regular evaluate loop then restores the card after the restore delay, and not at all
// while a real game is still using it (foreign use keeps the card hot on its own). Release does
// not restore inline, so it never stamps the cold timer off the wall clock the evaluate loop uses.
func (c *Controller) SetManual(gpu int, on bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if on {
		c.manual[gpu] = true
		if _, already := c.yielded[gpu]; !already {
			c.doYield(gpu)
		}
		return
	}
	delete(c.manual, gpu)
}

func (c *Controller) enabled() bool { return c.policy != nil && c.policy.Enabled }

// evaluate is one decision pass over every card in the latest sample.
func (c *Controller) evaluate(now time.Time) {
	snap := c.tel.Snapshot()
	threshold := c.policy.ForeignThreshold()
	hold := c.policy.Hold()
	restore := c.policy.Restore()

	c.mu.Lock()
	defer c.mu.Unlock()

	for _, g := range snap.GPUs {
		manual := c.manual[g.Index]
		hot := manual || (c.enabled() && g.ForeignMiB >= threshold)
		_, isYielded := c.yielded[g.Index]

		if hot {
			c.overCold[g.Index] = time.Time{}
			if c.overHot[g.Index].IsZero() {
				c.overHot[g.Index] = now
			}
			ready := manual || now.Sub(c.overHot[g.Index]) >= hold
			if ready && !isYielded {
				c.doYield(g.Index)
			}
			continue
		}

		// Not hot.
		c.overHot[g.Index] = time.Time{}
		if !isYielded {
			continue
		}
		if c.overCold[g.Index].IsZero() {
			c.overCold[g.Index] = now
		}
		if now.Sub(c.overCold[g.Index]) >= restore {
			c.doRestore(g.Index)
		}
	}
}

// doYield unloads every llama-server slot on the card and marks it yielded. Caller holds c.mu.
func (c *Controller) doYield(gpu int) {
	var moved []config.Slot
	for _, s := range c.rig.Snapshot() {
		if s.Kind != "llama-server" || !onGPU(s.GPUs, gpu) {
			continue
		}
		def, ok := c.slots[s.Name]
		if !ok {
			// Not in the configured set (loaded at runtime); reconstruct a minimal reload spec
			// from the live status so restore still works.
			def = config.Slot{Alias: s.Name, Model: s.Model, GPUs: s.GPUs, CtxSize: s.Ctx, Parallel: s.Parallel}
		}
		c.prewarmPeers(s.Name)
		if err := c.rig.Unload(s.Name); err != nil {
			c.log.Printf("[yield] gpu %d: unload %q failed: %v", gpu, s.Name, err)
			continue
		}
		moved = append(moved, def)
		c.log.Printf("[yield] gpu %d: unloaded %q; requests fail over to the fleet", gpu, s.Name)
	}
	c.tel.SetYielding(gpu, true)
	c.yielded[gpu] = moved
	if len(moved) == 0 {
		c.log.Printf("[yield] gpu %d: marked yielding, but no movable (llama-server) slot is on it", gpu)
	}
}

// doRestore reloads the slots that were yielded off the card. Caller holds c.mu.
func (c *Controller) doRestore(gpu int) {
	for _, def := range c.yielded[gpu] {
		if err := c.rig.Load(def); err != nil {
			c.log.Printf("[yield] gpu %d: restore %q failed: %v (will retry next pass)", gpu, def.Name(), err)
			return // keep the yield state; try again next tick rather than half-restoring
		}
		c.log.Printf("[yield] gpu %d: restored %q", gpu, def.Name())
	}
	delete(c.yielded, gpu)
	c.overCold[gpu] = time.Time{}
	c.tel.SetYielding(gpu, false)
}

// prewarmPeers best-effort asks each online peer that could serve this slot to ensure it, so the
// hand-off is warm rather than a cold load on the peer. Fire-and-forget; failures are fine (the
// peer may already serve it, or warm standby may be off there).
func (c *Controller) prewarmPeers(name string) {
	if c.fed == nil {
		return
	}
	body, _ := json.Marshal(map[string]string{"alias": name, "model": name})
	for _, p := range c.fed.Peers() {
		go func(url string) {
			req, err := http.NewRequest(http.MethodPost, url+"/api/ensure", bytes.NewReader(body))
			if err != nil {
				return
			}
			req.Header.Set("Content-Type", "application/json")
			resp, err := c.client.Do(req)
			if err == nil {
				resp.Body.Close()
			}
		}(p.URL)
	}
}

func onGPU(gpus []int, g int) bool {
	for _, x := range gpus {
		if x == g {
			return true
		}
	}
	return false
}
