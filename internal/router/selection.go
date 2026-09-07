package router

import (
	"fmt"
	"hash/fnv"
	"net/url"
	"sort"
	"sync"
	"time"

	"github.com/cpuchip/llama-chip/internal/fed"
	"github.com/cpuchip/llama-chip/internal/telemetry"
)

// telSnapshot returns the latest telemetry sample, or a zero snapshot when no sampler is attached
// (tests, or a node started without telemetry).
func (rt *Router) telSnapshot() telemetry.Snapshot {
	if rt.tel == nil {
		return telemetry.Snapshot{}
	}
	return rt.tel.Snapshot()
}

// affinityTTL is how long a conversation sticks to the node that already holds its prefix cache.
// Moving a 20k-token KV between boxes costs more than re-prefilling it at home speeds, so the win
// is keeping a conversation put, not shipping its cache. A continued conversation keeps landing on
// the same node until it idles past this, or that node stops serving the model.
const affinityTTL = 30 * time.Minute

// candidate is one place a request could go.
type candidate struct {
	local  bool
	name   string   // node name (peer) or slot name (local)
	url    *url.URL // nil for local (proxied to the slot's own port by the caller)
	bearer string
	load   int  // inflight + queued; lower is less busy
	loaded bool // the model is resident (not a cold load)
}

// affinity remembers which node a conversation last used, so its cache is reused.
type affinity struct {
	mu  sync.Mutex
	m   map[string]affinityEntry
	now func() time.Time
}

type affinityEntry struct {
	node string
	exp  time.Time
}

func newAffinity() *affinity {
	return &affinity{m: map[string]affinityEntry{}, now: time.Now}
}

// get returns the sticky node for a key if it has not expired.
func (a *affinity) get(key string) (string, bool) {
	if key == "" {
		return "", false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	e, ok := a.m[key]
	if !ok || a.now().After(e.exp) {
		return "", false
	}
	return e.node, true
}

// put records (or refreshes) the sticky node for a key.
func (a *affinity) put(key, node string) {
	if key == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.m[key] = affinityEntry{node: node, exp: a.now().Add(affinityTTL)}
	// Opportunistic sweep so the map cannot grow without bound on a long-lived node.
	if len(a.m) > 4096 {
		now := a.now()
		for k, e := range a.m {
			if now.After(e.exp) {
				delete(a.m, k)
			}
		}
	}
}

// affinityKey derives a stable key for a conversation from an explicit session header, or from a
// hash of the body prefix (the system prompt plus the start of the first turn, which stay constant
// as a conversation grows). Empty when there is nothing to key on (affinity then does not apply).
func affinityKey(sessionHeader string, body []byte) string {
	if sessionHeader != "" {
		return "h:" + sessionHeader
	}
	if len(body) == 0 {
		return ""
	}
	n := len(body)
	if n > 1024 {
		n = 1024
	}
	h := fnv.New64a()
	_, _ = h.Write(body[:n])
	return "b:" + strconvU64(h.Sum64())
}

func strconvU64(v uint64) string {
	const digits = "0123456789abcdef"
	var buf [16]byte
	for i := 15; i >= 0; i-- {
		buf[i] = digits[v&0xf]
		v >>= 4
	}
	return string(buf[:])
}

// rank orders candidates: the affinity target first (if present), then least-loaded, ties broken
// local-first then by name, so a fresh conversation spreads by real queue depth while a continued
// one sticks. Returns the chosen candidate and whether the choice was made by affinity.
func rank(cands []candidate, affNode string) (candidate, bool) {
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].load != cands[j].load {
			return cands[i].load < cands[j].load
		}
		if cands[i].local != cands[j].local {
			return cands[i].local
		}
		return cands[i].name < cands[j].name
	})
	if affNode != "" {
		for _, c := range cands {
			if c.name == affNode {
				return c, true
			}
		}
	}
	return cands[0], false
}

// peerCandidates builds candidates from the federation's peer health for a model. A peer is a
// candidate when it is online and advertises the model on a loaded slot (or, for a peer too old to
// publish the slot contract, simply lists the model). Load is that slot's inflight+queued.
func peerCandidates(peers []fed.PeerHealth, model string) []candidate {
	var out []candidate
	for _, p := range peers {
		if !p.Online {
			continue
		}
		load, loaded, serves := 0, false, false
		for _, s := range p.Slots {
			if s.Name == model {
				serves = true
				loaded = s.Loaded
				load = s.Inflight + s.Queued
				break
			}
		}
		if !serves { // an older peer with no slot contract: fall back to its model list
			for _, m := range p.Models {
				if m == model {
					serves, loaded = true, true // assume resident; the peer cannot tell us otherwise
					break
				}
			}
		}
		if !serves {
			continue
		}
		u, err := url.Parse(p.URL)
		if err != nil {
			continue
		}
		out = append(out, candidate{name: p.Name, url: u, load: load, loaded: loaded})
	}
	return out
}

// pickTarget chooses where a request for model goes, honoring prefix affinity then load. It
// returns the reverse-proxy target, a human label, the outbound bearer, and statSlot (the local
// slot name to record request stats against, empty when the request goes to a peer). ok is false
// when nothing serves the model.
//
// Local availability respects the yield controller: a llama-server slot whose card is marked
// yielding is treated as unavailable so the request fails over to the fleet while a game holds the
// card. An external slot (a vLLM container) is never withdrawn by us, so it stays a candidate.
func (rt *Router) pickTarget(model string, body []byte, sessionHeader string) (target *url.URL, label, bearer, statSlot string, ok bool) {
	var cands []candidate
	localName := rt.fed.NodeName()
	if localName == "" {
		localName = "local"
	}

	inst, hasLocal := rt.rig.Resolve(model)
	snap := rt.telSnapshot()
	if hasLocal {
		if inst.External != nil {
			load := 0
			if v, in := snap.Slots[inst.Slot.Name()]; in {
				load = v.Inflight + v.Queued
			}
			cands = append(cands, candidate{local: true, name: localName, load: load, loaded: true})
		} else if !anyGPUYielding(snap.GPUs, inst.Slot.GPUs) {
			v, in := snap.Slots[inst.Slot.Name()]
			// With no sampler (tests) treat the local slot as available; the rig only returns
			// slots it supervises.
			if !in || v.Requests >= 0 {
				cands = append(cands, candidate{local: true, name: localName, load: v.Inflight + v.Queued, loaded: true})
			}
		}
	}
	cands = append(cands, peerCandidates(rt.fed.Peers(), model)...)
	if len(cands) == 0 {
		return nil, "", "", "", false
	}

	key := affinityKey(sessionHeader, body)
	affNode, _ := rt.aff.get(key)
	chosen, _ := rank(cands, affNode)
	rt.aff.put(key, chosen.name)

	if chosen.local {
		if inst.External != nil {
			return inst.External, fmt.Sprintf("external slot %q (%s)", inst.Slot.Name(), inst.External), inst.Bearer(), inst.Slot.Name(), true
		}
		u, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", inst.Port))
		return u, fmt.Sprintf("local slot %q (:%d)", inst.Slot.Name(), inst.Port), "", inst.Slot.Name(), true
	}
	return chosen.url, fmt.Sprintf("peer %q (%s)", chosen.name, chosen.url), rt.peerBearer(chosen.name), "", true
}

// anyGPUYielding reports whether any of the slot's cards is currently marked yielding.
func anyGPUYielding(gpus []telemetry.GPUView, slotGPUs []int) bool {
	for _, g := range gpus {
		if !g.Yielding {
			continue
		}
		for _, s := range slotGPUs {
			if s == g.Index {
				return true
			}
		}
	}
	return false
}
