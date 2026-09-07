package router

import (
	"testing"
	"time"

	"github.com/cpuchip/llama-chip/internal/fed"
)

func TestAffinityStickyAndExpiry(t *testing.T) {
	a := newAffinity()
	now := time.Unix(1_000_000, 0)
	a.now = func() time.Time { return now }
	if _, ok := a.get("k"); ok {
		t.Fatal("empty key should miss")
	}
	a.put("k", "threadchip")
	if n, ok := a.get("k"); !ok || n != "threadchip" {
		t.Fatalf("sticky lookup = %q %v", n, ok)
	}
	now = now.Add(affinityTTL + time.Minute)
	if _, ok := a.get("k"); ok {
		t.Fatal("entry should have expired")
	}
}

func TestAffinityKeyStableAcrossGrowingBody(t *testing.T) {
	// A conversation's prefix (system + first turn) is constant as later turns are appended, so
	// the key must not change when the body grows past the hashed prefix.
	base := make([]byte, 1024)
	for i := range base {
		base[i] = byte('a' + i%26)
	}
	k1 := affinityKey("", base)
	grown := append(append([]byte{}, base...), []byte("...many more turns appended later...")...)
	k2 := affinityKey("", grown)
	if k1 != k2 || k1 == "" {
		t.Fatalf("key changed as body grew: %q vs %q", k1, k2)
	}
	// The session header wins and is independent of the body.
	if affinityKey("sess-42", base) != "h:sess-42" {
		t.Fatal("session header should be the key verbatim")
	}
	// Different prefixes hash apart.
	if affinityKey("", []byte("totally different opening")) == k1 {
		t.Fatal("different prefixes must not collide")
	}
}

func TestRankLoadThenLocalThenName(t *testing.T) {
	// Least loaded wins.
	c, byAff := rank([]candidate{
		{name: "a", load: 3},
		{name: "b", load: 1},
		{local: true, name: "self", load: 2},
	}, "")
	if byAff || c.name != "b" {
		t.Fatalf("least-loaded should win: %+v byAff=%v", c, byAff)
	}
	// Tie on load: local wins.
	c, _ = rank([]candidate{{name: "peer", load: 1}, {local: true, name: "self", load: 1}}, "")
	if !c.local {
		t.Fatalf("tie should go local: %+v", c)
	}
	// Affinity overrides load: the sticky node wins even when busier.
	c, byAff = rank([]candidate{{name: "idle", load: 0}, {name: "sticky", load: 5}}, "sticky")
	if !byAff || c.name != "sticky" {
		t.Fatalf("affinity should override load: %+v byAff=%v", c, byAff)
	}
	// A stale affinity target no longer in the set falls back to least-loaded.
	c, byAff = rank([]candidate{{name: "idle", load: 0}, {name: "busy", load: 9}}, "gone")
	if byAff || c.name != "idle" {
		t.Fatalf("stale affinity should fall back: %+v byAff=%v", c, byAff)
	}
}

func TestPeerCandidatesSlotContractAndFallback(t *testing.T) {
	peers := []fed.PeerHealth{
		{Name: "new", URL: "http://new:8090", Online: true, Slots: []fed.PeerSlot{{Name: "m", Loaded: true, Inflight: 2, Queued: 1}}},
		{Name: "old", URL: "http://old:8090", Online: true, Models: []string{"m"}}, // pre-slot-contract peer
		{Name: "off", URL: "http://off:8090", Online: false, Models: []string{"m"}},
		{Name: "other", URL: "http://other:8090", Online: true, Slots: []fed.PeerSlot{{Name: "n", Loaded: true}}},
	}
	cs := peerCandidates(peers, "m")
	if len(cs) != 2 {
		t.Fatalf("want new+old, got %d: %+v", len(cs), cs)
	}
	byName := map[string]candidate{}
	for _, c := range cs {
		byName[c.name] = c
	}
	if byName["new"].load != 3 || !byName["new"].loaded {
		t.Fatalf("new peer load/loaded wrong: %+v", byName["new"])
	}
	if _, ok := byName["off"]; ok {
		t.Fatal("offline peer must be excluded")
	}
	if _, ok := byName["other"]; ok {
		t.Fatal("peer not serving the model must be excluded")
	}
}
