package fed

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// fakePeer is a stand-in llama-chip node exposing /api/fed/local (and /v1/models for the
// fallback path). Its model list and the auth token it requires can be flipped at runtime.
type fakePeer struct {
	srv     *httptest.Server
	mu      sync.Mutex
	models  []string
	wantTok string // if set, requests without "Bearer <wantTok>" get 401
	gotAuth string // the last Authorization header seen
	v1Only  bool   // when true, /api/fed/local 404s to force the /v1/models fallback
}

func newFakePeer(models ...string) *fakePeer {
	p := &fakePeer{models: models}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/fed/local", func(w http.ResponseWriter, r *http.Request) {
		if !p.authOK(w, r) {
			return
		}
		p.mu.Lock()
		v1Only, models := p.v1Only, append([]string(nil), p.models...)
		p.mu.Unlock()
		if v1Only {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(localView{Node: "fake", Models: models})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if !p.authOK(w, r) {
			return
		}
		p.mu.Lock()
		models := append([]string(nil), p.models...)
		p.mu.Unlock()
		var data []map[string]any
		for _, m := range models {
			data = append(data, map[string]any{"id": m, "object": "model"})
		}
		json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	})
	p.srv = httptest.NewServer(mux)
	return p
}

func (p *fakePeer) authOK(w http.ResponseWriter, r *http.Request) bool {
	p.mu.Lock()
	p.gotAuth = r.Header.Get("Authorization")
	want := p.wantTok
	p.mu.Unlock()
	if want != "" && r.Header.Get("Authorization") != "Bearer "+want {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	return true
}

func (p *fakePeer) setModels(m ...string) {
	p.mu.Lock()
	p.models = m
	p.mu.Unlock()
}

func (p *fakePeer) lastAuth() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.gotAuth
}

func (p *fakePeer) close() { p.srv.Close() }

func testFed(t *testing.T, cfg Config) *Federation {
	t.Helper()
	cfg.NodeName = "test-head"
	f := New(cfg, log.New(io.Discard, "", 0))
	if f == nil {
		t.Fatal("New returned nil (expected an enabled federation)")
	}
	return f
}

func TestLearnsAndResolvesPeerModels(t *testing.T) {
	peer := newFakePeer("qwen3.6-35b-a3b", "gemma-4-26b-a4b")
	defer peer.close()

	f := testFed(t, Config{Peers: []PeerConfig{{Name: "home-gpu", URL: peer.srv.URL}}})
	f.pollAll(context.Background())

	if r, ok := f.Resolve("qwen3.6-35b-a3b"); !ok || r.PeerURL != peer.srv.URL {
		t.Fatalf("expected qwen routed to peer, got %+v ok=%v", r, ok)
	}
	if r, ok := f.Resolve("gemma-4-26b-a4b"); !ok || r.PeerName != "home-gpu" {
		t.Fatalf("expected gemma routed to home-gpu, got %+v ok=%v", r, ok)
	}
	if _, ok := f.Resolve("nonexistent-model"); ok {
		t.Fatal("resolved a model no peer serves")
	}
}

func TestSubstringMatchResolves(t *testing.T) {
	peer := newFakePeer("Qwen3.6-35B-A3B-Q4_K_M")
	defer peer.close()
	f := testFed(t, Config{Peers: []PeerConfig{{URL: peer.srv.URL}}})
	f.pollAll(context.Background())

	// a role-alias / shorthand substring should still route
	if _, ok := f.Resolve("qwen3.6-35b-a3b"); !ok {
		t.Fatal("expected substring handle to resolve to the full model name")
	}
}

func TestEvictsOnPeerDrop(t *testing.T) {
	peer := newFakePeer("qwen3.6-35b-a3b")
	f := testFed(t, Config{Peers: []PeerConfig{{URL: peer.srv.URL}}})
	f.pollAll(context.Background())
	if _, ok := f.Resolve("qwen3.6-35b-a3b"); !ok {
		t.Fatal("precondition: model should resolve while peer is up")
	}

	// peer goes away (laptop leaves the mesh / home box powers off)
	peer.close()
	f.pollAll(context.Background())
	if _, ok := f.Resolve("qwen3.6-35b-a3b"); ok {
		t.Fatal("model should be evicted once its peer is unreachable")
	}
	for _, h := range f.Peers() {
		if h.Online {
			t.Fatalf("peer should read offline after drop, got %+v", h)
		}
	}
}

func TestModelReappearsWhenPeerReturns(t *testing.T) {
	peer := newFakePeer() // starts serving nothing
	defer peer.close()
	f := testFed(t, Config{Peers: []PeerConfig{{URL: peer.srv.URL}}})
	f.pollAll(context.Background())
	if _, ok := f.Resolve("qwen3.6-35b-a3b"); ok {
		t.Fatal("nothing should resolve before the peer loads the model")
	}
	peer.setModels("qwen3.6-35b-a3b")
	f.pollAll(context.Background())
	if _, ok := f.Resolve("qwen3.6-35b-a3b"); !ok {
		t.Fatal("model should resolve once the peer loads it")
	}
}

func TestFirstPeerWinsTie(t *testing.T) {
	a := newFakePeer("shared-model")
	b := newFakePeer("shared-model")
	defer a.close()
	defer b.close()
	f := testFed(t, Config{Peers: []PeerConfig{
		{Name: "first", URL: a.srv.URL},
		{Name: "second", URL: b.srv.URL},
	}})
	f.pollAll(context.Background())
	r, ok := f.Resolve("shared-model")
	if !ok || r.PeerName != "first" {
		t.Fatalf("expected first-listed peer to win the tie, got %+v ok=%v", r, ok)
	}
}

func TestTokenIsSentToPeers(t *testing.T) {
	peer := newFakePeer("m1")
	peer.wantTok = "s3cret"
	defer peer.close()
	f := testFed(t, Config{Token: "s3cret", Peers: []PeerConfig{{URL: peer.srv.URL}}})
	f.pollAll(context.Background())
	if _, ok := f.Resolve("m1"); !ok {
		t.Fatal("expected to resolve with the right token")
	}
	if got := peer.lastAuth(); got != "Bearer s3cret" {
		t.Fatalf("peer saw Authorization %q, want %q", got, "Bearer s3cret")
	}
}

// A bearer-gated STATIC peer must be pollable with its per-peer token (the same override used to
// proxy to it) — otherwise a node with its own federation.token could never be discovered over
// static peering. Mirrors TestTokenIsSentToPeers but with a per-peer token distinct from the
// node's own (here the node has no token at all).
func TestPeerTokenIsSentWhenPollingStaticPeer(t *testing.T) {
	peer := newFakePeer("m1")
	peer.wantTok = "peercred" // gate the peer with ITS token
	defer peer.close()
	f := testFed(t, Config{
		Peers:      []PeerConfig{{Name: "nocix", URL: peer.srv.URL}},
		PeerTokens: map[string]string{"nocix": "peercred"},
	})
	f.pollAll(context.Background())
	if _, ok := f.Resolve("m1"); !ok {
		t.Fatal("expected the peer-gated static peer to be discovered using its per-peer token")
	}
	if got := peer.lastAuth(); got != "Bearer peercred" {
		t.Fatalf("peer saw Authorization %q, want Bearer peercred (per-peer token on the poll)", got)
	}
}

func TestV1ModelsFallback(t *testing.T) {
	peer := newFakePeer("legacy-model")
	peer.v1Only = true // /api/fed/local 404s; poller must fall back to /v1/models
	defer peer.close()
	f := testFed(t, Config{Peers: []PeerConfig{{URL: peer.srv.URL}}})
	f.pollAll(context.Background())
	if _, ok := f.Resolve("legacy-model"); !ok {
		t.Fatal("expected /v1/models fallback to discover the model")
	}
}

func TestNewReturnsNilWithoutPeers(t *testing.T) {
	if f := New(Config{NodeName: "x"}, log.New(io.Discard, "", 0)); f != nil {
		t.Fatal("federation with no peers, hub, or token should be nil (off)")
	}
	// nil-safe helpers
	var f *Federation
	if f.Enabled() || f.Token() != "" || f.NodeName() != "" || f.PeerToken("x") != "" {
		t.Fatal("nil federation should report disabled/empty")
	}
	if _, ok := f.Resolve("anything"); ok {
		t.Fatal("nil federation should resolve nothing")
	}
}

// A token WITHOUT peers/hub yields a non-nil, NOT-Enabled federation: it does not route, but it
// still exposes Token() so the router gates its inbound surface — a public leaf node (NOCIX) that
// must be bearer-protected even though it has no peers of its own.
func TestTokenOnlyFederationGatesButDoesNotRoute(t *testing.T) {
	f := New(Config{NodeName: "nocix", Token: "secret"}, log.New(io.Discard, "", 0))
	if f == nil {
		t.Fatal("a token-only config must yield a non-nil federation so the inbound gate works")
	}
	if f.Enabled() {
		t.Fatal("a token-only federation should NOT be Enabled (no peers/hub → no routing)")
	}
	if f.Token() != "secret" {
		t.Fatalf("Token()=%q want secret", f.Token())
	}
	if _, ok := f.Resolve("anything"); ok {
		t.Fatal("a token-only federation should resolve nothing")
	}
}

func TestPeerTokenLookup(t *testing.T) {
	f := New(Config{
		Peers:      []PeerConfig{{Name: "home", URL: "http://x"}},
		Token:      "node-tok",
		PeerTokens: map[string]string{"NOCIX": "nocix-tok"},
	}, log.New(io.Discard, "", 0))
	if got := f.PeerToken("NOCIX"); got != "nocix-tok" { // exact
		t.Fatalf("PeerToken(NOCIX)=%q want nocix-tok", got)
	}
	if got := f.PeerToken("nocix"); got != "nocix-tok" { // case-insensitive
		t.Fatalf("PeerToken(nocix)=%q want nocix-tok (case-insensitive)", got)
	}
	if got := f.PeerToken("home"); got != "" { // no per-peer entry → "" (caller falls back to Token)
		t.Fatalf("PeerToken(home)=%q want empty (no per-peer override)", got)
	}
	// nil-safe / empty map
	var nilF *Federation
	if nilF.PeerToken("x") != "" {
		t.Fatal("nil federation PeerToken should be empty")
	}
}

func TestRosterStoresGPUsAndSelf(t *testing.T) {
	f := testFed(t, Config{HubURL: "http://hub"}) // enabled (hub mode)
	f.ApplyRoster([]RosterEntry{
		{Name: "alpha", MeshAddr: "http://100.84.0.2:8090", Models: []string{"qwen3.6-35b-a3b", "gemma-4-26b-a4b"},
			GPUs: []RosterGPU{{Index: 0, Name: "RTX 4090", MemTotal: 24564, MemUsed: 21000}, {Index: 1, Name: "RTX 4090", MemTotal: 24564}}},
		{Name: "beta", MeshAddr: "http://100.84.0.3:8090", GPUs: []RosterGPU{{Index: 0, Name: "RTX 3500 Ada", MemTotal: 12282}}},
	}, "beta")

	roster, self := f.Roster()
	if self != "beta" {
		t.Fatalf("self=%q want beta", self)
	}
	if len(roster) != 2 {
		t.Fatalf("roster len=%d want 2", len(roster))
	}
	var alpha *RosterEntry
	for i := range roster {
		if roster[i].Name == "alpha" {
			alpha = &roster[i]
		}
	}
	if alpha == nil || len(alpha.GPUs) != 2 || alpha.GPUs[0].Name != "RTX 4090" {
		t.Fatalf("alpha GPUs not stored in roster: %+v", alpha)
	}
	if _, ok := f.Resolve("qwen3.6-35b-a3b"); !ok {
		t.Fatal("alpha's model should route (self excluded, peers included)")
	}
}

func TestNodeAddrAndIsSelf(t *testing.T) {
	// hub mode: roster carries the mesh addresses
	f := testFed(t, Config{HubURL: "http://hub"})
	f.ApplyRoster([]RosterEntry{
		{Name: "alpha", MeshAddr: "http://100.84.0.2:8090/", Models: []string{"qwen"}},
		{Name: "beta", MeshAddr: "http://100.84.0.3:8090"},
	}, "beta")

	if addr, ok := f.NodeAddr("alpha"); !ok || addr != "http://100.84.0.2:8090" {
		t.Fatalf("NodeAddr(alpha)=%q ok=%v, want trimmed mesh addr", addr, ok)
	}
	if addr, ok := f.NodeAddr("ALPHA"); !ok || addr == "" { // case-insensitive
		t.Fatalf("NodeAddr should be case-insensitive, got %q ok=%v", addr, ok)
	}
	if _, ok := f.NodeAddr("ghost"); ok {
		t.Fatal("NodeAddr resolved a node not in the roster")
	}
	if !f.IsSelf("beta") || !f.IsSelf("Beta") {
		t.Fatal("IsSelf should match the applied self name, case-insensitively")
	}
	if f.IsSelf("alpha") {
		t.Fatal("IsSelf should be false for a peer")
	}

	// static-peer mode: addresses come from polled peer health, self falls back to NodeName
	peer := newFakePeer("m1")
	defer peer.close()
	sf := testFed(t, Config{Peers: []PeerConfig{{Name: "home", URL: peer.srv.URL}}})
	sf.pollAll(context.Background())
	if addr, ok := sf.NodeAddr("home"); !ok || addr != strings.TrimRight(peer.srv.URL, "/") {
		t.Fatalf("static NodeAddr(home)=%q ok=%v, want the peer URL", addr, ok)
	}
	if !sf.IsSelf("test-head") { // testFed sets NodeName=test-head
		t.Fatal("IsSelf should fall back to the configured NodeName in static mode")
	}

	// nil-safe
	var nilF *Federation
	if _, ok := nilF.NodeAddr("x"); ok || nilF.IsSelf("x") {
		t.Fatal("nil federation should resolve nothing and be self of nothing")
	}
}

// guards the gossip contract: /api/fed/local must report LOCAL-only models so two
// mutually-peered nodes never loop. This documents the invariant the router must honor.
func TestLocalViewShapeIsParseable(t *testing.T) {
	body := `{"node":"home","advertise":"http://x:8090","models":["a","b"]}`
	var lv localView
	if err := json.NewDecoder(strings.NewReader(body)).Decode(&lv); err != nil {
		t.Fatal(err)
	}
	if lv.Node != "home" || len(lv.Models) != 2 {
		t.Fatalf("bad parse: %+v", lv)
	}
}
