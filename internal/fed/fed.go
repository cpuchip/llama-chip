// Package fed federates llama-chip nodes into one logical rig. Each node runs its own
// local rig (unchanged) and OPTIONALLY lists peers it can reach over a mesh (NetBird /
// Tailscale / LAN). The federation keeps a live map of which models each peer serves
// locally, so the router can route a request it can't serve itself to a peer that can —
// pooling GPUs across machines behind one OpenAI endpoint.
//
// Design: LOCAL-FIRST. A node's own models always resolve locally with zero dependency on
// peers; a peer's models become reachable only while the mesh reaches that peer, and vanish
// the moment it drops. A roaming laptop on its own keeps working on its own GPU; plugged
// into the mesh it sees the whole pool. No single point of failure, no head/worker hierarchy.
//
// Gossip is pull-only: each node polls every peer's /api/fed/local (LOCAL-only model list,
// so two mutually-peered nodes can't loop). The aggregated view (local + remote) is exposed
// to CLIENTS via /v1/models — peers never poll that, so aggregation never feeds back.
package fed

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// PeerConfig is one reachable llama-chip node.
type PeerConfig struct {
	Name string `json:"name,omitempty"` // friendly label; defaults to the host in URL
	URL  string `json:"url"`            // base URL, e.g. http://100.84.0.3:8090 (a mesh IP)
}

// Config configures this node's participation in a federation.
type Config struct {
	NodeName     string        // this node's name (shown in status + gossiped to peers)
	Advertise    string        // the URL peers use to reach THIS node (its mesh address)
	Token        string        // optional bearer token; when set, peer requests must carry it
	Peers        []PeerConfig  // STATIC peers to gossip with (LAN / hand-listed)
	HubURL       string        // optional: a hub (llama.example.com) supplies the roster instead of static peers
	PollInterval time.Duration // how often to refresh; default 5s
}

// Route is a resolved remote placement for a model.
type Route struct {
	Model    string // model name as the peer reports it
	PeerURL  string // the peer's base URL to proxy to
	PeerName string
}

// PeerHealth is the observed state of one peer (for status/UI).
type PeerHealth struct {
	Name     string    `json:"name"`
	URL      string    `json:"url"`
	Online   bool      `json:"online"`
	Models   []string  `json:"models"`
	LastSeen time.Time `json:"last_seen,omitempty"`
	LastErr  string    `json:"last_err,omitempty"`
}

// Federation maintains the live remote-model map by polling peers.
type Federation struct {
	cfg    Config
	client *http.Client
	log    *log.Logger

	mu       sync.RWMutex
	routes   map[string]Route      // lowercased model name -> remote route
	health   map[string]PeerHealth // peer URL -> health
	roster   []RosterEntry         // full hub roster (incl self), for the pool view
	selfName string                // this node's name (to flag self in the roster)
}

// localView is what a peer returns from /api/fed/local — its LOCAL-only model list.
type localView struct {
	Node      string   `json:"node"`
	Advertise string   `json:"advertise,omitempty"`
	Models    []string `json:"models"`
}

// New builds a Federation from config. Returns nil if neither static peers nor a hub are
// configured (federation off — the rig behaves exactly as a standalone node).
func New(cfg Config, logger *log.Logger) *Federation {
	if len(cfg.Peers) == 0 && cfg.HubURL == "" {
		return nil
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 5 * time.Second
	}
	if cfg.NodeName == "" {
		cfg.NodeName = "llama-chip"
	}
	return &Federation{
		cfg:    cfg,
		client: &http.Client{Timeout: 4 * time.Second},
		log:    logger,
		routes: map[string]Route{},
		health: map[string]PeerHealth{},
	}
}

// Enabled reports whether federation is active (always true for a non-nil Federation, so
// callers can write `if f.Enabled()` uniformly on a possibly-nil pointer).
func (f *Federation) Enabled() bool { return f != nil && (len(f.cfg.Peers) > 0 || f.cfg.HubURL != "") }

// HubManaged reports whether this node's roster comes from a hub (vs static peers). In hub mode
// the internal static poller is disabled — an external hub client drives ApplyRoster.
func (f *Federation) HubManaged() bool { return f != nil && f.cfg.HubURL != "" }

// NodeName returns this node's federation name.
func (f *Federation) NodeName() string {
	if f == nil {
		return ""
	}
	return f.cfg.NodeName
}

// Advertise returns this node's advertised URL (may be empty).
func (f *Federation) Advertise() string {
	if f == nil {
		return ""
	}
	return f.cfg.Advertise
}

// Token returns the bearer token used to reach peers (may be empty).
func (f *Federation) Token() string {
	if f == nil {
		return ""
	}
	return f.cfg.Token
}

// Start launches the static-peer poll loop until ctx is cancelled. It polls once immediately so
// routes are populated quickly, then on the configured interval. In hub-managed mode it does
// nothing — an external hub client supplies the roster via ApplyRoster.
func (f *Federation) Start(ctx context.Context) {
	if !f.Enabled() {
		return
	}
	if f.HubManaged() {
		f.log.Printf("federation: node %q hub-managed via %s (poll %s)", f.cfg.NodeName, f.cfg.HubURL, f.cfg.PollInterval)
		return
	}
	f.log.Printf("federation: node %q with %d peer(s), poll %s", f.cfg.NodeName, len(f.cfg.Peers), f.cfg.PollInterval)
	go func() {
		f.pollAll(ctx)
		t := time.NewTicker(f.cfg.PollInterval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				f.pollAll(ctx)
			}
		}
	}()
}

// Refresh polls every peer once, synchronously, and rebuilds the route map. Start calls this
// on its own cadence; callers can invoke it to refresh on demand (and tests use it).
func (f *Federation) Refresh(ctx context.Context) {
	if f.Enabled() && !f.HubManaged() {
		f.pollAll(ctx)
	}
}

// RosterGPU is one node's GPU as the hub reports it (JSON tags match the hub's wire shape).
type RosterGPU struct {
	Index    int    `json:"index"`
	Name     string `json:"name"`
	MemUsed  int    `json:"mem_used"`
	MemTotal int    `json:"mem_total"`
	Util     int    `json:"util"`
}

// RosterEntry is one node from a hub's roster (the hub-managed discovery source). Its JSON tags
// match the hub's Node so a roster response decodes straight into a slice of these.
type RosterEntry struct {
	Name     string      `json:"name"`
	MeshAddr string      `json:"mesh_addr"`
	Models   []string    `json:"models"`
	GPUs     []RosterGPU `json:"gpus,omitempty"`
}

// ApplyRoster rebuilds the route map from an externally-supplied roster (hub-managed mode),
// excluding this node itself. The hub client calls this each heartbeat. Like the static poller,
// first entry wins a model-name tie; an absent node simply drops out of the next roster.
func (f *Federation) ApplyRoster(entries []RosterEntry, selfName string) {
	if f == nil {
		return
	}
	routes := map[string]Route{}
	health := map[string]PeerHealth{}
	for _, e := range entries {
		if strings.EqualFold(e.Name, selfName) {
			continue // never route to ourselves
		}
		url := strings.TrimRight(e.MeshAddr, "/")
		if url == "" {
			continue
		}
		health[url] = PeerHealth{Name: e.Name, URL: url, Online: true, Models: e.Models, LastSeen: time.Now()}
		for _, m := range e.Models {
			key := strings.ToLower(strings.TrimSpace(m))
			if key == "" {
				continue
			}
			if _, taken := routes[key]; taken {
				continue
			}
			routes[key] = Route{Model: m, PeerURL: url, PeerName: e.Name}
		}
	}
	f.mu.Lock()
	f.routes = routes
	f.health = health
	f.roster = entries // full roster incl self — drives the pool view
	f.selfName = selfName
	f.mu.Unlock()
}

// Roster returns the full hub roster (all nodes incl self, with GPUs + models) for the pool view,
// and this node's name so callers can flag which entry is self. Empty in static-peer mode.
func (f *Federation) Roster() ([]RosterEntry, string) {
	if f == nil {
		return nil, ""
	}
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]RosterEntry, len(f.roster))
	copy(out, f.roster)
	return out, f.selfName
}

// NodeAddr returns the mesh base URL for a named node — from the hub roster (hub mode) or the
// configured peers (static mode), whichever is populated — and whether it was found with a usable
// address. The router uses it to proxy management calls and node-pinned chats to a specific peer
// over the mesh. Self is included when it appears in the roster, so callers should check IsSelf
// first when they want to avoid a redundant hop.
func (f *Federation) NodeAddr(name string) (string, bool) {
	if f == nil {
		return "", false
	}
	key := strings.TrimSpace(name)
	f.mu.RLock()
	defer f.mu.RUnlock()
	for _, e := range f.roster {
		if strings.EqualFold(e.Name, key) {
			addr := strings.TrimRight(e.MeshAddr, "/")
			return addr, addr != ""
		}
	}
	for _, h := range f.health {
		if strings.EqualFold(h.Name, key) {
			addr := strings.TrimRight(h.URL, "/")
			return addr, addr != ""
		}
	}
	return "", false
}

// IsSelf reports whether name is this node (case-insensitive). Used to skip a redundant
// node-override hop when a client pins a chat to "this node" explicitly. Falls back to the
// configured NodeName when no roster self-name has been applied yet (static-peer mode).
func (f *Federation) IsSelf(name string) bool {
	if f == nil {
		return false
	}
	f.mu.RLock()
	self := f.selfName
	f.mu.RUnlock()
	if self == "" {
		self = f.cfg.NodeName
	}
	return self != "" && strings.EqualFold(strings.TrimSpace(name), self)
}

// pollAll refreshes every peer concurrently, then rebuilds the route map. First-listed peer
// wins a model-name tie (deterministic by config order).
func (f *Federation) pollAll(ctx context.Context) {
	var wg sync.WaitGroup
	results := make([]PeerHealth, len(f.cfg.Peers))
	for i, p := range f.cfg.Peers {
		wg.Add(1)
		go func(i int, p PeerConfig) {
			defer wg.Done()
			results[i] = f.pollPeer(ctx, p)
		}(i, p)
	}
	wg.Wait()

	routes := map[string]Route{}
	health := map[string]PeerHealth{}
	for _, h := range results {
		health[h.URL] = h
		if !h.Online {
			continue
		}
		for _, m := range h.Models {
			key := strings.ToLower(strings.TrimSpace(m))
			if key == "" {
				continue
			}
			if _, taken := routes[key]; taken {
				continue // first peer in config order wins the tie
			}
			routes[key] = Route{Model: m, PeerURL: h.URL, PeerName: h.Name}
		}
	}
	f.mu.Lock()
	f.routes = routes
	f.health = health
	f.mu.Unlock()
}

// pollPeer fetches one peer's local model list. Prefers /api/fed/local (local-only, loop-safe)
// and falls back to /v1/models for resilience against an older peer build.
func (f *Federation) pollPeer(ctx context.Context, p PeerConfig) PeerHealth {
	name := p.Name
	if name == "" {
		name = hostOf(p.URL)
	}
	h := PeerHealth{Name: name, URL: p.URL}

	models, err := f.fetchLocal(ctx, p.URL)
	if err != nil {
		if models, err = f.fetchV1Models(ctx, p.URL); err != nil {
			h.LastErr = err.Error()
			return h
		}
	}
	h.Online = true
	h.Models = models
	h.LastSeen = nowFromCtx(ctx)
	return h
}

func (f *Federation) fetchLocal(ctx context.Context, base string) ([]string, error) {
	var lv localView
	if err := f.getJSON(ctx, base+"/api/fed/local", &lv); err != nil {
		return nil, err
	}
	return lv.Models, nil
}

func (f *Federation) fetchV1Models(ctx context.Context, base string) ([]string, error) {
	var resp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := f.getJSON(ctx, base+"/v1/models", &resp); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(resp.Data))
	for _, d := range resp.Data {
		out = append(out, d.ID)
	}
	return out, nil
}

func (f *Federation) getJSON(ctx context.Context, url string, v any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if f.cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+f.cfg.Token)
	}
	resp, err := f.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s: %s %s", url, resp.Status, strings.TrimSpace(string(b)))
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

// Resolve returns a remote route for a model handle, if a reachable peer serves it. It also
// matches a peer model whose name CONTAINS the handle (the same loose match the local rig
// uses), so role aliases / substrings route too.
func (f *Federation) Resolve(model string) (Route, bool) {
	if !f.Enabled() {
		return Route{}, false
	}
	key := strings.ToLower(strings.TrimSpace(model))
	f.mu.RLock()
	defer f.mu.RUnlock()
	if r, ok := f.routes[key]; ok {
		return r, true
	}
	for k, r := range f.routes {
		if strings.Contains(k, key) {
			return r, true
		}
	}
	return Route{}, false
}

// RemoteModels lists every model reachable on a peer (sorted, for the aggregated /v1/models).
func (f *Federation) RemoteModels() []Route {
	if !f.Enabled() {
		return nil
	}
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]Route, 0, len(f.routes))
	for _, r := range f.routes {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Model < out[j].Model })
	return out
}

// Peers returns the current health of every configured peer (sorted by name, for status/UI).
func (f *Federation) Peers() []PeerHealth {
	if !f.Enabled() {
		return nil
	}
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]PeerHealth, 0, len(f.health))
	for _, h := range f.health {
		out = append(out, h)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func hostOf(rawurl string) string {
	s := strings.TrimPrefix(strings.TrimPrefix(rawurl, "http://"), "https://")
	if i := strings.IndexAny(s, "/:"); i >= 0 {
		return s[:i]
	}
	return s
}

// nowFromCtx returns the current time. (Isolated so tests can swap a deterministic clock if
// ever needed; the real path just uses time.Now.)
func nowFromCtx(_ context.Context) time.Time { return time.Now() }
