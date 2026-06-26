// Package hubclient is the NODE side of the hub: it registers this llama-chip node with a
// coordinator (llama.example.com) on a heartbeat — announcing the models it serves and its GPU
// state — and feeds the returned roster into the federation, so the node can route to every other
// node in the group over the mesh. The hub is discovery + auth only; inference stays peer-to-peer.
package hubclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/cpuchip/llama-chip/internal/fed"
)

// errForbidden means the token may read the roster but not register (a user/client token).
var errForbidden = errors.New("forbidden")

// GPUStat is one GPU as reported to the hub (JSON tags match the hub's wire shape).
type GPUStat struct {
	Index    int    `json:"index"`
	Name     string `json:"name"`
	MemUsed  int    `json:"mem_used"`
	MemTotal int    `json:"mem_total"`
	Util     int    `json:"util"`
}

// LocalState is what this node currently serves — the heartbeat payload's variable part.
type LocalState struct {
	Models []string
	GPUs   []GPUStat
}

// Client heartbeats this node into the hub and applies the roster to the federation.
type Client struct {
	hubURL    string
	token     string
	nodeName  string
	advertise string // this node's mesh address (the URL peers route to)
	interval  time.Duration
	fedn      *fed.Federation
	local     func() LocalState
	http      *http.Client
	log       *log.Logger
}

// New builds a hub client. Returns nil if hubURL is empty (no hub configured).
func New(hubURL, token, nodeName, advertise string, interval time.Duration, fedn *fed.Federation, local func() LocalState, logger *log.Logger) *Client {
	if strings.TrimSpace(hubURL) == "" {
		return nil
	}
	if interval <= 0 {
		interval = 5 * time.Second
	}
	return &Client{
		hubURL: strings.TrimRight(hubURL, "/"), token: token, nodeName: nodeName,
		advertise: strings.TrimRight(advertise, "/"), interval: interval, fedn: fedn,
		local: local, http: &http.Client{Timeout: 6 * time.Second}, log: logger,
	}
}

// Run registers immediately, then heartbeats until ctx is cancelled. A hub blip does NOT clear
// the last-known roster — peers reachable over the mesh stay routable even if the hub flickers.
func (c *Client) Run(ctx context.Context) {
	if c == nil {
		return
	}
	c.log.Printf("hub: registering with %s as %q (mesh %s)", c.hubURL, c.nodeName, c.advertise)
	if err := c.registerOnce(ctx); err != nil {
		c.log.Printf("hub: initial register failed: %v (will retry)", err)
	}
	go func() {
		t := time.NewTicker(c.interval)
		defer t.Stop()
		// Log STATE TRANSITIONS, not every tick: one line when the hub link goes degraded, one
		// when it recovers (with how long it was down + how many beats it missed). The loop never
		// stops retrying — but previously every failed beat logged "register failed (keeping last
		// roster)", spamming the log while silently recovering, so an operator couldn't tell
		// whether the multi-box link was actually down or back up.
		fails := 0
		var downSince time.Time
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				if err := c.registerOnce(ctx); err != nil {
					if fails == 0 {
						downSince = time.Now()
						c.log.Printf("hub: register failing (%v) — keeping last roster, retrying every %s", err, c.interval)
					}
					fails++
				} else if fails > 0 {
					c.log.Printf("hub: re-registered OK after %d missed beat(s) over %s", fails, time.Since(downSince).Round(time.Second))
					fails = 0
				}
			}
		}
	}()
}

// registerOnce announces this node and applies the returned roster. A contributor (node token)
// POSTs /api/register (announce + roster in one). A client (user token) can't register, so on a
// 403 it falls back to a read-only GET /api/roster — it still routes to the pool, just contributes
// nothing.
func (c *Client) registerOnce(ctx context.Context) error {
	roster, err := c.postRegister(ctx)
	if errors.Is(err, errForbidden) {
		roster, err = c.getRoster(ctx)
	}
	if err != nil {
		return err
	}
	c.fedn.ApplyRoster(roster, c.nodeName)
	return nil
}

func (c *Client) postRegister(ctx context.Context) ([]fed.RosterEntry, error) {
	st := LocalState{}
	if c.local != nil {
		st = c.local()
	}
	if st.Models == nil {
		st.Models = []string{}
	}
	payload := map[string]any{
		"name": c.nodeName, "mesh_addr": c.advertise, "models": st.Models, "gpus": st.GPUs,
	}
	body, _ := json.Marshal(payload)
	resp, err := c.send(ctx, http.MethodPost, "/api/register", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusForbidden {
		return nil, errForbidden
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("register %s: %s %s", c.hubURL, resp.Status, strings.TrimSpace(string(b)))
	}
	var out struct {
		Roster []fed.RosterEntry `json:"roster"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Roster, nil
}

func (c *Client) getRoster(ctx context.Context) ([]fed.RosterEntry, error) {
	resp, err := c.send(ctx, http.MethodGet, "/api/roster", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("roster %s: %s %s", c.hubURL, resp.Status, strings.TrimSpace(string(b)))
	}
	var out struct {
		Nodes []fed.RosterEntry `json:"nodes"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Nodes, nil
}

func (c *Client) send(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.hubURL+path, rdr)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	return c.http.Do(req)
}
