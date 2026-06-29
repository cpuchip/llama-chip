// Package config defines the rig: which models load on which GPU(s), at what context,
// and how the router fronts them.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/cpuchip/llama-chip/internal/fed"
)

// Slot is one model to load, pinned to one or more GPUs.
type Slot struct {
	Model     string   `json:"model"`                // model handle (ID or unique name substring) from `llama-chip models`
	Alias     string   `json:"alias,omitempty"`      // the name clients use in the OpenAI `model` field (defaults to Model)
	GPUs      []int    `json:"gpus"`                 // [0] pins to GPU0; [0,1] splits one model across both cards
	CtxSize   int      `json:"ctx_size,omitempty"`   // TOTAL context across slots (--ctx-size); per-request = ctx_size / parallel
	Parallel  int      `json:"parallel,omitempty"`   // concurrent request slots (--parallel); default 1. per-slot ctx = ctx_size/parallel
	KVCache   string   `json:"kv_cache,omitempty"`   // "q8_0" (default, halves KV bytes) | "f16" | "q4_0"
	NoFlash   bool     `json:"no_flash,omitempty"`   // flash-attention is on by default; set to disable
	SplitMode string   `json:"split_mode,omitempty"` // for multi-GPU slots: "layer" (default) | "row"
	Port      int      `json:"port,omitempty"`       // backend port; 0 = auto-assign
	MMProj    string   `json:"mmproj,omitempty"`     // explicit multimodal projector (vision/audio) path; "" = auto-detect the co-located mmproj-*.gguf
	NoMMProj  bool     `json:"no_mmproj,omitempty"`  // suppress mmproj auto-detect (load text-only even if a projector is present)
	ExtraArgs []string `json:"extra_args,omitempty"`
}

// Name is the handle clients address this slot by.
func (s Slot) Name() string {
	if s.Alias != "" {
		return s.Alias
	}
	return s.Model
}

// Peer is one reachable llama-chip node in the federation.
type Peer struct {
	Name string `json:"name,omitempty"` // friendly label; defaults to the host in URL
	URL  string `json:"url"`            // base URL, e.g. http://100.84.0.3:8090 (a mesh IP)
}

// Federation pools GPUs across machines behind this node's one endpoint. When peers are
// listed (or a hub is set), this node routes any model it can't serve locally to a peer that
// can — local-first, so the node always works on its own GPUs even when every peer is offline.
// See package fed. Two discovery modes: static `peers` (LAN / hand-listed), or a `hub_url`
// (llama.example.com) that supplies the live roster and issues join tokens.
type Federation struct {
	NodeName        string `json:"node_name,omitempty"` // this node's name (status + gossip)
	Advertise       string `json:"advertise,omitempty"` // URL peers use to reach THIS node (its mesh address)
	Token           string `json:"token,omitempty"`     // optional bearer; when set, peer requests must carry it
	PollIntervalSec int    `json:"poll_interval_sec,omitempty"`
	Peers           []Peer `json:"peers,omitempty"`
	HubURL          string `json:"hub_url,omitempty"`   // optional coordinator (llama.example.com) for roster + tokens
	HubToken        string `json:"hub_token,omitempty"` // this node's join token for the hub
}

// Config is the whole rig.
type Config struct {
	Backend     string `json:"backend,omitempty"`      // variant: "cuda12" (default) | "cuda" | "rocm" | "vulkan" | "cpu" | or an explicit dir
	Listen      string `json:"listen,omitempty"`       // router address, default ":8090" (drop-in for FlexLLama)
	BackendsDir string `json:"backends_dir,omitempty"` // override LM Studio backends dir
	ModelsDir   string `json:"models_dir,omitempty"`   // override LM Studio models dir
	BasePort    int    `json:"base_port,omitempty"`    // first auto-assigned backend port, default 9100
	Slots       []Slot `json:"slots"`

	// Profiles are named GPU layouts you can switch between at runtime (the dynamic-management
	// vision): e.g. "diverse" = qwen on GPU0 + gemma on GPU1, "big" = one model split across both.
	// Applying a profile unloads everything and loads its slots. The substrate switches profiles
	// (or calls /api/ensure) to load what a work phase needs.
	Profiles map[string][]Slot `json:"profiles,omitempty"`

	// DefaultProfile, when set and Slots is empty, is applied on startup so the rig boots into a
	// known layout (e.g. "dance-moe") instead of coming up empty and waiting for a manual
	// /api/profile call. Lets a restart land on the right models without "which button do I press".
	DefaultProfile string `json:"default_profile,omitempty"`

	// Federation is optional: list peer nodes to pool GPUs across machines (see package fed).
	Federation *Federation `json:"federation,omitempty"`
}

// FedConfig converts the JSON federation block into a fed.Config. Returns the zero value
// when no peers are configured (federation off).
func (c *Config) FedConfig() fed.Config {
	if c.Federation == nil {
		return fed.Config{}
	}
	f := c.Federation
	peers := make([]fed.PeerConfig, 0, len(f.Peers))
	for _, p := range f.Peers {
		peers = append(peers, fed.PeerConfig{Name: p.Name, URL: strings.TrimRight(p.URL, "/")})
	}
	return fed.Config{
		NodeName:     f.NodeName,
		Advertise:    f.Advertise,
		Token:        f.Token,
		Peers:        peers,
		HubURL:       strings.TrimRight(f.HubURL, "/"),
		PollInterval: time.Duration(f.PollIntervalSec) * time.Second,
	}
}

// Load reads a config file and fills defaults.
func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	c.applyDefaults()
	return &c, c.validate()
}

func (c *Config) applyDefaults() {
	if c.Backend == "" {
		c.Backend = "cuda12"
	}
	if c.Listen == "" {
		c.Listen = ":8090"
	}
	if c.BasePort == 0 {
		c.BasePort = 9100
	}
	port := c.BasePort
	seen := map[string]bool{}
	for i := range c.Slots {
		if c.Slots[i].KVCache == "" {
			c.Slots[i].KVCache = "q8_0"
		}
		if c.Slots[i].SplitMode == "" {
			c.Slots[i].SplitMode = "layer"
		}
		if c.Slots[i].Port == 0 {
			c.Slots[i].Port = port
			port++
		}
		_ = seen
	}
}

func (c *Config) validate() error {
	names := map[string]bool{}
	for _, s := range c.Slots {
		if s.Model == "" {
			return fmt.Errorf("a slot has no model")
		}
		if len(s.GPUs) == 0 {
			return fmt.Errorf("slot %q must pin at least one gpu (gpus: [0] or [0,1])", s.Name())
		}
		if names[strings.ToLower(s.Name())] {
			return fmt.Errorf("duplicate slot name %q", s.Name())
		}
		names[strings.ToLower(s.Name())] = true
	}
	return nil
}
