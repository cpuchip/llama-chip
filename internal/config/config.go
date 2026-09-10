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
	GPUs      []int    `json:"gpus"`                 // [0] pins to GPU0; [0,1] splits one model across both cards; [] (or omitted) = CPU-only slot (CUDA_VISIBLE_DEVICES="")
	Backend   string   `json:"backend,omitempty"`    // per-slot backend override: a variant ("cuda12") OR a dir holding a self-managed llama-server (e.g. a fresh ggml-org build); "" = the rig's config.Backend
	CtxSize   int      `json:"ctx_size,omitempty"`   // TOTAL context across slots (--ctx-size); per-request = ctx_size / parallel
	Parallel  int      `json:"parallel,omitempty"`   // concurrent request slots (--parallel); default 1. per-slot ctx = ctx_size/parallel
	KVCache   string   `json:"kv_cache,omitempty"`   // "q8_0" (default, halves KV bytes) | "f16" | "q4_0"
	NoFlash   bool     `json:"no_flash,omitempty"`   // flash-attention is on by default; set to disable
	SplitMode string   `json:"split_mode,omitempty"` // for multi-GPU slots: "layer" (default) | "row"
	Port      int      `json:"port,omitempty"`       // backend port; 0 = auto-assign
	MMProj    string   `json:"mmproj,omitempty"`     // explicit multimodal projector (vision/audio) path; "" = auto-detect the co-located mmproj-*.gguf
	NoMMProj  bool     `json:"no_mmproj,omitempty"`  // suppress mmproj auto-detect (load text-only even if a projector is present)
	ExtraArgs []string `json:"extra_args,omitempty"`

	// External slot: an OpenAI-compatible server this rig does NOT launch (a vLLM container, say).
	// The rig only health-checks it (GET <external>/health) and the router proxies to it, so it shows
	// up in /v1/models, in profiles and in the federation roster like any other slot. Alias is
	// required and must be the model name the upstream serves (the request body is forwarded as-is).
	External  string `json:"external,omitempty"`    // base URL of the server root, e.g. http://127.0.0.1:18020 (no /v1)
	APIKey    string `json:"api_key,omitempty"`     // bearer the upstream requires; sent as Authorization
	APIKeyEnv string `json:"api_key_env,omitempty"` // env var holding that bearer (wins over api_key when set)

	// Container slot: an OpenAI-compatible server the rig launches as a Docker container (a vLLM
	// image, say) and then fronts exactly like an external slot. The lab panel builds these: the
	// knobs are the container's environment, so a setting can be tried without editing a launcher.
	// Alias is required (the model name the server inside serves); port is the HOST port the
	// container's server is published on (loopback only).
	Container *Container `json:"container,omitempty"`
}

// Container describes a server the rig runs with `docker run`. Everything the image needs is
// declared here; nothing is read from the host besides the bearer named by the slot's
// api_key_env, which is handed to the container as the environment variable KeyEnvIn without
// ever appearing on the docker command line.
type Container struct {
	Image    string            `json:"image"`                 // e.g. an inference image tag
	Cmd      string            `json:"cmd,omitempty"`         // command run inside the container (bash -c); "" = the image's default
	Pre      string            `json:"pre,omitempty"`         // shell run before Cmd, inside the container: an experiment's one-line edit, a sed on a constant
	Env      map[string]string `json:"env,omitempty"`         // the knobs (MAX_LEN, GPU_UTIL, ...)
	Mounts   []string          `json:"mounts,omitempty"`      // docker -v entries: "host-path-or-volume:/container/path[:ro]"
	Port     int               `json:"port,omitempty"`        // the server's port INSIDE the container (default 18020)
	Extra    []string          `json:"extra_args,omitempty"`  // extra `docker run` flags (--ipc host, --shm-size 64m, ...)
	KeyEnvIn string            `json:"key_env_in,omitempty"`  // container env var that receives the slot's bearer (default VLLM_API_KEY)
	Health   string            `json:"health,omitempty"`      // health path on the server (default /health)
	Served   string            `json:"served,omitempty"`      // the model name the server inside answers to; when it differs from the alias the router rewrites `model` on the way in, so two lab slots of the same image can coexist under different aliases
}

// Lab holds the defaults the lab panel starts from, so a config carries the image, the mounts
// and the key once and the UI only asks for the knobs. Optional; without it the panel is hidden.
type Lab struct {
	Image     string            `json:"image"`
	Mounts    []string          `json:"mounts,omitempty"`
	Extra     []string          `json:"extra_args,omitempty"`
	APIKeyEnv string            `json:"api_key_env,omitempty"` // host env var holding the bearer the server expects
	KeyEnvIn  string            `json:"key_env_in,omitempty"`
	Port      int               `json:"port,omitempty"`     // the server's port inside the container
	Served    string            `json:"served,omitempty"`   // the model name the image's server answers to (see Container.Served)
	BasePort  int               `json:"base_port,omitempty"` // first host port the lab publishes on (default 18100)
	Env       map[string]string `json:"env,omitempty"`      // env every lab launch gets (VLLM_NO_USAGE_STATS=1, ...)
	Presets   []LabPreset       `json:"presets,omitempty"`  // named starting points (cmd + env) shown in the panel
}

// LabPreset is one named starting point in the lab panel: a command inside the image plus the
// environment it wants. The panel lets the knobs be edited before launch.
type LabPreset struct {
	Name string            `json:"name"`
	Cmd  string            `json:"cmd"`
	Env  map[string]string `json:"env,omitempty"`
	Pre  string            `json:"pre,omitempty"`  // shell run inside the container before cmd (an experiment's edit)
	Note string            `json:"note,omitempty"` // what this preset is for, shown beside the launch button
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
	TokenEnv        string `json:"token_env,omitempty"` // env var holding that bearer (wins over token when set and non-empty)
	PollIntervalSec int    `json:"poll_interval_sec,omitempty"`
	Peers           []Peer `json:"peers,omitempty"`
	HubURL          string `json:"hub_url,omitempty"`       // optional coordinator (llama.example.com) for roster + tokens
	HubToken        string `json:"hub_token,omitempty"`     // this node's join token for the hub
	HubTokenEnv     string `json:"hub_token_env,omitempty"` // env var holding the join token (wins over hub_token when set and non-empty); keeps secrets out of a config that gets pasted

	// PeerTokens is an optional per-peer OUTBOUND bearer map (peer/node name -> token) attached
	// when THIS node proxies a chat to that peer. It's how a keyless mesh node still authenticates
	// to a peer that runs its own federation.token — e.g. a public, bearer-gated NOCIX node. Keyed
	// by peer name (matches a static peer's name, a hub-roster entry's name, or a ?node= pin);
	// works in both static and hub-managed modes. A peer with no entry keeps legacy behavior.
	PeerTokens map[string]string `json:"peer_tokens,omitempty"`
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

	// DefaultMaxTokens caps a completion when the caller sends no max_tokens (0 = leave it to the
	// upstream). Born 2026-09-09: a phone chat with no cap ran a reasoning loop to the model's whole
	// context, 3 min 20 s at 300 tok/s, and nothing on the wire said so.
	DefaultMaxTokens int `json:"default_max_tokens,omitempty"`

	// Federation is optional: list peer nodes to pool GPUs across machines (see package fed).
	Federation *Federation `json:"federation,omitempty"`

	// Yield is optional: give a card up to a game and move the work to the fleet (see Yield).
	Yield *Yield `json:"yield,omitempty"`

	// Lab is optional: the defaults behind the lab panel, which launches container slots with
	// editable knobs so a server setting can be tried and felt before it becomes a launcher's default.
	Lab *Lab `json:"lab,omitempty"`
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
	token := f.Token
	if f.TokenEnv != "" {
		if v := os.Getenv(f.TokenEnv); v != "" {
			token = v
		}
	}
	return fed.Config{
		NodeName:     f.NodeName,
		Advertise:    f.Advertise,
		Token:        token,
		Peers:        peers,
		HubURL:       strings.TrimRight(f.HubURL, "/"),
		PollInterval: time.Duration(f.PollIntervalSec) * time.Second,
		PeerTokens:   f.PeerTokens,
	}
}

// ResolvedHubToken is the hub join token after the env override.
func (c *Config) ResolvedHubToken() string {
	if c.Federation == nil {
		return ""
	}
	if c.Federation.HubTokenEnv != "" {
		if v := os.Getenv(c.Federation.HubTokenEnv); v != "" {
			return v
		}
	}
	return c.Federation.HubToken
}

// Yield is the yield-to-the-foreground policy: when something not ours takes VRAM on a card
// (a game), the rig gives the card up, moves the work to the fleet, and takes it back when the
// foreign use is gone.
type Yield struct {
	Enabled    bool `json:"enabled"`
	ForeignMiB int  `json:"foreign_mib,omitempty"` // foreign VRAM that counts as "taken" (default 1024)
	HoldSec    int  `json:"hold_s,omitempty"`      // foreign must persist this long before yielding (default 5)
	RestoreSec int  `json:"restore_s,omitempty"`   // foreign must be gone this long before restoring (default 30)
}

func (y *Yield) ForeignThreshold() int {
	if y == nil || y.ForeignMiB <= 0 {
		return 1024
	}
	return y.ForeignMiB
}

func (y *Yield) Hold() time.Duration {
	if y == nil || y.HoldSec <= 0 {
		return 5 * time.Second
	}
	return time.Duration(y.HoldSec) * time.Second
}

func (y *Yield) Restore() time.Duration {
	if y == nil || y.RestoreSec <= 0 {
		return 30 * time.Second
	}
	return time.Duration(y.RestoreSec) * time.Second
}

// Validate checks what a container slot must carry before anything is launched.
func (ct *Container) Validate(s Slot) error {
	if s.Alias == "" {
		return fmt.Errorf("container slot needs an alias (the model name the server inside serves)")
	}
	if strings.TrimSpace(ct.Image) == "" {
		return fmt.Errorf("container slot %q needs an image", s.Alias)
	}
	for _, m := range ct.Mounts {
		if !strings.Contains(m, ":/") {
			return fmt.Errorf("container slot %q: mount %q is not host:/container[:ro]", s.Alias, m)
		}
	}
	return nil
}

// ContainerPort is the server's port inside the container (18020 unless set).
func (ct *Container) ContainerPort() int {
	if ct.Port > 0 {
		return ct.Port
	}
	return 18020
}

// KeyVar is the container environment variable that receives the slot's bearer.
func (ct *Container) KeyVar() string {
	if ct.KeyEnvIn != "" {
		return ct.KeyEnvIn
	}
	return "VLLM_API_KEY"
}

// HealthPath is the path polled for readiness on the server inside the container.
func (ct *Container) HealthPath() string {
	if ct.Health != "" {
		return ct.Health
	}
	return "/health"
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
		if s.Container != nil {
			if err := s.Container.Validate(s); err != nil {
				return err
			}
		} else if s.External != "" {
			if s.Alias == "" {
				return fmt.Errorf("external slot %q needs an alias (the model name the upstream serves)", s.External)
			}
		} else if s.Model == "" {
			return fmt.Errorf("a slot has no model")
		}
		// An empty gpus ([] or omitted) is intentional: a CPU-only slot (CUDA_VISIBLE_DEVICES="").
		// [0] pins to a card; [0,1] splits across two. No lower bound to enforce.
		if names[strings.ToLower(s.Name())] {
			return fmt.Errorf("duplicate slot name %q", s.Name())
		}
		names[strings.ToLower(s.Name())] = true
	}
	return nil
}
