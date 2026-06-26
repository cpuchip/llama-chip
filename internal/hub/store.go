// Package hub is the llama-chip coordination service (llama.example.com): the control plane for
// a federated GPU pool. It manages access TOKENS (an admin mints node/user join tokens) and a
// live ROSTER of which nodes are online, what models each has loaded, and their free GPU — so a
// group of machines (yours + co-workers') can discover each other and share compute.
//
// It is CONTROL PLANE ONLY: it never carries inference traffic. Nodes route to each other
// peer-to-peer over a mesh (NetBird); the hub just tells them who's in the group and where.
//
// Storage is deliberately dependency-free: tokens persist to a JSON file (low volume, mutex-
// guarded, write-through); the roster is in-memory and rebuilt from node heartbeats. Keeps the
// hub a single pure-stdlib Go binary, like the rest of llama-chip.
package hub

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base32"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// Kind is a token's role.
type Kind string

const (
	KindAdmin Kind = "admin" // full control: mint/revoke tokens, see everything
	KindNode  Kind = "node"  // a GPU node may register + read the roster
	KindUser  Kind = "user"  // a human/client may read the roster (route through the pool)
)

// Token is a minted credential. The plaintext is shown only once at mint; only its hash is stored.
type Token struct {
	ID        string     `json:"id"`
	Hash      string     `json:"hash"` // sha256(plaintext) hex — the plaintext is never stored
	Kind      Kind       `json:"kind"`
	Label     string     `json:"label"`
	Scope     []string   `json:"scope,omitempty"` // reserved: pin private intents / node-group limits
	CreatedAt time.Time  `json:"created_at"`
	CreatedBy string     `json:"created_by,omitempty"`
	RevokedAt *time.Time `json:"revoked_at,omitempty"`
}

func (t *Token) active() bool { return t != nil && t.RevokedAt == nil }

// GPUStat is one GPU as a node reports it (mirrors internal/gpu, decoupled for the wire).
type GPUStat struct {
	Index    int    `json:"index"`
	Name     string `json:"name"`
	MemUsed  int    `json:"mem_used"`
	MemTotal int    `json:"mem_total"`
	Util     int    `json:"util"`
}

// Node is one online GPU node in the roster (ephemeral — rebuilt from heartbeats).
type Node struct {
	Name     string    `json:"name"`
	MeshAddr string    `json:"mesh_addr"` // base URL peers route to, e.g. http://100.84.0.2:8090
	Models   []string  `json:"models"`    // models this node serves locally
	GPUs     []GPUStat `json:"gpus"`
	Owner    string    `json:"owner,omitempty"` // token label that registered it (who contributed it)
	LastSeen time.Time `json:"last_seen"`
}

// Store holds tokens (persisted) and the live roster (in-memory).
type Store struct {
	mu      sync.RWMutex
	path    string // tokens JSON file ("" = in-memory only, for tests)
	tokens  map[string]*Token
	roster  map[string]*Node // keyed by node name
	nodeTTL time.Duration
	clock   func() time.Time
}

// NewStore loads (or creates) the token file and prepares an empty roster. nodeTTL is how long a
// node stays in the roster after its last heartbeat.
func NewStore(path string, nodeTTL time.Duration) (*Store, error) {
	if nodeTTL <= 0 {
		nodeTTL = 30 * time.Second
	}
	s := &Store{
		path:    path,
		tokens:  map[string]*Token{},
		roster:  map[string]*Node{},
		nodeTTL: nodeTTL,
		clock:   time.Now,
	}
	if path != "" {
		if err := s.load(); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func (s *Store) load() error {
	b, err := os.ReadFile(s.path)
	if os.IsNotExist(err) {
		return nil // fresh store
	}
	if err != nil {
		return err
	}
	var toks []*Token
	if err := json.Unmarshal(b, &toks); err != nil {
		return fmt.Errorf("parse token file %s: %w", s.path, err)
	}
	for _, t := range toks {
		s.tokens[t.ID] = t
	}
	return nil
}

// saveLocked writes the token file (caller holds the write lock). Atomic via temp+rename.
func (s *Store) saveLocked() error {
	if s.path == "" {
		return nil
	}
	toks := make([]*Token, 0, len(s.tokens))
	for _, t := range s.tokens {
		toks = append(toks, t)
	}
	sort.Slice(toks, func(i, j int) bool { return toks[i].CreatedAt.Before(toks[j].CreatedAt) })
	b, err := json.MarshalIndent(toks, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

func hashToken(plaintext string) string {
	sum := sha256.Sum256([]byte(plaintext))
	return hex.EncodeToString(sum[:])
}

// EnsureAdmin makes sure a usable admin token exists for the given plaintext key (seeded from the
// environment). Idempotent — if a token already hashes to it, nothing changes. Returns true if it
// created one. A blank key is a no-op.
func (s *Store) EnsureAdmin(plaintext string) (bool, error) {
	if strings.TrimSpace(plaintext) == "" {
		return false, nil
	}
	h := hashToken(plaintext)
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, t := range s.tokens {
		if t.Hash == h && t.Kind == KindAdmin && t.active() {
			return false, nil
		}
	}
	id := newID()
	s.tokens[id] = &Token{
		ID: id, Hash: h, Kind: KindAdmin, Label: "admin (env)", CreatedAt: s.clock(),
	}
	return true, s.saveLocked()
}

// Mint creates a new token and returns it WITH the plaintext (shown once). createdBy is the label
// of the admin token that minted it.
func (s *Store) Mint(kind Kind, label, createdBy string, scope []string) (*Token, string, error) {
	switch kind {
	case KindAdmin, KindNode, KindUser:
	default:
		return nil, "", fmt.Errorf("unknown token kind %q", kind)
	}
	plaintext := newSecret()
	t := &Token{
		ID: newID(), Hash: hashToken(plaintext), Kind: kind, Label: label,
		Scope: scope, CreatedAt: s.clock(), CreatedBy: createdBy,
	}
	s.mu.Lock()
	s.tokens[t.ID] = t
	err := s.saveLocked()
	s.mu.Unlock()
	return t, plaintext, err
}

// Verify resolves a plaintext token to its (active) record. ok=false if unknown or revoked.
func (s *Store) Verify(plaintext string) (*Token, bool) {
	if plaintext == "" {
		return nil, false
	}
	h := hashToken(plaintext)
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, t := range s.tokens {
		if t.Hash == h && t.active() {
			return t, true
		}
	}
	return nil, false
}

// Revoke marks a token revoked by ID.
func (s *Store) Revoke(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tokens[id]
	if !ok {
		return fmt.Errorf("no token %q", id)
	}
	now := s.clock()
	t.RevokedAt = &now
	return s.saveLocked()
}

// Tokens lists every token (hashes masked) for the admin UI, newest first.
func (s *Store) Tokens() []Token {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Token, 0, len(s.tokens))
	for _, t := range s.tokens {
		c := *t
		c.Hash = "" // never expose the hash
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.After(out[j].CreatedAt) })
	return out
}

// Register upserts a node into the roster from a heartbeat. owner is the registering token's label.
func (s *Store) Register(n Node, owner string) {
	n.Owner = owner
	n.LastSeen = s.clock()
	if n.Models == nil {
		n.Models = []string{}
	}
	s.mu.Lock()
	cp := n
	s.roster[n.Name] = &cp
	s.mu.Unlock()
}

// Roster returns the currently-online nodes (pruning any past the TTL), sorted by name.
func (s *Store) Roster() []Node {
	cutoff := s.clock().Add(-s.nodeTTL)
	s.mu.Lock()
	for name, n := range s.roster {
		if n.LastSeen.Before(cutoff) {
			delete(s.roster, name)
		}
	}
	out := make([]Node, 0, len(s.roster))
	for _, n := range s.roster {
		out = append(out, *n)
	}
	s.mu.Unlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// newSecret returns a high-entropy token like "lck_<base32>". 30 bytes ≈ 240 bits.
func newSecret() string {
	b := make([]byte, 30)
	_, _ = rand.Read(b)
	enc := strings.ToLower(base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(b))
	return "lck_" + enc
}
