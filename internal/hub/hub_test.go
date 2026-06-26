package hub

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func newTestStore(t *testing.T) *Store {
	t.Helper()
	s, err := NewStore("", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestMintVerifyRevoke(t *testing.T) {
	s := newTestStore(t)
	tok, plaintext, err := s.Mint(KindNode, "my laptop", "admin", nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := s.Verify(plaintext); !ok || got.ID != tok.ID || got.Kind != KindNode {
		t.Fatalf("verify failed: %+v ok=%v", got, ok)
	}
	if _, ok := s.Verify("lck_wrong"); ok {
		t.Fatal("a bogus token verified")
	}
	if err := s.Revoke(tok.ID); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Verify(plaintext); ok {
		t.Fatal("a revoked token still verifies")
	}
}

func TestPlaintextNeverStored(t *testing.T) {
	s := newTestStore(t)
	_, plaintext, _ := s.Mint(KindUser, "u", "admin", nil)
	for _, tk := range s.Tokens() {
		if tk.Hash != "" {
			t.Fatal("Tokens() leaked a hash")
		}
	}
	// the store holds only the hash, never the plaintext
	for _, tk := range s.tokens {
		if tk.Hash == plaintext || tk.Hash == "" {
			t.Fatalf("token stored badly: hash=%q", tk.Hash)
		}
	}
}

func TestEnsureAdminIdempotent(t *testing.T) {
	s := newTestStore(t)
	created, err := s.EnsureAdmin("super-secret-admin-key")
	if err != nil || !created {
		t.Fatalf("first EnsureAdmin should create: created=%v err=%v", created, err)
	}
	created2, _ := s.EnsureAdmin("super-secret-admin-key")
	if created2 {
		t.Fatal("second EnsureAdmin should be a no-op")
	}
	if t2, ok := s.Verify("super-secret-admin-key"); !ok || t2.Kind != KindAdmin {
		t.Fatal("admin key should verify as admin")
	}
	if b, _ := s.EnsureAdmin(""); b {
		t.Fatal("blank admin key should be a no-op")
	}
}

func TestTokenPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokens.json")
	s1, _ := NewStore(path, time.Minute)
	_, plaintext, _ := s1.Mint(KindNode, "node1", "admin", nil)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("token file not written: %v", err)
	}
	// reload from disk — token must survive a restart
	s2, err := NewStore(path, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := s2.Verify(plaintext); !ok {
		t.Fatal("token did not persist across reload")
	}
}

func TestRosterRegisterAndPrune(t *testing.T) {
	s := newTestStore(t)
	now := time.Unix(1_000_000, 0)
	s.clock = func() time.Time { return now }
	s.nodeTTL = 30 * time.Second

	s.Register(Node{Name: "home-gpu", MeshAddr: "http://100.84.0.2:8090", Models: []string{"qwen3.6-35b-a3b"}}, "owner")
	if r := s.Roster(); len(r) != 1 || r[0].Name != "home-gpu" || len(r[0].Models) != 1 {
		t.Fatalf("expected 1 node with models, got %+v", r)
	}
	// advance past the TTL with no heartbeat → pruned
	now = now.Add(31 * time.Second)
	if r := s.Roster(); len(r) != 0 {
		t.Fatalf("stale node should be pruned, got %+v", r)
	}
}

// --- HTTP layer ---

func newTestServer(t *testing.T) (*httptest.Server, *Store) {
	t.Helper()
	s := newTestStore(t)
	srv := httptest.NewServer(NewServer(s, log.New(io.Discard, "", 0)).Handler())
	t.Cleanup(srv.Close)
	return srv, s
}

func do(t *testing.T, method, url, token string, body any) (int, map[string]any) {
	t.Helper()
	var rdr io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rdr = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, url, rdr)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func TestHTTPAuthGates(t *testing.T) {
	srv, s := newTestServer(t)
	s.EnsureAdmin("admin-key")
	_, nodeTok, _ := s.Mint(KindNode, "node1", "admin", nil)

	// no token → 401
	if code, _ := do(t, "GET", srv.URL+"/api/roster", "", nil); code != 401 {
		t.Fatalf("no token should 401, got %d", code)
	}
	// node token cannot mint (admin-only) → 403
	if code, _ := do(t, "GET", srv.URL+"/api/tokens", nodeTok, nil); code != 403 {
		t.Fatalf("node token on admin endpoint should 403, got %d", code)
	}
	// admin can list tokens
	if code, _ := do(t, "GET", srv.URL+"/api/tokens", "admin-key", nil); code != 200 {
		t.Fatalf("admin token list should 200, got %d", code)
	}
	// /healthz is open
	if code, _ := do(t, "GET", srv.URL+"/healthz", "", nil); code != 200 {
		t.Fatalf("/healthz should be open, got %d", code)
	}
}

func TestHTTPRegisterThenRoster(t *testing.T) {
	srv, s := newTestServer(t)
	_, nodeTok, _ := s.Mint(KindNode, "home", "admin", nil)

	code, out := do(t, "POST", srv.URL+"/api/register", nodeTok, map[string]any{
		"name": "home-gpu", "mesh_addr": "http://100.84.0.2:8090/", "models": []string{"qwen3.6-35b-a3b", "gemma-4-26b-a4b"},
	})
	if code != 200 || out["ok"] != true {
		t.Fatalf("register should 200/ok, got %d %+v", code, out)
	}
	// the register response carries the roster, and mesh_addr trailing slash is trimmed
	roster, _ := out["roster"].([]any)
	if len(roster) != 1 {
		t.Fatalf("register should echo a 1-node roster, got %+v", out["roster"])
	}
	n := roster[0].(map[string]any)
	if n["mesh_addr"] != "http://100.84.0.2:8090" {
		t.Fatalf("mesh_addr should be trimmed, got %v", n["mesh_addr"])
	}

	// a user token can read the roster but not register
	_, userTok, _ := s.Mint(KindUser, "client", "admin", nil)
	if code, _ := do(t, "GET", srv.URL+"/api/roster", userTok, nil); code != 200 {
		t.Fatalf("user roster read should 200, got %d", code)
	}
	if code, _ := do(t, "POST", srv.URL+"/api/register", userTok, map[string]any{"name": "x", "mesh_addr": "http://y:1"}); code != 403 {
		t.Fatalf("user token should not be allowed to register, got %d", code)
	}
}

func TestHTTPMintReturnsPlaintextOnce(t *testing.T) {
	srv, s := newTestServer(t)
	s.EnsureAdmin("admin-key")
	code, out := do(t, "POST", srv.URL+"/api/tokens", "admin-key", map[string]any{"kind": "node", "label": "node-1"})
	if code != 201 {
		t.Fatalf("mint should 201, got %d", code)
	}
	plaintext, _ := out["token"].(string)
	if plaintext == "" {
		t.Fatal("mint should return the plaintext token once")
	}
	// and that token immediately works
	if _, ok := s.Verify(plaintext); !ok {
		t.Fatal("minted token should verify")
	}
}
