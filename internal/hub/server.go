package hub

import (
	"embed"
	"encoding/json"
	"io/fs"
	"log"
	"net/http"
	"strings"
)

//go:embed static/*
var staticFS embed.FS

// Server is the hub's HTTP surface: a token-gated control plane + a small admin UI.
type Server struct {
	store *Store
	log   *log.Logger
}

func NewServer(store *Store, logger *log.Logger) *Server {
	return &Server{store: store, log: logger}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, 200, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/api/me", s.me)             // any valid token → its role
	mux.HandleFunc("/api/register", s.register) // node token → upsert + roster
	mux.HandleFunc("/api/roster", s.roster)     // any valid token → online nodes
	mux.HandleFunc("/api/tokens", s.tokens)     // admin: GET list / POST mint
	mux.HandleFunc("/api/tokens/revoke", s.revoke)
	sub, _ := fs.Sub(staticFS, "static")
	mux.Handle("/", http.FileServer(http.FS(sub)))
	return mux
}

// authed extracts the bearer token and resolves it. Writes 401 and returns ok=false if missing
// or invalid.
func (s *Server) authed(w http.ResponseWriter, r *http.Request) (*Token, bool) {
	bearer := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
	t, ok := s.store.Verify(bearer)
	if !ok {
		writeErr(w, 401, "valid token required")
		return nil, false
	}
	return t, true
}

func (s *Server) adminOnly(w http.ResponseWriter, r *http.Request) (*Token, bool) {
	t, ok := s.authed(w, r)
	if !ok {
		return nil, false
	}
	if t.Kind != KindAdmin {
		writeErr(w, 403, "admin token required")
		return nil, false
	}
	return t, true
}

func (s *Server) me(w http.ResponseWriter, r *http.Request) {
	t, ok := s.authed(w, r)
	if !ok {
		return
	}
	writeJSON(w, 200, map[string]any{"kind": t.Kind, "label": t.Label})
}

func (s *Server) register(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, 405, "POST only")
		return
	}
	t, ok := s.authed(w, r)
	if !ok {
		return
	}
	if t.Kind != KindAdmin && t.Kind != KindNode {
		writeErr(w, 403, "node or admin token required to register")
		return
	}
	var n Node
	if err := json.NewDecoder(r.Body).Decode(&n); err != nil {
		writeErr(w, 400, "bad node json: "+err.Error())
		return
	}
	if strings.TrimSpace(n.Name) == "" || strings.TrimSpace(n.MeshAddr) == "" {
		writeErr(w, 400, "node needs a name and mesh_addr")
		return
	}
	n.MeshAddr = strings.TrimRight(n.MeshAddr, "/")
	s.store.Register(n, t.Label)
	writeJSON(w, 200, map[string]any{"ok": true, "roster": s.store.Roster()})
}

func (s *Server) roster(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.authed(w, r); !ok {
		return
	}
	writeJSON(w, 200, map[string]any{"nodes": s.store.Roster()})
}

func (s *Server) tokens(w http.ResponseWriter, r *http.Request) {
	t, ok := s.adminOnly(w, r)
	if !ok {
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, 200, map[string]any{"tokens": s.store.Tokens()})
	case http.MethodPost:
		var body struct {
			Kind  Kind     `json:"kind"`
			Label string   `json:"label"`
			Scope []string `json:"scope"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeErr(w, 400, "bad json: "+err.Error())
			return
		}
		if body.Kind == "" {
			body.Kind = KindNode
		}
		tok, plaintext, err := s.store.Mint(body.Kind, body.Label, t.Label, body.Scope)
		if err != nil {
			writeErr(w, 400, err.Error())
			return
		}
		// the plaintext is returned ONCE, here — it is never stored or shown again
		writeJSON(w, 201, map[string]any{"token": plaintext, "id": tok.ID, "kind": tok.Kind, "label": tok.Label})
	default:
		writeErr(w, 405, "GET or POST")
	}
}

func (s *Server) revoke(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, 405, "POST only")
		return
	}
	if _, ok := s.adminOnly(w, r); !ok {
		return
	}
	var body struct {
		ID string `json:"id"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	if err := s.store.Revoke(body.ID); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true, "revoked": body.ID})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": map[string]string{"message": strings.TrimSpace(msg)}})
}
