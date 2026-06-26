package hubclient

import (
	"context"
	"io"
	"log"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cpuchip/llama-chip/internal/fed"
	"github.com/cpuchip/llama-chip/internal/hub"
)

// stands up a real hub, registers a node, and proves the client pulls the roster into the
// federation — routing to a peer's model while excluding its own.
func TestHubClientRegistersAndAppliesRoster(t *testing.T) {
	store, _ := hub.NewStore("", time.Minute)
	_, nodeTok, _ := store.Mint(hub.KindNode, "laptop", "admin", nil)
	srv := httptest.NewServer(hub.NewServer(store, log.New(io.Discard, "", 0)).Handler())
	defer srv.Close()

	// another peer is already in the pool (home box with the big model)
	store.Register(hub.Node{Name: "home-gpu", MeshAddr: "http://100.84.0.2:8090", Models: []string{"qwen3.6-35b-a3b"}}, "admin")

	fedn := fed.New(fed.Config{NodeName: "laptop", HubURL: srv.URL}, log.New(io.Discard, "", 0))
	if fedn == nil || !fedn.HubManaged() {
		t.Fatal("federation should be enabled + hub-managed")
	}

	c := New(srv.URL, nodeTok, "laptop", "http://100.84.0.3:8090", time.Second, fedn,
		func() LocalState { return LocalState{Models: []string{"gemma-4-26b-a4b"}} },
		log.New(io.Discard, "", 0))
	if err := c.registerOnce(context.Background()); err != nil {
		t.Fatal(err)
	}

	// the laptop can now route the home box's model over the mesh...
	if r, ok := fedn.Resolve("qwen3.6-35b-a3b"); !ok || r.PeerName != "home-gpu" || r.PeerURL != "http://100.84.0.2:8090" {
		t.Fatalf("expected qwen routed to home-gpu, got %+v ok=%v", r, ok)
	}
	// ...but never routes its OWN model to a peer (self excluded from the roster)
	if _, ok := fedn.Resolve("gemma-4-26b-a4b"); ok {
		t.Fatal("a node must not route its own model remotely")
	}
}

// a client with only a USER token can't register, but still discovers the pool read-only.
func TestHubClientUserTokenReadsRoster(t *testing.T) {
	store, _ := hub.NewStore("", time.Minute)
	_, userTok, _ := store.Mint(hub.KindUser, "my client", "admin", nil)
	srv := httptest.NewServer(hub.NewServer(store, log.New(io.Discard, "", 0)).Handler())
	defer srv.Close()
	store.Register(hub.Node{Name: "home-gpu", MeshAddr: "http://100.84.0.2:8090", Models: []string{"qwen3.6-35b-a3b"}}, "admin")

	fedn := fed.New(fed.Config{NodeName: "client-1", HubURL: srv.URL}, log.New(io.Discard, "", 0))
	c := New(srv.URL, userTok, "client-1", "http://100.84.0.4:8090", time.Second, fedn,
		func() LocalState { return LocalState{} }, log.New(io.Discard, "", 0))
	if err := c.registerOnce(context.Background()); err != nil {
		t.Fatalf("user-token client should discover the pool via the roster fallback, got %v", err)
	}
	if _, ok := fedn.Resolve("qwen3.6-35b-a3b"); !ok {
		t.Fatal("client should route to the pool's model after the roster fallback")
	}
}

func TestHubClientBadTokenErrors(t *testing.T) {
	store, _ := hub.NewStore("", time.Minute)
	srv := httptest.NewServer(hub.NewServer(store, log.New(io.Discard, "", 0)).Handler())
	defer srv.Close()
	fedn := fed.New(fed.Config{NodeName: "x", HubURL: srv.URL}, log.New(io.Discard, "", 0))
	c := New(srv.URL, "lck_bogus", "x", "http://y:1", time.Second, fedn,
		func() LocalState { return LocalState{} }, log.New(io.Discard, "", 0))
	if err := c.registerOnce(context.Background()); err == nil {
		t.Fatal("expected register to fail with an invalid token")
	}
}

func TestNewNilWithoutHub(t *testing.T) {
	if New("", "t", "n", "a", time.Second, nil, nil, log.New(io.Discard, "", 0)) != nil {
		t.Fatal("no hub URL should yield a nil client")
	}
}
