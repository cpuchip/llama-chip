// llama-hub — the llama-chip coordination service (control plane for a shared GPU pool).
//
// It hands out access tokens (an admin mints node/user join tokens) and tracks a live roster of
// which nodes are online, what models each serves, and their free GPU — so a group of machines
// (yours + co-workers', across LANs via a mesh) can discover each other and share compute. It
// never carries inference traffic; nodes route to each other peer-to-peer over the mesh.
//
//	llama-hub --listen :8088 --tokens ./data/tokens.json
//
// Seed the admin key (the key you log into the UI with, and use to mint other tokens) via the
// LLAMA_HUB_ADMIN_KEY environment variable on first boot.
package main

import (
	"context"
	"flag"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/cpuchip/llama-chip/internal/hub"
)

func main() {
	listen := flag.String("listen", envOr("LLAMA_HUB_LISTEN", ":8088"), "listen address")
	tokens := flag.String("tokens", envOr("LLAMA_HUB_TOKENS", "tokens.json"), "token store file (persisted)")
	ttl := flag.Duration("node-ttl", 30*time.Second, "how long a node stays in the roster after its last heartbeat")
	flag.Parse()

	logger := log.New(os.Stderr, "llama-hub ", log.LstdFlags)
	store, err := hub.NewStore(*tokens, *ttl)
	if err != nil {
		logger.Fatalf("token store: %v", err)
	}
	if created, err := store.EnsureAdmin(os.Getenv("LLAMA_HUB_ADMIN_KEY")); err != nil {
		logger.Fatalf("seed admin: %v", err)
	} else if created {
		logger.Printf("seeded admin token from LLAMA_HUB_ADMIN_KEY")
	} else if os.Getenv("LLAMA_HUB_ADMIN_KEY") == "" {
		logger.Printf("WARNING: no LLAMA_HUB_ADMIN_KEY set and no admin token exists — set one to manage the hub")
	}

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		logger.Fatalf("listen %s: %v (port already in use?)", *listen, err)
	}
	srv := &http.Server{Handler: hub.NewServer(store, logger).Handler()}
	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		<-c
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()
	logger.Printf("hub on %s — UI at /, roster at /api/roster, tokens at /api/tokens (admin), store %s", *listen, *tokens)
	if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
		logger.Fatalf("serve: %v", err)
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
