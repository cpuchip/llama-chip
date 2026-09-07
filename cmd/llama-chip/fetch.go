package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cpuchip/llama-chip/internal/models"
	"github.com/cpuchip/llama-chip/internal/share"
)

// cmdFetch implements: llama-chip fetch <model> --from http://<peer>:8090
//
// Peer to peer, never through the hub. The model is named the way a human names it — a
// substring of the id, or the id itself — and resolved against the PEER's catalogue, because
// the peer is the node that has the file. The transfer is by content hash, resumable, and
// verified before the file takes a real name.
func cmdFetch(args []string) error {
	fs := flag.NewFlagSet("fetch", flag.ContinueOnError)
	from := fs.String("from", "", "peer base URL, e.g. http://peer-host:8090 (required)")
	dst := fs.String("dir", "", "destination directory (default: this node's models dir)")
	hash := fs.String("sha256", "", "fetch this exact hash, skipping catalogue lookup")
	timeout := fs.Duration("connect-timeout", 15*time.Second, "peer connect/handshake timeout")
	// Go's flag package stops at the first non-flag argument, so the documented form
	// `fetch <model> --from <peer>` would never see --from. Hoist the positionals out first
	// so flags may appear on either side of the model name.
	flagArgs, positional := splitArgs(args)
	if err := fs.Parse(flagArgs); err != nil {
		return err
	}
	if *from == "" {
		return fmt.Errorf("--from is required (the peer holding the model)")
	}
	dir := *dst
	if dir == "" {
		dir = models.ModelsDir()
	}

	// A long-lived client: the body governs a multi-gigabyte transfer, not an overall deadline.
	// Only the connection handshake is bounded.
	client := &http.Client{Transport: &http.Transport{
		ResponseHeaderTimeout: *timeout,
		IdleConnTimeout:       90 * time.Second,
	}}

	sum := strings.ToLower(strings.TrimSpace(*hash))
	name := ""
	if sum == "" {
		if len(positional) < 1 {
			return fmt.Errorf("usage: llama-chip fetch <model> --from <peer>   (or --sha256 <hex> --from <peer>)")
		}
		query := positional[0]
		m, err := resolveOnPeer(client, *from, query)
		if err != nil {
			return err
		}
		sum, name = m.SHA256, m.ID
		fmt.Printf("resolved %q on %s: %s\n  %s  (%.2f GiB)\n",
			query, *from, m.ID, sum, float64(m.Bytes)/(1<<30))
	}

	start := time.Now()
	last := int64(0)
	path, err := share.Fetch(client, *from, sum, dir, func(done, total int64) {
		rate := float64(done-last) / (1 << 20) // MiB since the last tick (~1s)
		last = done
		if total > 0 {
			fmt.Printf("\r  %.1f%%  %.2f/%.2f GiB  %.0f MiB/s        ",
				100*float64(done)/float64(total), float64(done)/(1<<30), float64(total)/(1<<30), rate)
		} else {
			fmt.Printf("\r  %.2f GiB  %.0f MiB/s        ", float64(done)/(1<<30), rate)
		}
	})
	fmt.Println()
	if err != nil {
		return err
	}
	fmt.Printf("verified %s\n  -> %s  (%s)\n", sum, path, time.Since(start).Round(time.Second))
	if name != "" && filepath.Base(path) != filepath.Base(name) {
		fmt.Printf("  note: saved under the peer's filename; its catalogue id there was %s\n", name)
	}
	return nil
}

// peerModel is the subset of /api/models this command needs.
type peerModel struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

// resolveOnPeer finds one model on the peer by id or substring, and requires it to advertise a
// hash — a peer that has not hashed the file cannot serve it by content address yet.
func resolveOnPeer(client *http.Client, peer, query string) (peerModel, error) {
	base := strings.TrimRight(peer, "/")
	resp, err := client.Get(base + "/api/models")
	if err != nil {
		return peerModel{}, fmt.Errorf("cannot reach peer %s: %w", base, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return peerModel{}, fmt.Errorf("peer %s returned %s for /api/models", base, resp.Status)
	}
	var body struct {
		Models []peerModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return peerModel{}, fmt.Errorf("cannot read the peer's model list: %w", err)
	}

	q := strings.ToLower(query)
	var hits []peerModel
	for _, m := range body.Models {
		if strings.EqualFold(m.ID, query) { // an exact id wins outright
			hits = []peerModel{m}
			break
		}
		if strings.Contains(strings.ToLower(m.ID), q) || strings.Contains(strings.ToLower(m.Name), q) {
			hits = append(hits, m)
		}
	}
	switch len(hits) {
	case 0:
		return peerModel{}, fmt.Errorf("peer %s has no model matching %q", base, query)
	case 1:
	default:
		fmt.Fprintf(os.Stderr, "%q matches %d models on %s:\n", query, len(hits), base)
		for _, m := range hits {
			fmt.Fprintf(os.Stderr, "  %s\n", m.ID)
		}
		return peerModel{}, fmt.Errorf("be more specific, or pass --sha256")
	}
	m := hits[0]
	if m.SHA256 == "" {
		return peerModel{}, fmt.Errorf(
			"peer has %s but has not hashed it yet, so it cannot be fetched by content address.\n"+
				"  Ask that node to index it (its /api/models will then carry sha256), or pass --sha256 if you know it.", m.ID)
	}
	return m, nil
}

// splitArgs separates flag arguments from positional ones so flags may appear anywhere on the
// command line. Anything starting with '-' is treated as a flag; the single-value flags this
// command defines also consume the token after them unless it was given as --flag=value.
func splitArgs(args []string) (flags, positional []string) {
	takesValue := map[string]bool{
		"-from": true, "--from": true,
		"-dir": true, "--dir": true,
		"-sha256": true, "--sha256": true,
		"-connect-timeout": true, "--connect-timeout": true,
	}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" { // everything after is positional
			positional = append(positional, args[i+1:]...)
			break
		}
		if strings.HasPrefix(a, "-") {
			flags = append(flags, a)
			if takesValue[a] && i+1 < len(args) { // --flag value (not --flag=value)
				i++
				flags = append(flags, args[i])
			}
			continue
		}
		positional = append(positional, a)
	}
	return flags, positional
}
