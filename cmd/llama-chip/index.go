package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/cpuchip/llama-chip/internal/models"
	"github.com/cpuchip/llama-chip/internal/share"
)

// cmdIndex implements: llama-chip index [--all] [model...]
//
// Hashing is the price of sharing by content address, and it is minutes per model, so a node
// never pays it on its own — not at boot, not on a list request. The operator says which models
// this box is willing to serve, and pays once. After this, /api/models carries their sha256 and
// a peer's `llama-chip fetch` can name them.
//
// A running server picks up what this writes without a restart (see Index.Lookup).
func cmdIndex(args []string) error {
	fs := flag.NewFlagSet("index", flag.ContinueOnError)
	all := fs.Bool("all", false, "index every discovered model (hours for a large library)")
	force := fs.Bool("force", false, "re-hash even models already in the cache")
	cache := fs.String("cache", "", "index file (default: the node's cache path)")
	flagArgs, positional := splitArgs(args)
	if err := fs.Parse(flagArgs); err != nil {
		return err
	}
	if !*all && len(positional) == 0 {
		return fmt.Errorf("usage: llama-chip index <model>...   (or --all)")
	}

	found, err := models.Discover()
	if err != nil {
		return fmt.Errorf("cannot read the model library: %w", err)
	}
	if len(found) == 0 {
		return fmt.Errorf("no models found in %s", models.ModelsDir())
	}

	var todo []models.Model
	if *all {
		todo = found
	} else {
		for _, q := range positional {
			ms, err := matchLocal(found, q)
			if err != nil {
				return err
			}
			todo = append(todo, ms...)
		}
	}

	path := *cache
	if path == "" {
		path = share.DefaultIndexPath()
	}
	ix := share.OpenIndex(path)

	var hashed, skipped int
	var bytes int64
	start := time.Now()
	for _, m := range todo {
		if !*force {
			if sum, ok := ix.Lookup(m.Path); ok {
				fmt.Printf("cached   %s\n         %s\n", m.ID, sum)
				skipped++
				continue
			}
		}
		fmt.Printf("hashing  %s  (%.2f GiB)...\n", m.ID, float64(m.Bytes)/(1<<30))
		t0 := time.Now()
		sum, err := ix.Hash(m.Path)
		if err != nil {
			// One unreadable model must not abandon the rest of the run.
			fmt.Fprintf(os.Stderr, "  FAILED %s: %v\n", m.ID, err)
			continue
		}
		d := time.Since(t0)
		fmt.Printf("indexed  %s\n         %s  (%s, %.0f MiB/s)\n",
			m.ID, sum, d.Round(time.Second), float64(m.Bytes)/(1<<20)/d.Seconds())
		hashed++
		bytes += m.Bytes
	}

	fmt.Printf("\n%d indexed, %d already cached, %.2f GiB read in %s\n",
		hashed, skipped, float64(bytes)/(1<<30), time.Since(start).Round(time.Second))
	fmt.Printf("cache: %s\n", path)
	if hashed > 0 {
		fmt.Println("a running node picks these up without a restart; peers can now fetch them by name")
	}
	return nil
}

// matchLocal resolves one operator-typed name against the local library, by exact id or
// substring — the same rule `fetch` applies on the peer's catalogue, so a name that works for
// one works for the other.
func matchLocal(found []models.Model, query string) ([]models.Model, error) {
	q := strings.ToLower(query)
	var hits []models.Model
	for _, m := range found {
		if strings.EqualFold(m.ID, query) {
			return []models.Model{m}, nil
		}
		if strings.Contains(strings.ToLower(m.ID), q) || strings.Contains(strings.ToLower(m.Name), q) {
			hits = append(hits, m)
		}
	}
	if len(hits) == 0 {
		return nil, fmt.Errorf("no local model matches %q", query)
	}
	return hits, nil
}
