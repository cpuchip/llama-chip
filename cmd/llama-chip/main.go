// llama-chip — a native local LLM rig manager.
//
// Drives the llama.cpp builds LM Studio already downloaded (rocm / cuda / cuda12 /
// vulkan / cpu) against the GGUFs in LM Studio's model dir, with per-GPU pinning,
// an OpenAI-compatible router, auto-restart, and a loader UI. No Docker, no Python.
//
// Phase 1 (this file): discovery CLI — see what backends + models are available.
//
//	llama-chip backends           # list discovered llama.cpp builds (newest first)
//	llama-chip models             # list GGUFs in the LM Studio library
//	llama-chip gpu                # live GPU state
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/cpuchip/llama-chip/internal/backends"
	"github.com/cpuchip/llama-chip/internal/config"
	"github.com/cpuchip/llama-chip/internal/fed"
	"github.com/cpuchip/llama-chip/internal/gpu"
	"github.com/cpuchip/llama-chip/internal/hubclient"
	"github.com/cpuchip/llama-chip/internal/lms"
	"github.com/cpuchip/llama-chip/internal/models"
	"github.com/cpuchip/llama-chip/internal/rig"
	"github.com/cpuchip/llama-chip/internal/router"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "backends":
		err = cmdBackends()
	case "models":
		err = cmdModels()
	case "gpu":
		err = cmdGPU()
	case "serve":
		err = cmdServe(os.Args[2:])
	case "runtime":
		err = lms.Run(append([]string{"runtime"}, os.Args[2:]...)...) // passthrough to LM Studio's lms
	case "pull":
		err = cmdPull(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `llama-chip — native local LLM rig manager

usage:
  llama-chip backends   list the llama.cpp builds LM Studio has (rocm/cuda/cuda12/vulkan/cpu), newest first
  llama-chip models     list the GGUF models in the LM Studio library
  llama-chip gpu        show live NVIDIA GPU state
  llama-chip serve      start the rig: process manager + OpenAI router (:8090) + loader UI
  llama-chip runtime …  manage inference runtimes via LM Studio's lms (ls|get|select|update|remove)
  llama-chip pull [q]   download the latest runtime via LM Studio (default q: llama.cpp:cuda12)
`)
}

func cmdBackends() error {
	bs, err := backends.Discover()
	if err != nil {
		return err
	}
	if len(bs) == 0 {
		fmt.Printf("no llama.cpp backends found under %s\n", backends.BackendsDir())
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(w, "VARIANT\tVERSION\tRUNNABLE\tSERVER")
	for _, b := range bs {
		runnable := "stub"
		if b.Complete {
			runnable = "yes"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", b.Variant, b.Version, runnable, b.Server)
	}
	w.Flush()
	if b, ok := backends.Latest(backends.CUDA12); ok {
		fmt.Printf("\nlatest runnable CUDA12: %s (%s)\n", b.Version, b.Dir)
	} else {
		fmt.Println("\nno runnable CUDA12 backend — open LM Studio and let it download one, or pick another variant")
	}
	return nil
}

func cmdModels() error {
	ms, err := models.Discover()
	if err != nil {
		return err
	}
	if len(ms) == 0 {
		fmt.Printf("no GGUF models found under %s\n", models.ModelsDir())
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(w, "SIZE\tMODEL")
	for _, m := range ms {
		fmt.Fprintf(w, "%.1f GB\t%s\n", float64(m.Bytes)/(1<<30), m.ID)
	}
	w.Flush()
	fmt.Printf("\n%d models in %s\n", len(ms), models.ModelsDir())
	return nil
}

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	cfgPath := fs.String("config", "config.json", "rig config file (which models load on which GPUs)")
	listen := fs.String("listen", "", "override the router listen address (default from config / :8090)")
	_ = fs.Parse(args)

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return fmt.Errorf("config: %w (see config.example.json)", err)
	}
	if *listen != "" {
		cfg.Listen = *listen
	}
	logger := log.New(os.Stderr, "llama-chip ", log.LstdFlags)
	r, err := rig.New(cfg, logger)
	if err != nil {
		return err
	}
	// Bind the router BEFORE spawning any llama-server, so a port clash can't orphan a
	// loading model process.
	ln, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		return fmt.Errorf("listen %s: %w (port already in use?)", cfg.Listen, err)
	}
	r.Start()

	// Boot into the default profile when no static slots are configured, so a restart comes up
	// serving the right layout (e.g. dance-moe: reasoner + vision-capable gemma) instead of empty.
	if len(cfg.Slots) == 0 && cfg.DefaultProfile != "" {
		if err := r.ApplyProfile(cfg.DefaultProfile); err != nil {
			logger.Printf("default_profile %q: %v (continuing — apply one via /api/profile)", cfg.DefaultProfile, err)
		} else {
			logger.Printf("booted into default_profile %q", cfg.DefaultProfile)
		}
	}

	// Federation (optional): pool GPUs across peer nodes behind this one endpoint.
	fedCtx, fedCancel := context.WithCancel(context.Background())
	f := fed.New(cfg.FedConfig(), logger)
	f.Start(fedCtx)

	// Hub-managed mode (optional): a coordinator supplies the roster + issues join tokens. The
	// hub client heartbeats this node's models + GPU state and applies the roster to the federation.
	if cfg.Federation != nil && cfg.Federation.HubURL != "" {
		fc := cfg.Federation
		interval := time.Duration(fc.PollIntervalSec) * time.Second
		local := func() hubclient.LocalState {
			var ms []string
			for _, s := range r.Snapshot() {
				if s.State == rig.Healthy {
					ms = append(ms, s.Name)
				}
			}
			var gpus []hubclient.GPUStat
			if gs, err := gpu.Query(); err == nil {
				for _, g := range gs {
					gpus = append(gpus, hubclient.GPUStat{Index: g.Index, Name: g.Name, MemUsed: g.MemUsed, MemTotal: g.MemTotal, Util: g.Util})
				}
			}
			return hubclient.LocalState{Models: ms, GPUs: gpus}
		}
		hc := hubclient.New(fc.HubURL, fc.HubToken, f.NodeName(), fc.Advertise, interval, f, local, logger)
		hc.Run(fedCtx)
	}

	srv := &http.Server{Handler: router.New(r, f, logger).Handler()}
	go func() {
		c := make(chan os.Signal, 1)
		// Catch SIGTERM too (e.g. `kill <pid>`, systemd/docker stop), not just SIGINT —
		// otherwise a `kill` skips StopAll() and orphans the llama-server children (they keep
		// holding GPU VRAM + their backend ports until a manual kill).
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		<-c
		logger.Println("shutting down — stopping instances")
		fedCancel()
		r.StopAll()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()
	logger.Printf("router on %s — %d slot(s); OpenAI at %s/v1, status at %s/api/status", cfg.Listen, len(cfg.Slots), cfg.Listen, cfg.Listen)
	if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}

// cmdPull downloads (or lists) a runtime via LM Studio's official mechanism.
func cmdPull(args []string) error {
	if !lms.Available() {
		return fmt.Errorf("lms (LM Studio CLI) not found at ~/.lmstudio/bin — install LM Studio, or just use what `llama-chip backends` already shows")
	}
	query := "llama.cpp:cuda12" // default: latest CUDA12 (the 4090s)
	if len(args) > 0 && args[0] != "" {
		query = args[0]
	}
	fmt.Printf("→ lms runtime get %s -y\n", query)
	if err := lms.Run("runtime", "get", query, "-y"); err != nil {
		return err
	}
	fmt.Println("\ndone — run `llama-chip backends` to see it; the latest runnable is picked automatically.")
	return nil
}

func cmdGPU() error {
	gs, err := gpu.Query()
	if err != nil {
		return fmt.Errorf("nvidia-smi: %w", err)
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(w, "GPU\tNAME\tMEM\tUTIL\tTEMP")
	for _, g := range gs {
		fmt.Fprintf(w, "%d\t%s\t%d/%d MiB\t%d%%\t%d°C\n", g.Index, g.Name, g.MemUsed, g.MemTotal, g.Util, g.TempC)
	}
	w.Flush()
	return nil
}
