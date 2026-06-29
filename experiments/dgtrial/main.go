// Command dgtrial drives Unsloth's prebuilt llama-diffusion-gemma-visual-server over its
// stdin/stdout protocol, running pg-ai-stewards work-item-stage fixtures through DiffusionGemma.
//
// DETACHED trial — no live substrate coupling. Fixtures are snapshots of REAL work-item stages
// (the binding question + the substrate model's own output as a reference). The point is to
// measure whether DiffusionGemma is worth wiring into the substrate's dispatch (which would be
// llama-chip's E4 "runner" field) BEFORE building that plumbing. See README.md.
//
// The server is a stdio worker (the Python shim normally fronts it); we drive it raw in Go:
//
//	start:   NGL=99 MAXTOK=0 server <gguf>      -> prints "READY <n_vocab> <MAXTOK>" on stdout
//	request: write a request-file path on stdin; file = {"seed","n_blocks","messages":[...]}
//	stream:  F <block> <step> <total> <json>    (per-step canvas — ignored here)
//	         C <block> <json>                   (cumulative committed answer — last C is final)
//	         STATS <key=value ...>              (timing)
//	         DONE
//	quit:    close stdin (EOF) -> server exits
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type fixture struct {
	ID            string `json:"id"`
	Stage         string `json:"stage"`
	NBlocks       int    `json:"n_blocks"`
	System        string `json:"system"`
	User          string `json:"user"`
	RefModel      string `json:"ref_model"`
	RefOutput     string `json:"ref_output"`      // inline reference output, or…
	RefOutputFile string `json:"ref_output_file"` // …a file (relative to the fixtures dir) to load it from
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type request struct {
	Seed     int       `json:"seed"`
	NBlocks  int       `json:"n_blocks"`
	Messages []message `json:"messages"`
}

type result struct {
	fx      fixture
	answer  string
	stats   map[string]string
	tokS    float64
	decodeS float64
	predN   int
	errMsg  string
}

func main() {
	serverPath := flag.String("server", `C:\Users\cpuch\.unsloth\llama.cpp\build\bin\Release\llama-diffusion-gemma-visual-server.exe`, "visual-server binary")
	modelPath := flag.String("model", `C:\Users\cpuch\.unsloth\models\diffusiongemma-26B-A4B-it-GGUF\diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, "DiffusionGemma GGUF")
	fixDir := flag.String("fixtures", "fixtures", "dir of fixture JSON files")
	outPath := flag.String("out", "results.md", "results markdown")
	ngl := flag.String("ngl", "99", "GPU layers (NGL env)")
	gpu := flag.String("gpu", "0", "CUDA_VISIBLE_DEVICES")
	flag.Parse()

	fixtures, err := loadFixtures(*fixDir)
	if err != nil {
		fatal(err)
	}
	if len(fixtures) == 0 {
		fatal(fmt.Errorf("no fixtures in %s", *fixDir))
	}
	fmt.Printf("loaded %d fixtures from %s\n", len(fixtures), *fixDir)

	// Spawn the server ONCE; it services many requests over stdin (one model load for all fixtures).
	cmd := exec.Command(*serverPath, *modelPath)
	cmd.Env = append(os.Environ(), "NGL="+*ngl, "MAXTOK=0", "CUDA_VISIBLE_DEVICES="+*gpu)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fatal(err)
	}
	errLog, err := os.Create("server.stderr.log")
	if err != nil {
		fatal(err)
	}
	defer errLog.Close()
	cmd.Stderr = errLog
	if err := cmd.Start(); err != nil {
		fatal(err)
	}

	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 0, 1<<20), 32<<20) // C canvas lines can be large

	fmt.Print("loading model (16GB) onto GPU", *gpu, " …")
	t0 := time.Now()
	if !waitReady(sc) {
		fatal(fmt.Errorf("server never printed READY (see server.stderr.log)"))
	}
	loadSecs := time.Since(t0).Seconds()
	fmt.Printf(" READY in %.0fs\n\n", loadSecs)

	tmp, err := os.MkdirTemp("", "dgtrial")
	if err != nil {
		fatal(err)
	}
	defer os.RemoveAll(tmp)

	results := make([]result, 0, len(fixtures))
	for i, fx := range fixtures {
		fmt.Printf("[%d/%d] %-32s (%s) … ", i+1, len(fixtures), fx.ID, fx.Stage)
		r := runFixture(sc, stdin, tmp, i, fx)
		if r.errMsg != "" {
			fmt.Printf("ERR: %s\n", r.errMsg)
		} else {
			fmt.Printf("%d tok @ %.0f tok/s (%.1fs)\n", r.predN, r.tokS, r.decodeS)
		}
		results = append(results, r)
	}

	_ = stdin.Close() // EOF -> server exits
	_ = cmd.Wait()

	if err := writeResults(*outPath, *modelPath, loadSecs, results); err != nil {
		fatal(err)
	}
	fmt.Printf("\nwrote %s\n", *outPath)
}

func loadFixtures(dir string) ([]fixture, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []fixture
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		var fx fixture
		if err := json.Unmarshal(b, &fx); err != nil {
			return nil, fmt.Errorf("%s: %w", e.Name(), err)
		}
		if fx.ID == "" {
			fx.ID = strings.TrimSuffix(e.Name(), ".json")
		}
		if fx.RefOutputFile != "" {
			rb, err := os.ReadFile(filepath.Join(dir, fx.RefOutputFile))
			if err != nil {
				return nil, fmt.Errorf("%s ref_output_file: %w", e.Name(), err)
			}
			fx.RefOutput = string(rb)
		}
		out = append(out, fx)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func waitReady(sc *bufio.Scanner) bool {
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "READY ") {
			return true
		}
	}
	return false
}

func runFixture(sc *bufio.Scanner, stdin io.Writer, tmp string, i int, fx fixture) result {
	nb := fx.NBlocks
	if nb <= 0 {
		nb = 8
	}
	req := request{Seed: 0, NBlocks: nb}
	if fx.System != "" {
		req.Messages = append(req.Messages, message{Role: "system", Content: fx.System})
	}
	req.Messages = append(req.Messages, message{Role: "user", Content: fx.User})
	b, _ := json.Marshal(req)
	rp := filepath.Join(tmp, fmt.Sprintf("req-%02d.json", i))
	if err := os.WriteFile(rp, b, 0o644); err != nil {
		return result{fx: fx, errMsg: err.Error()}
	}
	if _, err := fmt.Fprintln(stdin, rp); err != nil {
		return result{fx: fx, errMsg: "stdin write: " + err.Error()}
	}

	r := result{fx: fx, stats: map[string]string{}}
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "C "):
			r.answer = decodeC(line)
		case strings.HasPrefix(line, "STATS "):
			parseStats(line, r.stats)
		case line == "DONE":
			r.finish()
			return r
		case strings.HasPrefix(line, "ERR"):
			r.errMsg = line
			return r
			// F (per-step canvas) lines are ignored
		}
	}
	r.errMsg = "stream ended before DONE"
	return r
}

// decodeC parses `C <block> <json-string>` -> the decoded answer text.
func decodeC(line string) string {
	parts := strings.SplitN(line, " ", 3)
	if len(parts) < 3 {
		return ""
	}
	var s string
	if err := json.Unmarshal([]byte(parts[2]), &s); err != nil {
		return parts[2]
	}
	return s
}

func parseStats(line string, m map[string]string) {
	for _, f := range strings.Fields(strings.TrimPrefix(line, "STATS ")) {
		if k, v, ok := strings.Cut(f, "="); ok {
			m[k] = v
		}
	}
}

func (r *result) finish() {
	r.predN, _ = strconv.Atoi(r.stats["predicted_n"])
	if ms, err := strconv.ParseFloat(r.stats["decode_ms"], 64); err == nil && ms > 0 {
		r.decodeS = ms / 1000
		r.tokS = float64(r.predN) / r.decodeS
	}
}

func writeResults(path, model string, loadSecs float64, results []result) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# DiffusionGemma × pg-ai-stewards — detached trial\n\n")
	fmt.Fprintf(&b, "Model: `%s` (Q4_K_M, one RTX 4090). Model load: %.0fs. Runner: Unsloth's "+
		"`llama-diffusion-gemma-visual-server`, driven over stdio (no Python).\n\n", filepath.Base(model), loadSecs)

	fmt.Fprintf(&b, "## Summary\n\n| Fixture | Stage | tok | DiffusionGemma tok/s | decode s | ref model |\n")
	fmt.Fprintf(&b, "|---|---|--:|--:|--:|---|\n")
	for _, r := range results {
		if r.errMsg != "" {
			fmt.Fprintf(&b, "| %s | %s | — | ERR | — | %s |\n", r.fx.ID, r.fx.Stage, r.fx.RefModel)
			continue
		}
		fmt.Fprintf(&b, "| %s | %s | %d | **%.0f** | %.1f | %s |\n",
			r.fx.ID, r.fx.Stage, r.predN, r.tokS, r.decodeS, r.fx.RefModel)
	}

	for _, r := range results {
		fmt.Fprintf(&b, "\n---\n\n## %s — `%s`\n\n", r.fx.ID, r.fx.Stage)
		fmt.Fprintf(&b, "**Task (binding question):**\n\n> %s\n\n", strings.ReplaceAll(r.fx.User, "\n", "\n> "))
		if r.errMsg != "" {
			fmt.Fprintf(&b, "**DiffusionGemma:** ERROR — %s\n", r.errMsg)
		} else {
			fmt.Fprintf(&b, "**DiffusionGemma** (%d tok, %.0f tok/s, %.1fs):\n\n```\n%s\n```\n\n",
				r.predN, r.tokS, r.decodeS, strings.TrimSpace(r.answer))
		}
		fmt.Fprintf(&b, "<details><summary>Reference — %s (substrate)</summary>\n\n```\n%s\n```\n</details>\n",
			r.fx.RefModel, strings.TrimSpace(r.fx.RefOutput))
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "dgtrial: "+err.Error())
	os.Exit(1)
}
