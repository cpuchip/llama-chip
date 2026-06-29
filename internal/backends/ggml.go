// ggml.go — llama-chip's OWN managed llama.cpp backends, pulled straight from
// ggml-org/llama.cpp GitHub releases (MIT, published ~daily) instead of waiting
// for LM Studio to ship a build. This is how a slot can run a bleeding-edge arch
// (cohere2moe, etc.) that LM Studio's bundled runtime doesn't have yet.
//
// Builds land under ManagedDir()/ggml-<tag>/ as a SELF-CONTAINED directory
// (llama-server.exe + ggml DLLs + the cudart DLLs all co-located), which the rig
// launches via "backend": "ggml@latest" | "ggml@bNNNN" (or an explicit dir).
package backends

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// ManagedDir is where llama-chip keeps the ggml-org builds it pulls itself
// (separate from LM Studio's). Override with LLAMACHIP_GGML_DIR.
func ManagedDir() string {
	if d := os.Getenv("LLAMACHIP_GGML_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".llama-chip", "backends")
}

// ggmlCudaTag returns the CUDA-family tag used in this host's release-asset names
// (the server binary is "…bin-win-<tag>-x64.zip", the cudart is
// "cudart-…-<tag>-x64.zip"). Windows + CUDA 12 today (the 4090 box); other hosts
// download a release by hand + use an explicit backend dir.
func ggmlCudaTag() (string, error) {
	if runtime.GOOS != "windows" {
		return "", fmt.Errorf("pull-ggml currently supports Windows + CUDA 12 only (host is %s); download a llama.cpp release by hand and point a slot at it with an explicit backend dir", runtime.GOOS)
	}
	return "cuda-12.4", nil
}

type ghRelease struct {
	TagName string `json:"tag_name"`
	Assets  []struct {
		Name string `json:"name"`
		URL  string `json:"browser_download_url"`
	} `json:"assets"`
}

func fetchRelease(ref string) (ghRelease, error) {
	url := "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
	if ref != "" && ref != "latest" {
		url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/" + ref
	}
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	req.Header.Set("User-Agent", "llama-chip")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ghRelease{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ghRelease{}, fmt.Errorf("github release %q: HTTP %d (rate-limited? releases are public, try again shortly)", ref, resp.StatusCode)
	}
	var r ghRelease
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return ghRelease{}, err
	}
	return r, nil
}

func managedBackend(dir, tag string) Backend {
	return Backend{Variant: Variant("ggml"), Version: tag, Dir: dir, Server: filepath.Join(dir, ServerBinName()), Complete: true}
}

// PullGGML downloads the ggml-org llama.cpp release `ref` ("latest" or a build
// tag like "b9837") for this host — the server binary AND the cudart runtime —
// and extracts BOTH into ManagedDir()/ggml-<tag>/. Idempotent: a build already
// present is reused without re-downloading.
func PullGGML(ref string) (Backend, error) {
	cudaTag, err := ggmlCudaTag()
	if err != nil {
		return Backend{}, err
	}
	rel, err := fetchRelease(ref)
	if err != nil {
		return Backend{}, err
	}
	dir := filepath.Join(ManagedDir(), "ggml-"+rel.TagName)
	server := filepath.Join(dir, ServerBinName())
	if fi, err := os.Stat(server); err == nil && fi.Size() > 0 {
		fmt.Printf("ggml-%s already present — reusing %s\n", rel.TagName, dir)
		return managedBackend(dir, rel.TagName), nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Backend{}, err
	}
	var binURL, cudartURL string
	for _, a := range rel.Assets {
		n := a.Name
		if !strings.HasSuffix(n, ".zip") {
			continue
		}
		// cudart's own name also contains "bin-win-<tag>", so match cudart FIRST.
		switch {
		case strings.Contains(n, "cudart") && strings.Contains(n, cudaTag):
			cudartURL = a.URL
		case strings.Contains(n, "bin-win-"+cudaTag):
			binURL = a.URL
		}
	}
	if binURL == "" || cudartURL == "" {
		return Backend{}, fmt.Errorf("release %s has no win-%s server+cudart .zip assets (asset names change sometimes — try another build or an explicit dir)", rel.TagName, cudaTag)
	}
	for _, u := range []string{binURL, cudartURL} {
		fmt.Printf("  downloading %s ...\n", filepath.Base(u))
		if err := downloadAndExtractZip(u, dir); err != nil {
			return Backend{}, fmt.Errorf("%s: %w", filepath.Base(u), err)
		}
	}
	if fi, err := os.Stat(server); err != nil || fi.Size() == 0 {
		return Backend{}, fmt.Errorf("extracted %s but found no %s in %s", rel.TagName, ServerBinName(), dir)
	}
	fmt.Printf("ggml-%s ready: %s\n", rel.TagName, dir)
	return managedBackend(dir, rel.TagName), nil
}

// downloadAndExtractZip streams a zip to a temp file and flattens its entries
// (by basename) into dest. ggml-org's Windows zips hold the binary + DLLs at the
// root, so flattening co-locates everything the launcher needs on one PATH dir.
func downloadAndExtractZip(url, dest string) error {
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	req.Header.Set("User-Agent", "llama-chip")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	tmp, err := os.CreateTemp("", "llama-chip-*.zip") // zip.OpenReader needs a ReaderAt
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return err
	}
	tmp.Close()
	zr, err := zip.OpenReader(tmp.Name())
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, f := range zr.File {
		if f.FileInfo().IsDir() {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(filepath.Join(dest, filepath.Base(f.Name)))
		if err != nil {
			rc.Close()
			return err
		}
		_, cerr := io.Copy(out, rc)
		rc.Close()
		out.Close()
		if cerr != nil {
			return cerr
		}
	}
	return nil
}

// DiscoverManaged returns the ggml-org builds llama-chip has pulled, newest first.
func DiscoverManaged() []Backend {
	ents, err := os.ReadDir(ManagedDir())
	if err != nil {
		return nil
	}
	var out []Backend
	for _, e := range ents {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "ggml-") {
			continue
		}
		tag := strings.TrimPrefix(e.Name(), "ggml-")
		dir := filepath.Join(ManagedDir(), e.Name())
		server := filepath.Join(dir, ServerBinName())
		complete := false
		if fi, err := os.Stat(server); err == nil && fi.Size() > 0 {
			complete = true
		}
		out = append(out, Backend{Variant: "ggml", Version: tag, Dir: dir, Server: server, Complete: complete})
	}
	sort.Slice(out, func(i, j int) bool { return ggmlBuildNum(out[i].Version) > ggmlBuildNum(out[j].Version) })
	return out
}

// ResolveGGML resolves "ggml@latest" / "ggml@bNNNN" to a build llama-chip has
// already pulled. It does NOT download (a `serve` shouldn't silently fetch
// ~600 MB) — if the build isn't present it errors with the pull command.
func ResolveGGML(ref string) (Backend, error) {
	managed := DiscoverManaged() // newest-first
	if ref == "" || ref == "latest" {
		for _, b := range managed {
			if b.Complete {
				return b, nil
			}
		}
		return Backend{}, fmt.Errorf("no pulled ggml build in %s — run `llama-chip pull-ggml latest`", ManagedDir())
	}
	for _, b := range managed {
		if b.Version == ref && b.Complete {
			return b, nil
		}
	}
	return Backend{}, fmt.Errorf("ggml build %q not pulled — run `llama-chip pull-ggml %s`", ref, ref)
}

func ggmlBuildNum(tag string) int { // "b9837" -> 9837
	n, _ := strconv.Atoi(strings.TrimPrefix(strings.TrimPrefix(tag, "b"), "B"))
	return n
}
