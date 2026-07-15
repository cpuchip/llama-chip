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
	"archive/tar"
	"archive/zip"
	"compress/gzip"
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

// winCudaTag is the CUDA-family tag in this host's Windows release-asset names (the server
// binary is "…bin-win-<tag>-x64.zip", the cudart is "cudart-…-<tag>-x64.zip"). CUDA 12.4 is
// what the 4090 box wants.
const winCudaTag = "cuda-12.4"

// ggmlAssetURLs returns the release-asset download URLs llama-chip needs for THIS host:
//
//   - Windows: the CUDA-12 llama-server build ("…bin-win-cuda-12.4-x64.zip") PLUS the matching
//     cudart runtime ("cudart-…-cuda-12.4-x64.zip") — the GPU box needs both.
//   - Linux: the ubuntu x64 CPU build ("…bin-ubuntu-x64.tar.gz"). That generic ubuntu build IS
//     the CPU build (AVX/AVX2 kernels dispatched at runtime); there is no separate cudart to
//     fetch. This is what the GPU-less NOCIX node runs.
//
// Any other OS (macOS, …) is unsupported here — download a release by hand and point a slot at
// an explicit backend dir. Verified against ggml-org/llama.cpp release b10012 (2026-07).
func ggmlAssetURLs(rel ghRelease) ([]string, error) {
	return matchGGMLAssets(runtime.GOOS, rel)
}

// matchGGMLAssets is the OS-parameterized core of ggmlAssetURLs (goos passed in so both branches
// are testable off their native host).
func matchGGMLAssets(goos string, rel ghRelease) ([]string, error) {
	switch goos {
	case "windows":
		var binURL, cudartURL string
		for _, a := range rel.Assets {
			n := a.Name
			if !strings.HasSuffix(n, ".zip") {
				continue
			}
			// cudart's own name also contains "bin-win-<tag>", so match cudart FIRST.
			switch {
			case strings.Contains(n, "cudart") && strings.Contains(n, winCudaTag):
				cudartURL = a.URL
			case strings.Contains(n, "bin-win-"+winCudaTag):
				binURL = a.URL
			}
		}
		if binURL == "" || cudartURL == "" {
			return nil, fmt.Errorf("release %s has no win-%s server+cudart .zip assets (asset names change sometimes — try another build or an explicit dir)", rel.TagName, winCudaTag)
		}
		return []string{binURL, cudartURL}, nil
	case "linux":
		// "bin-ubuntu-x64.tar.gz" is the plain CPU build; the vulkan/rocm/sycl/openvino ubuntu
		// builds carry an extra token (…-vulkan-x64, …-rocm-7.2-x64) so they won't match this.
		for _, a := range rel.Assets {
			if strings.Contains(a.Name, "bin-ubuntu-x64.tar.gz") {
				return []string{a.URL}, nil
			}
		}
		return nil, fmt.Errorf("release %s has no bin-ubuntu-x64.tar.gz CPU build (asset names change sometimes — try another build or an explicit dir)", rel.TagName)
	default:
		return nil, fmt.Errorf("pull-ggml supports Windows (CUDA 12) and Linux (CPU) only (host is %s); download a llama.cpp release by hand and point a slot at it with an explicit backend dir", goos)
	}
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

// PullGGML downloads the ggml-org llama.cpp release `ref` ("latest" or a build tag like "b9837")
// for this host and extracts it into ManagedDir()/ggml-<tag>/ as a self-contained dir. On Windows
// that's the CUDA-12 server binary AND the cudart runtime; on Linux it's the ubuntu x64 CPU build
// (no cudart). Idempotent: a build already present is reused without re-downloading.
func PullGGML(ref string) (Backend, error) {
	rel, err := fetchRelease(ref)
	if err != nil {
		return Backend{}, err
	}
	urls, err := ggmlAssetURLs(rel)
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
	for _, u := range urls {
		fmt.Printf("  downloading %s ...\n", filepath.Base(u))
		if err := downloadAndExtract(u, dir); err != nil {
			return Backend{}, fmt.Errorf("%s: %w", filepath.Base(u), err)
		}
	}
	if fi, err := os.Stat(server); err != nil || fi.Size() == 0 {
		return Backend{}, fmt.Errorf("extracted %s but found no %s in %s", rel.TagName, ServerBinName(), dir)
	}
	// Linux tarballs preserve the +x bit, but make sure the server is executable regardless.
	if runtime.GOOS != "windows" {
		_ = os.Chmod(server, 0o755)
	}
	fmt.Printf("ggml-%s ready: %s\n", rel.TagName, dir)
	return managedBackend(dir, rel.TagName), nil
}

// downloadAndExtract streams a release asset to a temp file and extracts it into dest, flattening
// entries by basename (ggml-org packs the binary + libs in a single dir, so everything the launcher
// needs lands co-located on one dir). Dispatches on the asset's extension: .zip (Windows) or
// .tar.gz/.tgz (Linux).
func downloadAndExtract(url, dest string) error {
	tmp, err := downloadToTemp(url)
	if err != nil {
		return err
	}
	defer os.Remove(tmp)
	low := strings.ToLower(url)
	switch {
	case strings.HasSuffix(low, ".zip"):
		return extractZip(tmp, dest)
	case strings.HasSuffix(low, ".tar.gz"), strings.HasSuffix(low, ".tgz"):
		return extractTarGz(tmp, dest)
	default:
		return fmt.Errorf("don't know how to extract %s (expected .zip or .tar.gz)", filepath.Base(url))
	}
}

// downloadToTemp streams url to a temp file (zip.OpenReader needs a ReaderAt; tar streams anyway).
func downloadToTemp(url string) (string, error) {
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	req.Header.Set("User-Agent", "llama-chip")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	tmp, err := os.CreateTemp("", "llama-chip-*.dl")
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return "", err
	}
	tmp.Close()
	return tmp.Name(), nil
}

// extractZip flattens a local zip's entries (by basename) into dest. ggml-org's Windows zips hold
// the binary + DLLs at the root, so flattening co-locates everything the launcher needs on one dir.
func extractZip(path, dest string) error {
	zr, err := zip.OpenReader(path)
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

// extractTarGz flattens a local .tar.gz's entries (by basename) into dest. ggml-org's Linux
// tarballs pack the binary + .so libs flat under one top-level dir (llama-bNNNN/), so flattening
// co-locates everything the loader needs (llama-server's RUNPATH is $ORIGIN). Symlinks matter:
// the dynamic linker resolves libs by SONAME (e.g. libggml.so.0 -> libggml.so.0.16.0), so those
// symlinks are recreated (flat, since every target is a sibling); regular files keep their mode
// (so the +x on llama-server survives).
func extractTarGz(path, dest string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		name := filepath.Base(hdr.Name)
		if name == "" || name == "." || name == ".." {
			continue
		}
		out := filepath.Join(dest, name)
		switch hdr.Typeflag {
		case tar.TypeDir:
			continue // flattening — the single top-level dir is dropped
		case tar.TypeSymlink, tar.TypeLink:
			// Flatten the link target to a sibling name too (all libs live in one dir here).
			target := filepath.Base(hdr.Linkname)
			_ = os.Remove(out) // idempotent re-extract
			if err := os.Symlink(target, out); err != nil {
				return fmt.Errorf("symlink %s -> %s: %w", name, target, err)
			}
		case tar.TypeReg:
			w, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(hdr.Mode)&0o777)
			if err != nil {
				return err
			}
			_, cerr := io.Copy(w, tr)
			w.Close()
			if cerr != nil {
				return cerr
			}
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
