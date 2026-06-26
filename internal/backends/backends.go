// Package backends discovers the llama.cpp builds that LM Studio has already
// downloaded (it does a good job of fetching the latest), so llama-chip can drive
// a maintained llama-server.exe without building llama.cpp itself.
//
// LM Studio stores them under ~/.lmstudio/extensions/backends/ as directories named
// like:
//
//	llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.22.0
//	llama.cpp-win-x86_64-nvidia-cuda-avx2-2.20.1
//	llama.cpp-win-x86_64-amd-rocm-avx2-2.16.0
//	llama.cpp-win-x86_64-vulkan-avx2-2.8.0
//	llama.cpp-win-x86_64-avx2-2.13.0           (cpu)
package backends

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// ServerBinName is the llama-server executable name for the current OS.
func ServerBinName() string {
	if runtime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

// coreLibName is the ggml core shared library — the "this build is actually
// downloaded, not a stub" signal. Its name is OS-specific.
func coreLibName() string {
	switch runtime.GOOS {
	case "windows":
		return "ggml-base.dll"
	case "darwin":
		return "libggml-base.dylib"
	default:
		return "libggml-base.so"
	}
}

// osToken is how LM Studio labels the platform in a backend dir name
// (llama.cpp-<osToken>-x86_64-...): "win" / "linux" / "mac".
func osToken() string {
	switch runtime.GOOS {
	case "windows":
		return "win"
	case "darwin":
		return "mac"
	default:
		return "linux"
	}
}

// Variant is the compute backend a build targets.
type Variant string

const (
	CUDA12 Variant = "cuda12" // NVIDIA, CUDA 12 — what the 4090s want
	CUDA   Variant = "cuda"   // NVIDIA, older CUDA
	ROCm   Variant = "rocm"   // AMD
	Vulkan Variant = "vulkan" // cross-vendor
	CPU    Variant = "cpu"    // avx2, no GPU
)

// Backend is one discovered llama.cpp build.
type Backend struct {
	Variant  Variant `json:"variant"`
	Version  string  `json:"version"` // e.g. "2.22.0"
	Dir      string  `json:"dir"`
	Server   string  `json:"server"`   // full path to llama-server.exe
	Complete bool    `json:"complete"` // the server binary is actually present (not a 0-byte stub)
}

// BackendsDir is the LM Studio backends directory. Override with LLAMACHIP_BACKENDS_DIR.
func BackendsDir() string {
	if d := os.Getenv("LLAMACHIP_BACKENDS_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".lmstudio", "extensions", "backends")
}

// Discover scans the backends directory and returns every build found, newest first.
func Discover() ([]Backend, error) {
	root := BackendsDir()
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	var out []Backend
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		v, ver, ok := parseDirName(e.Name())
		if !ok {
			continue
		}
		server := filepath.Join(root, e.Name(), ServerBinName())
		// "runnable" = the server binary exists (non-empty) AND the ggml core lib is present. In
		// newer LM Studio runtimes (2.18+) the server is a ~20KB shim that loads an impl lib, so
		// binary SIZE is not the signal — the presence of the ggml core lib (libggml-base.so on
		// Linux, ggml-base.dll on Windows, .dylib on macOS) is. Un-downloaded versions are bare
		// dirs / 0-byte stubs.
		complete := false
		if fi, err := os.Stat(server); err == nil && fi.Size() > 0 {
			if _, err := os.Stat(filepath.Join(root, e.Name(), coreLibName())); err == nil {
				complete = true
			}
		}
		out = append(out, Backend{Variant: v, Version: ver, Dir: filepath.Join(root, e.Name()), Server: server, Complete: complete})
	}
	sort.Slice(out, func(i, j int) bool { return cmpVersion(out[i].Version, out[j].Version) > 0 })
	return out, nil
}

// Latest returns the newest COMPLETE backend for a variant (the one we can actually run).
func Latest(variant Variant) (Backend, bool) {
	all, err := Discover()
	if err != nil {
		return Backend{}, false
	}
	for _, b := range all { // Discover is already newest-first
		if b.Variant == variant && b.Complete {
			return b, true
		}
	}
	return Backend{}, false
}

// DLLDirs returns the directories that must be on PATH to launch this backend's
// llama-server.exe: the backend dir (ggml-cuda.dll, etc.) PLUS the shared runtime
// vendor dir (cudart64_12.dll / cublas64_12.dll), which LM Studio keeps separate under
// backends/vendor/. Without the vendor dir, llama-server dies instantly with 0xC0000135
// (DLL not found).
func (b Backend) DLLDirs() []string {
	dirs := []string{b.Dir}
	if vd := b.vendorDir(); vd != "" {
		dirs = append(dirs, vd)
	}
	return dirs
}

func (b Backend) vendorDir() string {
	want := map[Variant]string{CUDA12: "cuda12", CUDA: "cuda", ROCm: "rocm", Vulkan: "vulkan"}[b.Variant]
	if want == "" {
		return ""
	}
	vendor := filepath.Join(filepath.Dir(b.Dir), "vendor")
	ents, err := os.ReadDir(vendor)
	if err != nil {
		return ""
	}
	var best string
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		n := e.Name()
		if b.Variant == CUDA && strings.Contains(n, "cuda12") {
			continue // keep plain "cuda" from matching the cuda12 vendor dir
		}
		if strings.Contains(n, want) {
			best = filepath.Join(vendor, n) // prefer the last (newest -vN) match
		}
	}
	return best
}

// parseDirName turns "llama.cpp-<os>-x86_64-nvidia-cuda12-avx2-2.22.0" into (CUDA12, "2.22.0", true).
// OS-flexible: matches the current platform's token (win/linux/mac) so the same binary works
// across Windows and Linux (LM Studio names the dirs llama.cpp-win-… or llama.cpp-linux-…).
func parseDirName(name string) (Variant, string, bool) {
	prefix := "llama.cpp-" + osToken() + "-x86_64-"
	if !strings.HasPrefix(name, prefix) {
		return "", "", false
	}
	rest := strings.TrimPrefix(name, prefix) // e.g. "nvidia-cuda12-avx2-2.22.0"
	// version = trailing "-X.Y.Z"
	idx := strings.LastIndex(rest, "-")
	if idx < 0 {
		return "", "", false
	}
	ver := rest[idx+1:]
	if !looksLikeVersion(ver) {
		return "", "", false
	}
	tag := rest[:idx] // e.g. "nvidia-cuda12-avx2"
	switch {
	case strings.Contains(tag, "cuda12"):
		return CUDA12, ver, true
	case strings.Contains(tag, "cuda"):
		return CUDA, ver, true
	case strings.Contains(tag, "rocm"):
		return ROCm, ver, true
	case strings.Contains(tag, "vulkan"):
		return Vulkan, ver, true
	default: // just "avx2"
		return CPU, ver, true
	}
}

func looksLikeVersion(s string) bool {
	if s == "" {
		return false
	}
	for _, p := range strings.Split(s, ".") {
		if _, err := strconv.Atoi(p); err != nil {
			return false
		}
	}
	return true
}

// cmpVersion compares dotted versions numerically: >0 if a>b, <0 if a<b, 0 if equal.
func cmpVersion(a, b string) int {
	pa, pb := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < len(pa) || i < len(pb); i++ {
		var x, y int
		if i < len(pa) {
			x, _ = strconv.Atoi(pa[i])
		}
		if i < len(pb) {
			y, _ = strconv.Atoi(pb[i])
		}
		if x != y {
			if x > y {
				return 1
			}
			return -1
		}
	}
	return 0
}
