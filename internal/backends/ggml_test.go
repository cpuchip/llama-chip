package backends

import (
	"strings"
	"testing"
)

// b10012Assets mirrors the real ggml-org/llama.cpp release b10012 asset set (verified 2026-07 via
// `gh api repos/ggml-org/llama.cpp/releases/latest`). The matcher must pick the right server (and,
// on Windows, cudart) out of this crowded list without grabbing a vulkan/rocm/sycl/openvino build.
func b10012Assets() ghRelease {
	names := []string{
		"cudart-llama-bin-win-cuda-12.4-x64.zip",
		"cudart-llama-bin-win-cuda-13.3-x64.zip",
		"llama-b10012-bin-android-arm64.tar.gz",
		"llama-b10012-bin-macos-arm64.tar.gz",
		"llama-b10012-bin-macos-x64.tar.gz",
		"llama-b10012-bin-ubuntu-arm64.tar.gz",
		"llama-b10012-bin-ubuntu-openvino-2026.2.1-x64.tar.gz",
		"llama-b10012-bin-ubuntu-rocm-7.2-x64.tar.gz",
		"llama-b10012-bin-ubuntu-s390x.tar.gz",
		"llama-b10012-bin-ubuntu-sycl-fp16-x64.tar.gz",
		"llama-b10012-bin-ubuntu-sycl-fp32-x64.tar.gz",
		"llama-b10012-bin-ubuntu-vulkan-arm64.tar.gz",
		"llama-b10012-bin-ubuntu-vulkan-x64.tar.gz",
		"llama-b10012-bin-ubuntu-x64.tar.gz",
		"llama-b10012-bin-win-cpu-arm64.zip",
		"llama-b10012-bin-win-cpu-x64.zip",
		"llama-b10012-bin-win-cuda-12.4-x64.zip",
		"llama-b10012-bin-win-cuda-13.3-x64.zip",
		"llama-b10012-bin-win-hip-radeon-x64.zip",
		"llama-b10012-bin-win-vulkan-x64.zip",
	}
	r := ghRelease{TagName: "b10012"}
	for _, n := range names {
		r.Assets = append(r.Assets, struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
		}{Name: n, URL: "https://example/" + n})
	}
	return r
}

func TestMatchGGMLAssetsLinuxPicksCPUTarball(t *testing.T) {
	urls, err := matchGGMLAssets("linux", b10012Assets())
	if err != nil {
		t.Fatal(err)
	}
	if len(urls) != 1 {
		t.Fatalf("linux should need exactly the one CPU tarball (no cudart), got %d: %v", len(urls), urls)
	}
	if !strings.HasSuffix(urls[0], "llama-b10012-bin-ubuntu-x64.tar.gz") {
		t.Fatalf("linux picked %q, want the plain ubuntu-x64 CPU build (not vulkan/rocm/sycl/openvino)", urls[0])
	}
}

func TestMatchGGMLAssetsWindowsPicksServerAndCudart(t *testing.T) {
	urls, err := matchGGMLAssets("windows", b10012Assets())
	if err != nil {
		t.Fatal(err)
	}
	if len(urls) != 2 {
		t.Fatalf("windows needs server + cudart, got %d: %v", len(urls), urls)
	}
	var haveBin, haveCudart bool
	for _, u := range urls {
		if strings.HasSuffix(u, "llama-b10012-bin-win-cuda-12.4-x64.zip") {
			haveBin = true
		}
		if strings.HasSuffix(u, "cudart-llama-bin-win-cuda-12.4-x64.zip") {
			haveCudart = true
		}
	}
	if !haveBin || !haveCudart {
		t.Fatalf("windows must pick the cuda-12.4 server AND its cudart, got %v", urls)
	}
}

func TestMatchGGMLAssetsUnsupportedOS(t *testing.T) {
	if _, err := matchGGMLAssets("darwin", b10012Assets()); err == nil {
		t.Fatal("darwin should be unsupported (explicit backend dir), expected an error")
	}
}

func TestMatchGGMLAssetsMissingLinuxBuild(t *testing.T) {
	// a release that only shipped Windows assets → linux match must error, not pick a wrong build
	rel := ghRelease{TagName: "bXXXX"}
	rel.Assets = append(rel.Assets, struct {
		Name string `json:"name"`
		URL  string `json:"browser_download_url"`
	}{Name: "llama-bXXXX-bin-win-cuda-12.4-x64.zip", URL: "https://example/win.zip"})
	if _, err := matchGGMLAssets("linux", rel); err == nil {
		t.Fatal("expected an error when no ubuntu-x64 CPU build is present")
	}
}
