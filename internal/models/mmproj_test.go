package models

import (
	"os"
	"path/filepath"
	"testing"
)

// A model dir with a co-located mmproj projector is detected as vision-capable;
// a plain model dir is not.
func TestFindMMProjAndDiscoverPairing(t *testing.T) {
	root := t.TempDir()

	// vision model: a GGUF + its mmproj projector side by side
	visDir := filepath.Join(root, "vendor", "gemma-vision")
	mustMkdir(t, visDir)
	visModel := filepath.Join(visDir, "gemma-4-26B-A4B-it-Q4.gguf")
	mustWrite(t, visModel, "MODELDATA")
	mustWrite(t, filepath.Join(visDir, "mmproj-gemma-4-26B-A4B-it-BF16.gguf"), "PROJDATA")

	// text-only model: just a GGUF, no projector
	txtDir := filepath.Join(root, "vendor", "qwen-text")
	mustMkdir(t, txtDir)
	txtModel := filepath.Join(txtDir, "qwen-text-Q8.gguf")
	mustWrite(t, txtModel, "MODELDATA2")

	// FindMMProj: present beside the vision model, absent beside the text model
	if mp, ok := FindMMProj(visModel); !ok || filepath.Base(mp) != "mmproj-gemma-4-26B-A4B-it-BF16.gguf" {
		t.Fatalf("FindMMProj(vision) = %q, %v; want the projector path", mp, ok)
	}
	if mp, ok := FindMMProj(txtModel); ok {
		t.Fatalf("FindMMProj(text) = %q, true; want no projector", mp)
	}

	// Discover pairs the projector onto the model + flags vision; the mmproj file
	// itself is never offered as a loadable model.
	t.Setenv("LLAMACHIP_MODELS_DIR", root)
	got, err := Discover()
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("Discover found %d models, want 2 (projector must be skipped): %+v", len(got), got)
	}
	for _, m := range got {
		switch filepath.Base(m.Path) {
		case "gemma-4-26B-A4B-it-Q4.gguf":
			if !m.SupportsVision || m.MMProj == "" {
				t.Errorf("vision model not flagged: SupportsVision=%v MMProj=%q", m.SupportsVision, m.MMProj)
			}
		case "qwen-text-Q8.gguf":
			if m.SupportsVision || m.MMProj != "" {
				t.Errorf("text model wrongly flagged vision: SupportsVision=%v MMProj=%q", m.SupportsVision, m.MMProj)
			}
		default:
			t.Errorf("unexpected model %q (mmproj leaked into the list?)", m.Path)
		}
	}
}

func mustMkdir(t *testing.T, d string) {
	t.Helper()
	if err := os.MkdirAll(d, 0o755); err != nil {
		t.Fatal(err)
	}
}

func mustWrite(t *testing.T, p, s string) {
	t.Helper()
	if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
}
