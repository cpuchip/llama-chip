// Package models discovers the GGUF model files LM Studio has already downloaded,
// so llama-chip reuses that library directly (no copies, no re-downloads).
package models

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Model is one discovered GGUF.
type Model struct {
	ID    string `json:"id"`    // "publisher/repo/file.gguf" — stable handle for the router
	Name  string `json:"name"`  // the file's base name without .gguf
	Path  string `json:"path"`  // absolute path
	Bytes int64  `json:"bytes"`
	// Multimodal: a co-located mmproj-*.gguf projector makes this a vision/audio model.
	MMProj         string `json:"mmproj,omitempty"`          // the projector path, if one sits beside the GGUF
	SupportsVision bool   `json:"supports_vision,omitempty"` // true when a projector is present
}

// ModelsDir is the LM Studio models directory. Override with LLAMACHIP_MODELS_DIR.
func ModelsDir() string {
	if d := os.Getenv("LLAMACHIP_MODELS_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".lmstudio", "models")
}

// Discover walks the models directory for loadable GGUFs, largest first.
// mmproj-* files (vision projectors) and obvious shards past the first are skipped.
func Discover() ([]Model, error) {
	root := ModelsDir()
	var out []Model
	mmprojByDir := map[string]string{} // model dir -> its co-located projector
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := d.Name()
		low := strings.ToLower(name)
		if !strings.HasSuffix(low, ".gguf") {
			return nil
		}
		if strings.HasPrefix(low, "mmproj") {
			mmprojByDir[filepath.Dir(path)] = path // a vision/audio projector; pairs with its sibling model
			return nil
		}
		if isSplitShardPastFirst(name) {
			return nil // a multi-file model; we load via the first shard
		}
		fi, e := d.Info()
		if e != nil {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		out = append(out, Model{
			ID:    filepath.ToSlash(rel),
			Name:  strings.TrimSuffix(name, filepath.Ext(name)),
			Path:  path,
			Bytes: fi.Size(),
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	// Pair each model with a co-located projector (the walk may visit it in any order,
	// so do it after the map is complete) → vision/multimodal capable.
	for i := range out {
		if mp, ok := mmprojByDir[filepath.Dir(out[i].Path)]; ok {
			out[i].MMProj = mp
			out[i].SupportsVision = true
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Bytes > out[j].Bytes })
	return out, nil
}

// FindMMProj returns the path to a co-located multimodal projector (mmproj-*.gguf)
// in the model's own directory, if one exists. Used at launch to add --mmproj.
func FindMMProj(modelPath string) (string, bool) {
	entries, err := os.ReadDir(filepath.Dir(modelPath))
	if err != nil {
		return "", false
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		n := strings.ToLower(e.Name())
		if strings.HasPrefix(n, "mmproj") && strings.HasSuffix(n, ".gguf") {
			return filepath.Join(filepath.Dir(modelPath), e.Name()), true
		}
	}
	return "", false
}

// Find resolves a model handle (full ID, or a name substring) to a single model.
func Find(query string) (Model, bool) {
	all, err := Discover()
	if err != nil {
		return Model{}, false
	}
	q := strings.ToLower(query)
	for _, m := range all { // exact ID
		if strings.EqualFold(m.ID, query) {
			return m, true
		}
	}
	var matches []Model
	for _, m := range all {
		if strings.Contains(strings.ToLower(m.ID), q) {
			matches = append(matches, m)
		}
	}
	if len(matches) == 1 {
		return matches[0], true
	}
	return Model{}, false
}

// isSplitShardPastFirst reports GGUF shards like "...-00002-of-00005.gguf" (keep 00001).
func isSplitShardPastFirst(name string) bool {
	i := strings.Index(name, "-of-")
	if i < 0 || i < 6 {
		return false
	}
	shard := name[i-5 : i] // the "NNNNN" before -of-
	return len(shard) == 5 && shard != "00001" && allDigits(shard)
}

func allDigits(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
