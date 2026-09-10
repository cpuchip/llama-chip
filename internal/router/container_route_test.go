package router

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cpuchip/llama-chip/internal/config"
	"github.com/cpuchip/llama-chip/internal/fed"
	"github.com/cpuchip/llama-chip/internal/rig"
)

// A container slot is addressed by its alias while the server inside answers only to its own
// served name: the router must rewrite `model` on the way in, or the upstream refuses it.
func TestProxyRewritesModelForContainerSlot(t *testing.T) {
	up := newFakeBackend("qwen3.8-27b") // the upstream serves this name and nothing else
	up.wantTok = "secret-key"
	defer up.close()
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(200) })
	mux.Handle("/", up.srv.Config.Handler)
	root := httptest.NewServer(mux)
	defer root.Close()

	r := rig.NewEmptyForTest(log.New(io.Discard, "", 0))
	slot := config.Slot{Alias: "qwen-lab", GPUs: []int{1}, APIKey: "secret-key",
		Container: &config.Container{Image: "example/vllm:tag", Served: "qwen3.8-27b"}}
	if err := r.AddContainerForTest(slot, root.URL); err != nil {
		t.Fatal(err)
	}
	f := fed.New(fed.Config{NodeName: "here"}, log.New(io.Discard, "", 0))
	front := httptest.NewServer(New(r, f, log.New(io.Discard, "", 0)).Handler())
	defer front.Close()

	resp := chatReq(t, front.URL, "qwen-lab")
	defer resp.Body.Close()
	var got map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %v (the upstream only serves the served name: was `model` rewritten?)", resp.StatusCode, got)
	}
	if up.gotAuth != "Bearer secret-key" {
		t.Fatalf("upstream saw auth %q", up.gotAuth)
	}
	// The status API labels the slot as a container.
	st, err := http.Get(front.URL + "/api/status")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Body.Close()
	var status struct {
		Slots []struct {
			Name string `json:"name"`
			Kind string `json:"kind"`
		} `json:"slots"`
	}
	if err := json.NewDecoder(st.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	if len(status.Slots) != 1 || status.Slots[0].Kind != "container" || status.Slots[0].Name != "qwen-lab" {
		t.Fatalf("status slots %+v", status.Slots)
	}
}

func TestReplaceModelKeepsOtherFields(t *testing.T) {
	body := []byte(`{"model":"qwen-lab","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":false}`)
	out, ok := replaceModel(body, "qwen3.8-27b")
	if !ok {
		t.Fatal("replace failed")
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if m["model"] != "qwen3.8-27b" || m["max_tokens"] != float64(8) || m["stream"] != false {
		t.Fatalf("fields changed: %v", m)
	}
	if _, ok := replaceModel([]byte("not json"), "x"); ok {
		t.Fatal("non-JSON body should not be rewritten")
	}
}
