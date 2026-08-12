package router

// Integration gate for the Anthropic door — the HANDLER, not the fold.
//
// The fold's own unit tests call foldSystemTurns directly, which means both
// routes could be deleted, the path guard could stop invoking the fold, or the
// query string could be dropped, and every one of them would stay green. A
// private replay is strong evidence today; it is not a regression gate. So
// this drives real requests through rt.Handler() into a synthetic upstream and
// asserts what actually reaches it.
//
// The upstream is reached through the federation route rather than a local rig
// slot: an empty &rig.Rig{} resolves nothing (nil map read, empty slice — no
// panic), so resolution falls through to a peer whose address is this test's
// httptest server. Same proxyByModel path, no llama-server required.

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/cpuchip/llama-chip/internal/fed"
	"github.com/cpuchip/llama-chip/internal/rig"
)

type seenReq struct {
	path  string
	query string
	body  []byte
	ctype string
}

// doorHarness wires: client -> rt.Handler() -> (federation route) -> upstream.
func doorHarness(t *testing.T, model string) (http.Handler, *seenReq, func()) {
	t.Helper()
	seen := &seenReq{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seen.path, seen.query, seen.body, seen.ctype = r.URL.Path, r.URL.RawQuery, b, r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}`))
	}))

	logger := log.New(os.Stderr, "", 0)
	f := fed.New(fed.Config{
		NodeName: "self",
		Peers:    []fed.PeerConfig{{Name: "peer", URL: upstream.URL}},
	}, logger)
	if f == nil {
		t.Fatal("fed.New returned nil — the harness needs an enabled federation")
	}
	f.ApplyRoster([]fed.RosterEntry{{Name: "peer", MeshAddr: upstream.URL, Models: []string{model}}}, "self")

	rt := New(&rig.Rig{}, f, logger)
	return rt.Handler(), seen, upstream.Close
}

func post(t *testing.T, h http.Handler, target string, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, target, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// THE ROUTE EXISTS AT ALL. Before this change the router answered 404 here,
// which is the entire defect; nothing in the fold's unit tests would notice if
// the registration were removed again.
func TestDoor_MessagesRouteIsServedAndProxied(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	body := []byte(`{"model":"test-model","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	rec := post(t, h, "/v1/messages?beta=true", body)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d (404 means the route is not registered): %s", rec.Code, rec.Body.String())
	}
	if seen.path != "/v1/messages" {
		t.Fatalf("upstream saw path %q, want /v1/messages", seen.path)
	}
	// The CLI really sends ?beta=true; dropping the query silently would change
	// what the upstream is asked for.
	if seen.query != "beta=true" {
		t.Fatalf("upstream saw query %q, want beta=true", seen.query)
	}
	if seen.ctype != "application/json" {
		t.Fatalf("upstream saw content-type %q", seen.ctype)
	}
}

// NO SYSTEM TURN -> the upstream must receive the caller's ORIGINAL BYTES.
// This is the promise the handler could break without any fold test noticing.
func TestDoor_NoSystem_ForwardsBytesUnchanged(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	body := []byte(`{"model":"test-model","max_tokens":8,"system":"top","messages":[{"role":"user","content":"hi"}],"temperature":0.2}`)
	if rec := post(t, h, "/v1/messages", body); rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	if !bytes.Equal(seen.body, body) {
		t.Fatalf("upstream body was rewritten with nothing to fold:\n sent: %s\n got:  %s", body, seen.body)
	}
}

// A MID-CONVERSATION SYSTEM TURN must arrive folded — i.e. the handler is
// actually invoking the fold on this path, not merely proxying.
func TestDoor_MidSystem_ArrivesFolded(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"u1"},{"role":"system","content":"SYS"}]}`)
	if rec := post(t, h, "/v1/messages", body); rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	if bytes.Contains(seen.body, []byte(`"system"`)) {
		t.Fatalf("a system role reached the upstream unfolded: %s", seen.body)
	}
	var doc struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(seen.body, &doc); err != nil {
		t.Fatalf("upstream body is not valid JSON: %v", err)
	}
	if len(doc.Messages) != 1 || doc.Messages[0].Role != "user" {
		t.Fatalf("want a single folded user turn, got %s", seen.body)
	}
	if len(doc.Messages[0].Content) != 2 ||
		doc.Messages[0].Content[0].Text != "u1" || doc.Messages[0].Content[1].Text != "SYS" {
		t.Fatalf("folded content wrong or out of order: %s", seen.body)
	}
}

// count_tokens is a separate registration — the CLI calls it, and it must
// route too. Its payloads carry no system turn, so it also takes the
// untouched-bytes path.
func TestDoor_CountTokensRouteIsServed(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
	rec := post(t, h, "/v1/messages/count_tokens?beta=true", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d (route missing?): %s", rec.Code, rec.Body.String())
	}
	if seen.path != "/v1/messages/count_tokens" {
		t.Fatalf("upstream saw %q", seen.path)
	}
	if !bytes.Equal(seen.body, body) {
		t.Fatalf("count_tokens body was rewritten: %s", seen.body)
	}
}

// MODEL ROUTING still decides the target: an unknown model must not reach any
// upstream, or "it proxied" would prove nothing about routing.
func TestDoor_UnknownModelIsRefused(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	rec := post(t, h, "/v1/messages", []byte(`{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}`))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status %d, want 404 for an unroutable model", rec.Code)
	}
	if seen.path != "" {
		t.Fatalf("an unroutable request still reached the upstream at %q", seen.path)
	}
}

// TRAILING DATA is refused at ROUTING, and this test records what I got wrong
// rather than smoothing it over.
//
// I first expected a 200 with the bytes forwarded untouched. It is a 400: the
// body cannot be read unambiguously, so no target is chosen and nothing is
// proxied. A body whose `model` cannot be read cannot be routed anywhere.
//
// The refusal now NAMES the ambiguity instead of reporting a missing field —
// "trailing data after the JSON object" rather than "missing model", which was
// true but pointed the caller at the wrong thing.
func TestDoor_TrailingData_RefusedAtRouting(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"u"},{"role":"system","content":"s"}]}TRAIL`)
	rec := post(t, h, "/v1/messages", body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400 for a body that cannot be read unambiguously", rec.Code)
	}
	if seen.path != "" {
		t.Fatalf("an unreadable body still reached the upstream at %q", seen.path)
	}
	if !strings.Contains(rec.Body.String(), "trailing data") {
		t.Fatalf("the refusal should name the actual problem, got %s", rec.Body.String())
	}
}

func TestDoor_DuplicateModel_RefusedAtRouting(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	body := []byte(`{"model":"decoy","messages":[{"role":"user","content":"a"}],"model":"test-model"}`)
	rec := post(t, h, "/v1/messages", body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400 for two `model` keys", rec.Code)
	}
	if seen.path != "" {
		t.Fatalf("an ambiguous request still reached the upstream at %q", seen.path)
	}
	if !strings.Contains(rec.Body.String(), "duplicate") {
		t.Fatalf("the refusal should name the ambiguity, got %s", rec.Body.String())
	}
}

// Same policy, not a model-only special case: any duplicate top-level key is
// ambiguous input. Declaring one rule and enforcing another is how the two
// readings got out of step in the first place.
func TestDoor_DuplicateMessages_RefusedAtRouting(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"a"},{"role":"system","content":"s"}],"messages":[{"role":"user","content":"b"}]}`)
	rec := post(t, h, "/v1/messages", body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400 for two `messages` keys", rec.Code)
	}
	if seen.path != "" {
		t.Fatalf("an ambiguous request still reached the upstream at %q", seen.path)
	}
}

// The policy is not Anthropic-only — proxyByModel serves every proxied
// endpoint, so the same body is refused on the OpenAI path too.
func TestDoor_DuplicateKeysRefusedOnChatCompletionsToo(t *testing.T) {
	h, seen, done := doorHarness(t, "test-model")
	defer done()

	body := []byte(`{"model":"decoy","model":"test-model","messages":[{"role":"user","content":"a"}]}`)
	if rec := post(t, h, "/v1/chat/completions", body); rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d on /v1/chat/completions, want 400 — one policy, every endpoint", rec.Code)
	}
	if seen.path != "" {
		t.Fatalf("an ambiguous request reached the upstream at %q", seen.path)
	}
}

// And the same two refusals at the fold's own level, so the reason is pinned
// to the parser rather than to the handler happening to pass bytes through.
func TestFold_TrailingDataAndDuplicateKeysAreRefused(t *testing.T) {
	trailing := []byte(`{"messages":[{"role":"user","content":"u"},{"role":"system","content":"s"}]}TRAIL`)
	out, n, err := foldSystemTurns(trailing)
	if err == nil {
		t.Fatal("trailing data must be refused, not rewritten")
	}
	if n != 0 || !bytes.Equal(out, trailing) {
		t.Fatal("trailing-data body must come back untouched")
	}

	dup := []byte(`{"messages":[{"role":"user","content":"a"},{"role":"system","content":"s"}],"messages":[{"role":"user","content":"b"}]}`)
	out, n, err = foldSystemTurns(dup)
	if err == nil {
		t.Fatal("duplicate top-level keys must be refused")
	}
	if n != 0 || !bytes.Equal(out, dup) {
		t.Fatal("duplicate-key body must come back untouched")
	}
}
