package router

// Unit cases for the system-turn fold.
//
// The shim's own scar sets the bar: the trivial shapes (multi-block system,
// tools) ALL PASSED while only the real captured payload showed the trigger.
// So these are the shapes that actually break a strict template, each with
// its inverse — plus the promise that is easiest to break silently, that a
// request with nothing to fold is forwarded byte-for-byte as it arrived.

import (
	"bytes"
	"encoding/json"
	"testing"
)

func roles(t *testing.T, body []byte) []string {
	t.Helper()
	var doc struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	out := make([]string, 0, len(doc.Messages))
	for _, m := range doc.Messages {
		out = append(out, m.Role)
	}
	return out
}

// texts returns every text block of message i, in order.
func texts(t *testing.T, body []byte, i int) []string {
	t.Helper()
	var doc struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(doc.Messages[i].Content, &blocks); err == nil {
		out := make([]string, 0, len(blocks))
		for _, b := range blocks {
			out = append(out, b.Text)
		}
		return out
	}
	var s string
	if err := json.Unmarshal(doc.Messages[i].Content, &s); err != nil {
		t.Fatalf("content is neither blocks nor string: %s", doc.Messages[i].Content)
	}
	return []string{s}
}

func eq(t *testing.T, got, want []string, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %v, want %v", what, got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: got %v, want %v", what, got, want)
		}
	}
}

// 1. NOTHING TO FOLD -> byte-identical passthrough. This is the promise most
//    likely to rot without anyone noticing, because a re-serialised body still
//    "works" — it just quietly rewrites a payload we said we would not touch.
func TestFold_NoSystem_IsByteIdentical(t *testing.T) {
	in := []byte(`{"model":"m","max_tokens":16,"system":"top-level system stays put","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}],"temperature":0.2}`)
	out, n, err := foldSystemTurns(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if n != 0 {
		t.Fatalf("folded %d, want 0", n)
	}
	if !bytes.Equal(in, out) {
		t.Fatalf("body was rewritten with nothing to fold:\n in: %s\nout: %s", in, out)
	}
}

// 2. SYSTEM-FIRST PASSTHROUGH — a leading system turn is legal for the
//    template, but the shim folds any system-role message it finds. Folding it
//    into the following user turn keeps the conversation valid and the content
//    in order; what must NOT happen is losing it.
func TestFold_LeadingSystem_NoPriorUser_FoldsIntoNextUser(t *testing.T) {
	in := []byte(`{"model":"m","messages":[{"role":"system","content":"SYS"},{"role":"user","content":"hello"}]}`)
	out, n, err := foldSystemTurns(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if n != 1 {
		t.Fatalf("folded %d, want 1", n)
	}
	eq(t, roles(t, out), []string{"user"}, "roles")
	eq(t, texts(t, out, 0), []string{"SYS", "hello"}, "order: system content precedes the user's")
}

// 3. THE REAL TRIGGER — a mid-conversation system turn, which is what Claude
//    Code 2.x sends and what Qwen's template refuses. It must fold BACKWARD
//    into the nearest preceding user turn.
func TestFold_MidConversationSystem_FoldsIntoPrecedingUser(t *testing.T) {
	in := []byte(`{"model":"m","messages":[
		{"role":"user","content":"first"},
		{"role":"assistant","content":"ack"},
		{"role":"user","content":"second"},
		{"role":"system","content":"Available agent types: ..."},
		{"role":"assistant","content":"ok"}
	]}`)
	out, n, err := foldSystemTurns(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if n != 1 {
		t.Fatalf("folded %d, want 1", n)
	}
	eq(t, roles(t, out), []string{"user", "assistant", "user", "assistant"}, "roles")
	eq(t, texts(t, out, 2), []string{"second", "Available agent types: ..."}, "folded into the NEAREST PRECEDING user")
	// and no system role survives anywhere
	for _, r := range roles(t, out) {
		if r == "system" {
			t.Fatalf("a system role survived the fold: %s", out)
		}
	}
}

// 4. NO USER AT ALL — the edge the shim's `pending` branch exists for. The
//    content must not be dropped; it becomes a user turn of its own.
func TestFold_NoUserAnywhere_AppendsUserTurn(t *testing.T) {
	in := []byte(`{"model":"m","messages":[{"role":"system","content":"SYS"},{"role":"assistant","content":"a"}]}`)
	out, n, err := foldSystemTurns(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if n != 1 {
		t.Fatalf("folded %d, want 1", n)
	}
	eq(t, roles(t, out), []string{"assistant", "user"}, "roles")
	eq(t, texts(t, out, 1), []string{"SYS"}, "content preserved as its own user turn")
}

// 5. MULTI-BLOCK content survives as blocks, and several system turns each
//    land on their own nearest preceding user.
func TestFold_MultiBlockAndMultipleSystems(t *testing.T) {
	in := []byte(`{"model":"m","messages":[
		{"role":"user","content":[{"type":"text","text":"u1a"},{"type":"text","text":"u1b"}]},
		{"role":"system","content":[{"type":"text","text":"s1"}]},
		{"role":"user","content":"u2"},
		{"role":"system","content":"s2"}
	]}`)
	out, n, err := foldSystemTurns(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if n != 2 {
		t.Fatalf("folded %d, want 2", n)
	}
	eq(t, roles(t, out), []string{"user", "user"}, "roles")
	eq(t, texts(t, out, 0), []string{"u1a", "u1b", "s1"}, "first system folds into first user, after its blocks")
	eq(t, texts(t, out, 1), []string{"u2", "s2"}, "second system folds into second user")
}

// 6. EVERYTHING ELSE IS UNTOUCHED, including top-level key ORDER — the part a
//    map-based round-trip silently breaks. Only `messages` may differ.
func TestFold_PreservesOtherFieldsAndKeyOrder(t *testing.T) {
	in := []byte(`{"model":"m","max_tokens":99,"messages":[{"role":"user","content":"u"},{"role":"system","content":"s"}],"tools":[{"name":"Read"}],"metadata":{"user_id":"x"},"temperature":0.7}`)
	out, n, err := foldSystemTurns(in)
	if err != nil || n != 1 {
		t.Fatalf("n=%d err=%v", n, err)
	}
	inOrd, err := parseOrdered(in)
	if err != nil {
		t.Fatalf("parse in: %v", err)
	}
	outOrd, err := parseOrdered(out)
	if err != nil {
		t.Fatalf("parse out: %v", err)
	}
	if len(inOrd) != len(outOrd) {
		t.Fatalf("field count changed: %d -> %d", len(inOrd), len(outOrd))
	}
	for i := range inOrd {
		if inOrd[i].Key != outOrd[i].Key {
			t.Fatalf("key order changed at %d: %q -> %q", i, inOrd[i].Key, outOrd[i].Key)
		}
		if inOrd[i].Key == "messages" {
			continue
		}
		if !bytes.Equal(inOrd[i].Val, outOrd[i].Val) {
			t.Fatalf("field %q was rewritten: %s -> %s", inOrd[i].Key, inOrd[i].Val, outOrd[i].Val)
		}
	}
}

// 7. A body it cannot parse is returned UNCHANGED with an error, so the caller
//    forwards the original and the upstream's own error is what the client
//    sees. An invented error here would be a worse answer than the real one.
func TestFold_Unparseable_ReturnsOriginal(t *testing.T) {
	in := []byte(`{"model":"m","messages":[{"role":"system","content":`)
	out, n, err := foldSystemTurns(in)
	if err == nil {
		t.Fatalf("want an error for a truncated body")
	}
	if n != 0 || !bytes.Equal(in, out) {
		t.Fatalf("body must come back untouched on error")
	}
}
