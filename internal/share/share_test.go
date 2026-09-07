package share

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// blob makes deterministic, incompressible-ish content of n bytes.
func blob(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + i/251)
	}
	return b
}

func sum256(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func writeFile(t *testing.T, dir, name string, b []byte) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
	return p
}

// serveFile stands up a node that shares exactly one file by hash.
func serveFile(t *testing.T, path, sum string) *httptest.Server {
	t.Helper()
	h := Handler(func(q string) (string, bool) {
		if q == sum {
			return path, true
		}
		return "", false
	})
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv
}

// TestFetchWholeFile is the happy path: a peer has it, we get it, the name is ours.
func TestFetchWholeFile(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	data := blob(300_000)
	want := sum256(data)
	srv := serveFile(t, writeFile(t, src, "model.gguf", data), want)

	got, err := Fetch(srv.Client(), srv.URL, want, dst, nil)
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	b, err := os.ReadFile(got)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if sum256(b) != want {
		t.Fatalf("fetched content hash %s, want %s", sum256(b), want)
	}
	if _, err := os.Stat(filepath.Join(dst, want+".part")); !os.IsNotExist(err) {
		t.Fatalf(".part file should be gone after a verified fetch")
	}
}

// TestFetchResumesAfterTruncation is the test that matters: the first attempt dies mid-body,
// the second must send a Range and finish the file rather than starting again.
func TestFetchResumesAfterTruncation(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	data := blob(400_000)
	want := sum256(data)
	path := writeFile(t, src, "model.gguf", data)

	cut := int64(150_000)
	var sawRange string
	var attempts int

	real := Handler(func(q string) (string, bool) { return path, q == want })
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			// Truncate: send a prefix, then hang up without the rest.
			w.Header().Set("Content-Length", fmt.Sprint(len(data)))
			w.WriteHeader(200)
			_, _ = w.Write(data[:cut])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			// Returning early closes the connection mid-body: the client sees a short read.
			return
		}
		sawRange = r.Header.Get("Range")
		real(w, r)
	}))
	t.Cleanup(srv.Close)

	// First attempt: expected to fail, leaving a .part behind.
	if _, err := Fetch(srv.Client(), srv.URL, want, dst, nil); err == nil {
		t.Fatalf("first attempt should have failed on a truncated body")
	}
	part := filepath.Join(dst, want+".part")
	fi, err := os.Stat(part)
	if err != nil {
		t.Fatalf("expected a .part to resume from: %v", err)
	}
	if fi.Size() == 0 || fi.Size() >= int64(len(data)) {
		t.Fatalf(".part is %d bytes; want a partial prefix of %d", fi.Size(), len(data))
	}
	kept := fi.Size()

	// Second attempt: must resume.
	got, err := Fetch(srv.Client(), srv.URL, want, dst, nil)
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if sawRange == "" {
		t.Fatalf("resume did not send a Range header; it restarted from zero")
	}
	if wantHdr := fmt.Sprintf("bytes=%d-", kept); sawRange != wantHdr {
		t.Fatalf("Range = %q, want %q (resume must continue from what is on disk)", sawRange, wantHdr)
	}
	b, err := os.ReadFile(got)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if sum256(b) != want {
		t.Fatalf("resumed file hash %s, want %s — the join is wrong", sum256(b), want)
	}
}

// TestFetchRefusesBadHash: a peer serving the wrong bytes must not produce a named model.
func TestFetchRefusesBadHash(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	data := blob(50_000)
	claimed := sum256(blob(49_999)) // a hash the content does not have
	path := writeFile(t, src, "model.gguf", data)

	// A dishonest node: answers any hash with this file.
	srv := httptest.NewServer(Handler(func(string) (string, bool) { return path, true }))
	t.Cleanup(srv.Close)

	got, err := Fetch(srv.Client(), srv.URL, claimed, dst, nil)
	if err == nil {
		t.Fatalf("Fetch accepted content whose hash did not match (returned %s)", got)
	}
	if !strings.Contains(err.Error(), "hash mismatch") {
		t.Fatalf("error should name the mismatch, got: %v", err)
	}
	// The bad bytes must not occupy a model name.
	ents, _ := os.ReadDir(dst)
	for _, e := range ents {
		if !strings.HasSuffix(e.Name(), ".part") {
			t.Fatalf("unverified content took a real name: %s", e.Name())
		}
	}
}

// TestHandlerServesRanges checks the serving half honours a byte range.
func TestHandlerServesRanges(t *testing.T) {
	src := t.TempDir()
	data := blob(100_000)
	want := sum256(data)
	srv := serveFile(t, writeFile(t, src, "m.gguf", data), want)

	req, _ := http.NewRequest("GET", srv.URL+BlobPath+"?sha256="+want, nil)
	req.Header.Set("Range", "bytes=1000-1999")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("range request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent {
		t.Fatalf("status = %d, want 206", resp.StatusCode)
	}
	b, _ := io.ReadAll(resp.Body)
	if len(b) != 1000 {
		t.Fatalf("got %d bytes, want 1000", len(b))
	}
	if string(b) != string(data[1000:2000]) {
		t.Fatalf("ranged bytes do not match the source")
	}
	if got := resp.Header.Get("X-Content-SHA256"); got != want {
		t.Fatalf("X-Content-SHA256 = %q, want the whole-file hash", got)
	}
}

// TestHandlerRejects covers the input guards: unknown hash, malformed hash.
func TestHandlerRejects(t *testing.T) {
	srv := httptest.NewServer(Handler(func(string) (string, bool) { return "", false }))
	t.Cleanup(srv.Close)

	for _, tc := range []struct {
		q    string
		want int
	}{
		{"", http.StatusBadRequest},
		{"not-a-hash", http.StatusBadRequest},
		{strings.Repeat("z", 64), http.StatusBadRequest}, // right length, not hex
		{strings.Repeat("a", 64), http.StatusNotFound},   // well-formed, nobody has it
	} {
		resp, err := srv.Client().Get(srv.URL + BlobPath + "?sha256=" + tc.q)
		if err != nil {
			t.Fatalf("get %q: %v", tc.q, err)
		}
		resp.Body.Close()
		if resp.StatusCode != tc.want {
			t.Fatalf("sha256=%q -> %d, want %d", tc.q, resp.StatusCode, tc.want)
		}
	}
}

// TestIndexCachesAndInvalidates: the cache must not serve a hash for content that changed,
// or a node advertises one file and serves another.
func TestIndexCachesAndInvalidates(t *testing.T) {
	dir := t.TempDir()
	p := writeFile(t, dir, "m.gguf", blob(10_000))
	ix := OpenIndex(filepath.Join(dir, "index.json"))

	h1, err := ix.Hash(p)
	if err != nil {
		t.Fatalf("Hash: %v", err)
	}
	if got, ok := ix.Lookup(p); !ok || got != h1 {
		t.Fatalf("Lookup after Hash = %q,%v; want the cached hash", got, ok)
	}
	if found, ok := ix.FindByHash(h1); !ok || found != p {
		t.Fatalf("FindByHash = %q,%v; want %q", found, ok, p)
	}

	// Rewrite with different content and a later mtime.
	if err := os.WriteFile(p, blob(20_000), 0o644); err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	future := time.Now().Add(2 * time.Second)
	_ = os.Chtimes(p, future, future)

	if _, ok := ix.Lookup(p); ok {
		t.Fatalf("Lookup returned a stale hash after the file changed")
	}
	h2, err := ix.Hash(p)
	if err != nil {
		t.Fatalf("rehash: %v", err)
	}
	if h2 == h1 {
		t.Fatalf("hash did not change after the content did")
	}

	// A fresh Index over the same cache file must load what was persisted.
	ix2 := OpenIndex(filepath.Join(dir, "index.json"))
	if got, ok := ix2.Lookup(p); !ok || got != h2 {
		t.Fatalf("persisted cache = %q,%v; want %q", got, ok, h2)
	}
}

// TestOpenIndexCorrupt: a damaged cache costs re-hashing, never a refusal to run.
func TestOpenIndexCorrupt(t *testing.T) {
	dir := t.TempDir()
	cache := filepath.Join(dir, "index.json")
	if err := os.WriteFile(cache, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	ix := OpenIndex(cache)
	p := writeFile(t, dir, "m.gguf", blob(1000))
	if _, err := ix.Hash(p); err != nil {
		t.Fatalf("a corrupt cache must not break hashing: %v", err)
	}
}
