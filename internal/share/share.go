// Package share moves GGUF files between fleet nodes over the mesh, peer to peer.
//
// Michael's mark: never through the hub. A node that has the file serves it; a node that
// wants it fetches it directly and verifies what arrived. The hub is a roster, not a pipe.
//
// Two halves:
//
//   - Serve: Handler answers GET /api/models/blob?sha256=<hex> with the file, honouring HTTP
//     ranges so an interrupted fetch resumes instead of restarting. A 40 GiB model over a home
//     link does not get to start again because a laptop slept.
//   - Fetch: Fetch downloads to a .part file, resumes from whatever is already there, and
//     verifies the sha256 before the file is given its real name. A partial or corrupted
//     download never occupies the name of a good model.
//
// Hashing is cached (see Index): a GGUF is tens of gigabytes and hashing one takes minutes,
// so /api/models cannot hash on every call. The cache is keyed by path, size and mtime, and
// a file whose size or mtime moved is rehashed.
package share

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// BlobPath is the endpoint a serving node exposes. Kept here so the router and the client
// cannot drift apart.
const BlobPath = "/api/models/blob"

// Entry is one file's identity in the hash cache.
type Entry struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	ModSec int64  `json:"mod_sec"` // mtime, whole seconds — enough to notice a rewrite
	SHA256 string `json:"sha256"`
}

// stale reports whether the cached entry no longer describes the file on disk.
func (e Entry) stale(fi os.FileInfo) bool {
	return e.Size != fi.Size() || e.ModSec != fi.ModTime().Unix()
}

// Index is a persistent path -> sha256 cache. Safe for concurrent use.
//
// It deliberately does NOT hash on construction: a fleet node starting up must not stall for
// minutes on a library of large files. Hashes are computed on demand and persisted as they
// are learned.
type Index struct {
	mu    sync.Mutex
	path  string // where the cache is persisted
	byKey map[string]Entry
}

// OpenIndex loads (or starts) the cache at path. A missing or corrupt cache is not an error:
// it costs re-hashing, not correctness.
func OpenIndex(path string) *Index {
	ix := &Index{path: path, byKey: map[string]Entry{}}
	b, err := os.ReadFile(path)
	if err != nil {
		return ix
	}
	var list []Entry
	if json.Unmarshal(b, &list) != nil {
		return ix // corrupt cache: start clean rather than refuse to run
	}
	for _, e := range list {
		ix.byKey[e.Path] = e
	}
	return ix
}

// Lookup returns a cached hash if it still matches the file on disk.
func (ix *Index) Lookup(path string) (string, bool) {
	fi, err := os.Stat(path)
	if err != nil {
		return "", false
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	e, ok := ix.byKey[path]
	if !ok || e.stale(fi) {
		return "", false
	}
	return e.SHA256, true
}

// Hash returns the file's sha256, computing and caching it on a miss. Callers that must not
// block should use Lookup and accept the miss.
func (ix *Index) Hash(path string) (string, error) {
	if h, ok := ix.Lookup(path); ok {
		return h, nil
	}
	fi, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	sum := hex.EncodeToString(h.Sum(nil))

	// Re-stat before caching: a file rewritten while we hashed would otherwise be cached
	// under a hash of content that no longer exists.
	if fi2, err := os.Stat(path); err != nil || fi2.Size() != fi.Size() || !fi2.ModTime().Equal(fi.ModTime()) {
		return sum, nil // return the hash we computed, but do not cache a moving target
	}
	ix.mu.Lock()
	ix.byKey[path] = Entry{Path: path, Size: fi.Size(), ModSec: fi.ModTime().Unix(), SHA256: sum}
	ix.mu.Unlock()
	ix.save()
	return sum, nil
}

// FindByHash returns the path of a cached file with this hash, if the node has one.
func (ix *Index) FindByHash(sum string) (string, bool) {
	sum = strings.ToLower(strings.TrimSpace(sum))
	ix.mu.Lock()
	defer ix.mu.Unlock()
	for p, e := range ix.byKey {
		if e.SHA256 == sum {
			return p, true
		}
	}
	return "", false
}

// save persists the cache. Best effort: a node that cannot write its cache still serves.
func (ix *Index) save() {
	ix.mu.Lock()
	list := make([]Entry, 0, len(ix.byKey))
	for _, e := range ix.byKey {
		list = append(list, e)
	}
	ix.mu.Unlock()
	sort.Slice(list, func(i, j int) bool { return list[i].Path < list[j].Path })
	b, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(ix.path), 0o755); err != nil {
		return
	}
	tmp := ix.path + ".tmp"
	if os.WriteFile(tmp, b, 0o644) == nil {
		_ = os.Rename(tmp, ix.path) // atomic: a reader never sees a half-written cache
	}
}

// Handler serves GET /api/models/blob?sha256=<hex> from the node's own library.
//
// resolve maps a hash to a local path; it is supplied by the caller so this package does not
// depend on the models package (and so tests can serve a temp dir).
func Handler(resolve func(sum string) (string, bool)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		sum := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("sha256")))
		if !validHash(sum) {
			http.Error(w, "sha256 query parameter required (64 hex chars)", http.StatusBadRequest)
			return
		}
		path, ok := resolve(sum)
		if !ok {
			http.Error(w, "no local file with that hash", http.StatusNotFound)
			return
		}
		f, err := os.Open(path)
		if err != nil {
			http.Error(w, "cannot open file", http.StatusInternalServerError)
			return
		}
		defer f.Close()
		fi, err := f.Stat()
		if err != nil {
			http.Error(w, "cannot stat file", http.StatusInternalServerError)
			return
		}
		// The hash IS the identity, so the content can never change under this URL.
		w.Header().Set("ETag", `"`+sum+`"`)
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("X-Content-SHA256", sum)
		// ServeContent gives us range requests, If-Range and 206 handling for free — which is
		// the whole resumability story on the serving side.
		http.ServeContent(w, r, filepath.Base(path), fi.ModTime(), f)
	}
}

func validHash(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// Progress reports fetch advancement. Called at most once a second.
type Progress func(done, total int64)

// Fetch downloads the blob with the given hash from peer into dstDir, resuming any prior
// partial download, and verifies the sha256 before the file takes its final name.
//
// Returns the final path. The name is chosen by the server's Content-Disposition when it
// offers one, else the hash — a fetched file is never allowed to land on a name we did not
// derive from the response.
func Fetch(client *http.Client, peer, sum, dstDir string, prog Progress) (string, error) {
	sum = strings.ToLower(strings.TrimSpace(sum))
	if !validHash(sum) {
		return "", fmt.Errorf("share: %q is not a sha256 hex digest", sum)
	}
	if client == nil {
		client = &http.Client{Timeout: 0} // large files: no overall deadline, the body governs
	}
	base := strings.TrimRight(peer, "/")
	u := base + BlobPath + "?sha256=" + url.QueryEscape(sum)

	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return "", err
	}
	part := filepath.Join(dstDir, sum+".part")

	var have int64
	if fi, err := os.Stat(part); err == nil {
		have = fi.Size()
	}

	req, err := http.NewRequest(http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// Server ignored the range (or we had nothing): start over from zero.
		have = 0
	case http.StatusPartialContent:
		// Resuming; keep what we have.
	case http.StatusRequestedRangeNotSatisfiable:
		// We already hold at least as many bytes as the server has. Fall through to verify:
		// either it checks out, or it is junk and we say so rather than silently retrying.
		return verifyAndName(part, sum, dstDir, resp)
	default:
		return "", fmt.Errorf("share: peer returned %s", resp.Status)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if have > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return "", err
	}

	total := have + resp.ContentLength
	if resp.ContentLength < 0 {
		total = 0 // unknown; progress reports done only
	}
	written := have
	last := time.Now()
	buf := make([]byte, 1<<20)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				f.Close()
				return "", werr
			}
			written += int64(n)
			if prog != nil && time.Since(last) >= time.Second {
				prog(written, total)
				last = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			// Partial file is kept on purpose: the next call resumes from here.
			return "", fmt.Errorf("share: transfer interrupted after %d bytes (resume by re-running): %w", written, rerr)
		}
	}
	if err := f.Close(); err != nil {
		return "", err
	}
	if prog != nil {
		prog(written, total)
	}
	return verifyAndName(part, sum, dstDir, resp)
}

// verifyAndName hashes the completed .part, refuses it if it does not match, and otherwise
// renames it into place.
func verifyAndName(part, sum, dstDir string, resp *http.Response) (string, error) {
	got, err := fileSHA256(part)
	if err != nil {
		return "", err
	}
	if got != sum {
		// Do NOT delete: an operator may want the bytes. But never let it take the real name.
		return "", fmt.Errorf("share: hash mismatch — wanted %s, got %s (kept as %s)", sum, got, part)
	}
	name := nameFromResponse(resp)
	if name == "" {
		name = sum + ".gguf"
	}
	final := filepath.Join(dstDir, name)
	if err := os.Rename(part, final); err != nil {
		return "", err
	}
	return final, nil
}

// nameFromResponse extracts a safe base name from Content-Disposition, if present.
func nameFromResponse(resp *http.Response) string {
	if resp == nil {
		return ""
	}
	cd := resp.Header.Get("Content-Disposition")
	if cd == "" {
		return ""
	}
	_, params, err := mime.ParseMediaType(cd)
	if err != nil {
		return ""
	}
	n := filepath.Base(params["filename"]) // Base defeats "../.." in a hostile filename
	if n == "." || n == string(filepath.Separator) || n == "" {
		return ""
	}
	return n
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
