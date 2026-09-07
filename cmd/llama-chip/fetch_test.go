package main

import (
	"reflect"
	"testing"
)

// TestSplitArgs guards the bug that shipped first: Go's flag package stops at the first
// non-flag argument, so the documented form `fetch <model> --from <peer>` silently lost
// --from and reported it as missing.
func TestSplitArgs(t *testing.T) {
	for _, tc := range []struct {
		name       string
		in         []string
		wantFlags  []string
		wantPosArg []string
	}{
		{"documented form: model first",
			[]string{"qwen3", "--from", "http://peer:8090"},
			[]string{"--from", "http://peer:8090"}, []string{"qwen3"}},
		{"flags first",
			[]string{"--from", "http://peer:8090", "qwen3"},
			[]string{"--from", "http://peer:8090"}, []string{"qwen3"}},
		{"equals form does not eat the next token",
			[]string{"--from=http://peer:8090", "qwen3"},
			[]string{"--from=http://peer:8090"}, []string{"qwen3"}},
		{"single dash",
			[]string{"qwen3", "-from", "http://peer:8090"},
			[]string{"-from", "http://peer:8090"}, []string{"qwen3"}},
		{"no positional at all",
			[]string{"--sha256", "abc", "--from", "http://p"},
			[]string{"--sha256", "abc", "--from", "http://p"}, nil},
		{"double dash ends flags",
			[]string{"--from", "http://p", "--", "-weird-model-name"},
			[]string{"--from", "http://p"}, []string{"-weird-model-name"}},
		{"valueless flag keeps the positional",
			[]string{"-h", "qwen3"},
			[]string{"-h"}, []string{"qwen3"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, p := splitArgs(tc.in)
			if !reflect.DeepEqual(f, tc.wantFlags) {
				t.Errorf("flags = %q, want %q", f, tc.wantFlags)
			}
			if !reflect.DeepEqual(p, tc.wantPosArg) {
				t.Errorf("positional = %q, want %q", p, tc.wantPosArg)
			}
		})
	}
}
