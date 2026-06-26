// Package lms wraps LM Studio's `lms` CLI so llama-chip can manage inference runtimes
// through LM Studio's OWN official download mechanism (rather than reinventing a llama.cpp
// downloader). `lms runtime get` pulls runtime extensions from LM Studio's source;
// `lms runtime ls/select/update` manage what's installed.
package lms

import (
	"os"
	"os/exec"
	"path/filepath"
)

// Path locates the lms executable (PATH, then ~/.lmstudio/bin).
func Path() string {
	if p, err := exec.LookPath("lms"); err == nil {
		return p
	}
	home, _ := os.UserHomeDir()
	if p := filepath.Join(home, ".lmstudio", "bin", "lms.exe"); fileExists(p) {
		return p
	}
	if p := filepath.Join(home, ".lmstudio", "bin", "lms"); fileExists(p) {
		return p
	}
	return "lms"
}

// Available reports whether the lms CLI can be found + run.
func Available() bool {
	return exec.Command(Path(), "version").Run() == nil
}

// Run executes `lms <args...>` as an interactive passthrough (inherits stdio), so
// `llama-chip runtime get llama.cpp:cuda12` behaves exactly like the lms command.
func Run(args ...string) error {
	cmd := exec.Command(Path(), args...)
	cmd.Stdout, cmd.Stderr, cmd.Stdin = os.Stdout, os.Stderr, os.Stdin
	return cmd.Run()
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }
