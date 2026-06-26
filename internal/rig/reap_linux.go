//go:build linux

package rig

import (
	"os/exec"
	"syscall"
)

// setChildReaping asks the kernel to SIGKILL this llama-server if the llama-chip supervisor dies
// abruptly (PR_SET_PDEATHSIG) — the belt to the SIGINT/SIGTERM handler's braces (StopAll). Without
// it, a SIGKILL'd or crashed llama-chip orphans its llama-server children, which keep holding GPU
// VRAM and their backend ports until a manual kill.
//
// Caveat: Pdeathsig is delivered when the OS *thread* that started the child exits. Our supervisor
// runs one goroutine per instance (launch → Wait), so the relationship holds in practice; if the Go
// runtime ever reaped that thread early this could fire prematurely — the graceful StopAll path
// (SIGINT/SIGTERM, now handled) remains the primary reaper.
func setChildReaping(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
}
