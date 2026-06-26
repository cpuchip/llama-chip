//go:build !linux

package rig

import "os/exec"

// setChildReaping is a no-op off Linux (PR_SET_PDEATHSIG is Linux-specific). The graceful
// SIGINT/SIGTERM → StopAll path still reaps children on a normal stop; only the abrupt-death
// belt (parent SIGKILL/crash) is Linux-only.
func setChildReaping(cmd *exec.Cmd) {}
