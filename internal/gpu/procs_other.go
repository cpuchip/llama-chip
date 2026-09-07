//go:build !windows

package gpu

import (
	"os/exec"
)

func processUsage(gpus []GPU) ([]ProcUsage, error) {
	out, err := exec.Command(smiPath(),
		"--query-compute-apps=gpu_uuid,pid,used_memory",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, err
	}
	return parseComputeApps(string(out), gpus), nil
}
