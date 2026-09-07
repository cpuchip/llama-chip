package gpu

import (
	"os/exec"
)

func processUsage(gpus []GPU) ([]ProcUsage, error) {
	out, err := exec.Command("typeperf",
		`\GPU Process Memory(*)\Dedicated Usage`,
		`\GPU Adapter Memory(*)\Dedicated Usage`,
		"-sc", "1").Output()
	if err != nil {
		return nil, err
	}
	procs, adapters, err := parseTypeperf(string(out))
	if err != nil {
		return nil, err
	}
	return mapLUIDs(procs, adapters, gpus), nil
}
