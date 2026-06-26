// Package gpu reports NVIDIA GPU state via nvidia-smi, for pinning decisions + the UI.
package gpu

import (
	"encoding/csv"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// GPU is one card's live state.
type GPU struct {
	Index    int    `json:"index"`
	Name     string `json:"name"`
	MemUsed  int    `json:"mem_used_mib"`
	MemTotal int    `json:"mem_total_mib"`
	Util     int    `json:"util_pct"`
	TempC    int    `json:"temp_c"`
}

func smiPath() string {
	if p, err := exec.LookPath("nvidia-smi"); err == nil {
		return p
	}
	// Windows fallback — nvidia-smi ships in System32.
	if win := os.Getenv("SystemRoot"); win != "" {
		p := filepath.Join(win, "System32", "nvidia-smi.exe")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "nvidia-smi"
}

// Query returns the current state of every visible NVIDIA GPU.
func Query() ([]GPU, error) {
	out, err := exec.Command(smiPath(),
		"--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(strings.NewReader(string(out)))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	var gpus []GPU
	for _, row := range rows {
		if len(row) < 6 {
			continue
		}
		gpus = append(gpus, GPU{
			Index:    atoi(row[0]),
			Name:     strings.TrimSpace(row[1]),
			MemUsed:  atoi(row[2]),
			MemTotal: atoi(row[3]),
			Util:     atoi(row[4]),
			TempC:    atoi(row[5]),
		})
	}
	return gpus, nil
}

func atoi(s string) int {
	n, _ := strconv.Atoi(strings.TrimSpace(s))
	return n
}
