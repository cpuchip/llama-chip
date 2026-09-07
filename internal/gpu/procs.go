package gpu

import (
	"encoding/csv"
	"errors"
	"regexp"
	"strconv"
	"strings"
)

// ProcUsage is one process's dedicated VRAM on one GPU, as the platform reports it.
// GPUIndex is -1 when the platform named an adapter we could not map to an nvidia-smi index
// (Windows names adapters by LUID; the mapping is learned from usage, see mapLUIDs).
type ProcUsage struct {
	PID      int    `json:"pid"`
	GPUIndex int    `json:"gpu"`
	UUID     string `json:"uuid,omitempty"`
	LUID     string `json:"luid,omitempty"`
	MiB      int    `json:"mib"`
}

// AdapterUsage is one adapter's total dedicated VRAM as Windows reports it (by LUID).
type AdapterUsage struct {
	LUID string
	MiB  int
}

// ProcessUsage lists per-process dedicated VRAM per GPU. Windows reads the GPU perf counters
// through typeperf (nvidia-smi cannot attribute per process there); Linux reads nvidia-smi's
// compute-apps query. gpus is the current Query() result, used to map adapters to indexes.
var ProcessUsage = processUsage

var (
	luidRe    = regexp.MustCompile(`pid_(\d+)_luid_(0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)_phys_\d+`)
	adapterRe = regexp.MustCompile(`GPU Adapter Memory\((luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)_phys_\d+\)`)
)

// parseTypeperf parses the CSV typeperf prints for the two GPU memory counters: a header row of
// instance names and one data row of byte values.
func parseTypeperf(out string) (procs []ProcUsage, adapters []AdapterUsage, err error) {
	r := csv.NewReader(strings.NewReader(out))
	r.FieldsPerRecord = -1
	rows, rerr := r.ReadAll()
	if rerr != nil {
		return nil, nil, rerr
	}
	var header, data []string
	for _, row := range rows {
		if len(row) < 2 {
			continue
		}
		if strings.Contains(row[0], "PDH-CSV") {
			header = row
			continue
		}
		if header != nil && data == nil {
			data = row
		}
	}
	if header == nil || data == nil {
		return nil, nil, errors.New("typeperf: no header or data row")
	}
	for i := 1; i < len(header) && i < len(data); i++ {
		bytesF, perr := strconv.ParseFloat(strings.TrimSpace(data[i]), 64)
		if perr != nil {
			continue
		}
		mib := int(bytesF / (1 << 20))
		if m := luidRe.FindStringSubmatch(header[i]); m != nil && strings.Contains(header[i], "GPU Process Memory") {
			pid, _ := strconv.Atoi(m[1])
			procs = append(procs, ProcUsage{PID: pid, GPUIndex: -1, LUID: "luid_" + strings.ToLower(m[2]), MiB: mib})
			continue
		}
		if m := adapterRe.FindStringSubmatch(header[i]); m != nil {
			adapters = append(adapters, AdapterUsage{LUID: strings.ToLower(m[1]), MiB: mib})
		}
	}
	return procs, adapters, nil
}

// luidToIndex remembers which Windows adapter LUID is which nvidia-smi index. Learned, never
// configured: an adapter whose dedicated usage matches one card's memory.used (within
// luidMatchMiB) and no other card's is that card. LUIDs are stable for a boot.
var luidToIndex = map[string]int{}

const luidMatchMiB = 768

// LearnLUID pins an adapter LUID to a card index from outside knowledge: the rig knows which
// card it pinned a llama-server to, so that process's LUID names that card, whatever the usage
// levels say. A stronger signal than usage matching; it wins.
func LearnLUID(luid string, index int) {
	if luid == "" || index < 0 {
		return
	}
	luidToIndex[luid] = index
}

// mapLUIDs updates luidToIndex from a fresh adapter list and the cards' current usage, then
// stamps every proc with the index its LUID maps to (-1 when unknown). Cards with no usage are
// indistinguishable from each other and from a non-NVIDIA adapter, so they stay unmapped until
// something uses them; a card nothing uses has no foreign usage to report anyway.
func mapLUIDs(procs []ProcUsage, adapters []AdapterUsage, gpus []GPU) []ProcUsage {
	for _, a := range adapters {
		if _, known := luidToIndex[a.LUID]; known {
			continue
		}
		best, second := -1, -1
		bestD, secondD := luidMatchMiB+1, luidMatchMiB+1
		for _, g := range gpus {
			taken := false
			for _, idx := range luidToIndex {
				if idx == g.Index {
					taken = true
				}
			}
			if taken {
				continue
			}
			d := g.MemUsed - a.MiB
			if d < 0 {
				d = -d
			}
			if d < bestD {
				second, secondD = best, bestD
				best, bestD = g.Index, d
			} else if d < secondD {
				second, secondD = g.Index, d
			}
		}
		_ = second
		if best >= 0 && bestD <= luidMatchMiB && secondD > luidMatchMiB && a.MiB > luidMatchMiB {
			luidToIndex[a.LUID] = best
		}
	}
	for i := range procs {
		if idx, ok := luidToIndex[procs[i].LUID]; ok {
			procs[i].GPUIndex = idx
		}
	}
	return procs
}

// parseComputeApps parses nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory (Linux),
// mapping each row's UUID to its index through gpus.
func parseComputeApps(out string, gpus []GPU) []ProcUsage {
	byUUID := map[string]int{}
	for _, g := range gpus {
		byUUID[g.UUID] = g.Index
	}
	var procs []ProcUsage
	for _, line := range strings.Split(out, "\n") {
		f := strings.Split(line, ",")
		if len(f) < 3 {
			continue
		}
		uuid := strings.TrimSpace(f[0])
		pid, _ := strconv.Atoi(strings.TrimSpace(f[1]))
		mib, err := strconv.Atoi(strings.TrimSpace(f[2]))
		if err != nil || pid == 0 {
			continue
		}
		idx, ok := byUUID[uuid]
		if !ok {
			idx = -1
		}
		procs = append(procs, ProcUsage{PID: pid, GPUIndex: idx, UUID: uuid, MiB: mib})
	}
	return procs
}
