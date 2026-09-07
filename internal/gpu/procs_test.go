package gpu

import "testing"

const typeperfSample = `"(PDH-CSV 4.0)","\\FERMION\GPU Process Memory(pid_42832_luid_0x00000000_0x00013C6C_phys_0)\Dedicated Usage","\\FERMION\GPU Process Memory(pid_11852_luid_0x00000000_0x00015C7C_phys_0)\Dedicated Usage","\\FERMION\GPU Process Memory(pid_777_luid_0x00000000_0x00014CA2_phys_0)\Dedicated Usage","\\FERMION\GPU Adapter Memory(luid_0x00000000_0x00013C6C_phys_0)\Dedicated Usage","\\FERMION\GPU Adapter Memory(luid_0x00000000_0x00015C7C_phys_0)\Dedicated Usage","\\FERMION\GPU Adapter Memory(luid_0x00000000_0x00014CA2_phys_0)\Dedicated Usage"
"09/07/2026 04:30:00.000","24813305856","4086366208","4194304","24817500160","4090560512","4194304"
`

func TestParseTypeperfAndMapLUIDs(t *testing.T) {
	procs, adapters, err := parseTypeperf(typeperfSample)
	if err != nil {
		t.Fatal(err)
	}
	if len(procs) != 3 || len(adapters) != 3 {
		t.Fatalf("procs %d adapters %d", len(procs), len(adapters))
	}
	if procs[0].PID != 42832 || procs[0].MiB != 23663 || procs[0].LUID != "luid_0x00000000_0x00013c6c" {
		t.Fatalf("proc 0 = %+v", procs[0])
	}
	// Two cards: card 0 holds the 23.6 GB container, card 1 is idle. Only card 0 is mappable;
	// the AMD adapter (3.9 GB) matches no NVIDIA card within tolerance; the 4 MB adapter is
	// below the learning floor.
	luidToIndex = map[string]int{}
	gpus := []GPU{{Index: 0, MemUsed: 23664, MemTotal: 24564}, {Index: 1, MemUsed: 0, MemTotal: 24564}}
	procs = mapLUIDs(procs, adapters, gpus)
	if procs[0].GPUIndex != 0 {
		t.Fatalf("container proc should map to card 0, got %d", procs[0].GPUIndex)
	}
	if procs[1].GPUIndex != -1 || procs[2].GPUIndex != -1 {
		t.Fatalf("iGPU and tiny adapter must stay unmapped: %+v %+v", procs[1], procs[2])
	}
	// The mapping is remembered across samples even when the usage changes.
	procs2, adapters2, _ := parseTypeperf(typeperfSample)
	procs2 = mapLUIDs(procs2, adapters2, []GPU{{Index: 0, MemUsed: 100}, {Index: 1, MemUsed: 0}})
	if procs2[0].GPUIndex != 0 {
		t.Fatalf("learned mapping lost: %+v", procs2[0])
	}
}

func TestParseComputeApps(t *testing.T) {
	out := "GPU-aaaa, 1234, 15100\nGPU-bbbb, 99, 300\n[N/A], 5, [N/A]\n"
	procs := parseComputeApps(out, []GPU{{Index: 0, UUID: "GPU-aaaa"}, {Index: 1, UUID: "GPU-bbbb"}})
	if len(procs) != 2 {
		t.Fatalf("got %d procs: %+v", len(procs), procs)
	}
	if procs[0].GPUIndex != 0 || procs[0].MiB != 15100 || procs[1].GPUIndex != 1 || procs[1].PID != 99 {
		t.Fatalf("bad parse: %+v", procs)
	}
}
