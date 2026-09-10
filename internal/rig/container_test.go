package rig

import (
	"strings"
	"testing"

	"github.com/cpuchip/llama-chip/internal/config"
)

func labSlot() config.Slot {
	return config.Slot{
		Alias:     "qwen-lab",
		GPUs:      []int{1},
		APIKeyEnv: "LAB_KEY",
		Container: &config.Container{
			Image:  "example/vllm:tag",
			Cmd:    "bash docker/entrypoint.sh single",
			Pre:    "sed -i 's/^MAX_QUERY_LEN_3D = 16$/MAX_QUERY_LEN_3D = 32/' /app/x.py",
			Env:    map[string]string{"MAX_LEN": "100000", "GPU_UTIL": "0.88", "DFLASH_TOKENS": "26"},
			Mounts: []string{"C:/models:/app/models", "cache-vol:/cache"},
			Extra:  []string{"--ipc", "host", "--shm-size", "64m"},
		},
	}
}

// The argv carries the knobs, the device, the loopback port and the pre snippet, and never a
// secret: the key variable is passed by NAME only.
func TestContainerArgsShape(t *testing.T) {
	t.Setenv("LAB_KEY", "sk-secret-value")
	s := labSlot()
	args := containerArgs(s, []string{"GPU-abc"}, 18101)
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"run -d --name llama-chip-qwen-lab",
		"--gpus device=GPU-abc",
		"-e NVIDIA_VISIBLE_DEVICES=GPU-abc",
		"-e CUDA_VISIBLE_DEVICES=GPU-abc",
		"-p 127.0.0.1:18101:18020",
		"-v C:/models:/app/models",
		"-v cache-vol:/cache",
		"-e DFLASH_TOKENS=26 -e GPU_UTIL=0.88 -e MAX_LEN=100000", // sorted, stable
		"-e VLLM_API_KEY ",                                        // name only
		"--ipc host --shm-size 64m",
		"--entrypoint bash example/vllm:tag -c",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("argv lacks %q:\n%s", want, joined)
		}
	}
	if strings.Contains(joined, "sk-secret-value") {
		t.Fatalf("the bearer leaked into the docker argv:\n%s", joined)
	}
	script := args[len(args)-1]
	if !strings.HasPrefix(script, "sed -i") || !strings.HasSuffix(script, "exec bash docker/entrypoint.sh single") {
		t.Errorf("script should be the pre snippet then exec cmd, got %q", script)
	}
}

// Without a command or a pre snippet the image's own entrypoint runs (no --entrypoint override).
func TestContainerArgsImageDefault(t *testing.T) {
	s := labSlot()
	s.Container.Cmd, s.Container.Pre = "", ""
	args := containerArgs(s, nil, 18101)
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--entrypoint") || strings.Contains(joined, "--gpus") {
		t.Errorf("no entrypoint override or gpus expected:\n%s", joined)
	}
	if args[len(args)-1] != "example/vllm:tag" {
		t.Errorf("image should be last, got %q", args[len(args)-1])
	}
}

// The log line never shows an env value, so a key variable given with a value elsewhere would
// still not reach the rig log.
func TestRedactedArgsHidesEnvValues(t *testing.T) {
	got := redactedArgs([]string{"run", "-e", "MAX_LEN=100000", "-e", "VLLM_API_KEY", "img"})
	if strings.Contains(got, "100000") {
		t.Errorf("env value shown: %s", got)
	}
	if !strings.Contains(got, "-e MAX_LEN=…") || !strings.Contains(got, "-e VLLM_API_KEY") {
		t.Errorf("unexpected rendering: %s", got)
	}
}

func TestContainerSlotValidation(t *testing.T) {
	s := labSlot()
	if err := s.Container.Validate(s); err != nil {
		t.Fatalf("valid slot rejected: %v", err)
	}
	s.Alias = ""
	if err := s.Container.Validate(s); err == nil {
		t.Error("alias-less container slot accepted")
	}
	s = labSlot()
	s.Container.Image = " "
	if err := s.Container.Validate(s); err == nil {
		t.Error("image-less container slot accepted")
	}
	s = labSlot()
	s.Container.Mounts = []string{"nocolon"}
	if err := s.Container.Validate(s); err == nil {
		t.Error("malformed mount accepted")
	}
}

func TestContainerNameIsDockerSafe(t *testing.T) {
	s := config.Slot{Alias: "Qwen 3.8/int4 k=26"}
	if got := containerName(s); got != "llama-chip-qwen-3.8-int4-k-26" {
		t.Errorf("got %q", got)
	}
}
