package rig

import (
	"io"
	"log"
	"testing"

	"github.com/cpuchip/llama-chip/internal/config"
)

// The API's container launch takes what touches the host from the lab block and only the
// knobs from the request: a client that sends its own image, mounts or docker flags gets the
// config's instead.
func TestLabSlotTakesHostFieldsFromConfig(t *testing.T) {
	r := NewEmptyForTest(log.New(io.Discard, "", 0))
	r.cfg = &config.Config{Lab: &config.Lab{
		Image: "cfg/image:tag", Mounts: []string{"D:/models:/app/models"}, Extra: []string{"--ipc", "host"},
		APIKeyEnv: "CFG_KEY", KeyEnvIn: "VLLM_API_KEY", Port: 18020, Served: "served-name", BasePort: 18100,
		Env: map[string]string{"HOME": "/cache", "GPU_UTIL": "0.88"},
	}}
	req := config.Slot{Alias: "lab-a", GPUs: []int{1}, Port: 18105, APIKeyEnv: "ATTACKER_KEY",
		Container: &config.Container{Image: "evil/image", Mounts: []string{"C:/:/host"}, Extra: []string{"--privileged"},
			Cmd: "bash run.sh", Pre: "echo hi", Env: map[string]string{"GPU_UTIL": "0.93", "MAX_LEN": "100000"}}}
	s, err := r.LabSlot(req)
	if err != nil {
		t.Fatal(err)
	}
	ct := s.Container
	if ct.Image != "cfg/image:tag" || len(ct.Mounts) != 1 || ct.Mounts[0] != "D:/models:/app/models" || len(ct.Extra) != 2 || ct.Extra[1] != "host" {
		t.Errorf("host fields not taken from the config: %+v", ct)
	}
	if s.APIKeyEnv != "CFG_KEY" || ct.KeyEnvIn != "VLLM_API_KEY" || ct.Served != "served-name" || ct.Port != 18020 {
		t.Errorf("key/served/port not from the config: slot %+v container %+v", s, ct)
	}
	if s.Alias != "lab-a" || s.Port != 18105 || len(s.GPUs) != 1 || s.GPUs[0] != 1 || ct.Cmd != "bash run.sh" || ct.Pre != "echo hi" {
		t.Errorf("knobs not taken from the request: slot %+v container %+v", s, ct)
	}
	if ct.Env["GPU_UTIL"] != "0.93" || ct.Env["MAX_LEN"] != "100000" || ct.Env["HOME"] != "/cache" {
		t.Errorf("env should be lab env under request env: %v", ct.Env)
	}
	if _, err := r.LabSlot(config.Slot{Alias: "x"}); err == nil {
		t.Error("a non-container slot should be refused")
	}
	r.cfg = nil
	if _, err := r.LabSlot(req); err == nil {
		t.Error("no lab block: launch must be refused")
	}
}

func TestNextLabPortSkipsHeldPorts(t *testing.T) {
	r := NewEmptyForTest(log.New(io.Discard, "", 0))
	r.cfg = &config.Config{Lab: &config.Lab{BasePort: 18100}}
	if err := r.AddContainerForTest(config.Slot{Alias: "a", Port: 18100, Container: &config.Container{Image: "i"}}, "http://127.0.0.1:18100"); err != nil {
		t.Fatal(err)
	}
	r.mu.Lock()
	p := r.nextLabPortLocked()
	r.mu.Unlock()
	if p != 18101 {
		t.Errorf("next lab port %d, want 18101", p)
	}
}
