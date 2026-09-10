package rig

// Container slots: a server the rig launches with `docker run` and then fronts exactly like an
// external slot (the router proxies to 127.0.0.1:<host port> with the slot's bearer). Born
// 2026-09-10 for the lab panel: a vLLM image has a dozen launch-time knobs (context, memory
// fraction, draft length, KV tier, an experiment's one-line edit) and the only way to get a
// feel for them was a shell script per cell. Here the knobs are the container's environment,
// the boot log streams into the slot's tail, and a crash shows its reason in the loaded table.
//
// There is no restart: a lab boot that dies did so because of its settings, and the reason is
// the result. Unload removes the container.

import (
	"bufio"
	"bytes"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"net/url"

	"github.com/cpuchip/llama-chip/internal/config"
	"github.com/cpuchip/llama-chip/internal/gpu"
	"github.com/cpuchip/llama-chip/internal/models"
)

// containerName is the docker name the rig gives a slot's container.
func containerName(s config.Slot) string {
	name := strings.ToLower(s.Name())
	var b strings.Builder
	for _, c := range name {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' {
			b.WriteRune(c)
		} else {
			b.WriteByte('-')
		}
	}
	return "llama-chip-" + b.String()
}

// containerArgs is the `docker run` argument list for a container slot. Pure: given the slot,
// the resolved GPU device ids and the host port, it returns the argv after "docker". The
// bearer is NOT in it: the key variable is passed by name only (docker copies the value from
// the launching process's environment), so a process listing never shows a key.
func containerArgs(s config.Slot, devices []string, hostPort int) []string {
	ct := s.Container
	args := []string{"run", "-d", "--name", containerName(s)}
	if len(devices) > 0 {
		ids := strings.Join(devices, ",")
		args = append(args, "--gpus", "device="+ids)
		// Inside the container the exposed cards are numbered from 0, so a host INDEX would
		// hide the very card just exposed; only a UUID names the same card on both sides.
		if strings.HasPrefix(devices[0], "GPU-") {
			args = append(args, "-e", "NVIDIA_VISIBLE_DEVICES="+ids, "-e", "CUDA_VISIBLE_DEVICES="+ids)
		}
	}
	args = append(args, "-p", fmt.Sprintf("127.0.0.1:%d:%d", hostPort, ct.ContainerPort()))
	for _, m := range ct.Mounts {
		args = append(args, "-v", m)
	}
	// The knobs, in a stable order so two launches with the same settings read the same.
	keys := make([]string, 0, len(ct.Env))
	for k := range ct.Env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		args = append(args, "-e", k+"="+ct.Env[k])
	}
	if s.APIKey != "" || s.APIKeyEnv != "" {
		args = append(args, "-e", ct.KeyVar()) // name only: the value rides in the environment
	}
	args = append(args, ct.Extra...)
	script := containerScript(ct)
	if script != "" {
		args = append(args, "--entrypoint", "bash", ct.Image, "-c", script)
	} else {
		args = append(args, ct.Image)
	}
	return args
}

// containerScript is what bash runs inside the container: the pre snippet, then the command
// (exec'd so signals reach the server). Empty when the image's own entrypoint should run.
func containerScript(ct *config.Container) string {
	pre := strings.TrimSpace(ct.Pre)
	cmd := strings.TrimSpace(ct.Cmd)
	if pre == "" && cmd == "" {
		return ""
	}
	var parts []string
	if pre != "" {
		parts = append(parts, pre)
	}
	if cmd != "" {
		parts = append(parts, "exec "+cmd)
	}
	return strings.Join(parts, "\n")
}

// deviceIDs maps the slot's GPU indexes to nvidia-smi UUIDs (a stable identity; indexes can
// move between boots and a container's own numbering is its own). Falls back to the index when
// the query fails, which docker also accepts.
func deviceIDs(gpus []int) []string {
	byIndex := map[int]string{}
	if list, err := gpu.Query(); err == nil {
		for _, g := range list {
			if g.UUID != "" {
				byIndex[g.Index] = g.UUID
			}
		}
	}
	out := make([]string, 0, len(gpus))
	for _, i := range gpus {
		if u, ok := byIndex[i]; ok {
			out = append(out, u)
		} else {
			out = append(out, strconv.Itoa(i))
		}
	}
	return out
}

// loadContainer is Load for a container slot: docker run, then front it as an external slot
// whose health is polled and whose log is tailed into the slot.
func (r *Rig) loadContainer(s config.Slot) error {
	ct := s.Container
	if err := ct.Validate(s); err != nil {
		return err
	}
	name := containerName(s)
	// Register under ONE lock: the host port is chosen and the instance published together,
	// so two concurrent launches cannot be handed the same port.
	r.mu.Lock()
	key := strings.ToLower(s.Name())
	if _, exists := r.byName[key]; exists {
		r.mu.Unlock()
		return fmt.Errorf("slot %q already loaded", s.Name())
	}
	if s.Port == 0 {
		s.Port = r.nextLabPortLocked()
	}
	s.External = fmt.Sprintf("http://127.0.0.1:%d", s.Port)
	u, _ := url.Parse(s.External)
	in := &Instance{Slot: s, Model: models.Model{ID: s.Alias, Name: s.Alias}, External: u, Port: s.Port, state: Starting, tail: newRing(300), docker: name}
	r.instances = append(r.instances, in)
	r.byName[key] = in
	r.mu.Unlock()

	// A leftover of the same name (a previous lab boot that was never unloaded) would make
	// `docker run` fail on the name; remove it first.
	_ = exec.Command("docker", "rm", "-f", name).Run()

	args := containerArgs(s, deviceIDs(s.GPUs), s.Port)
	cmd := exec.Command("docker", args...)
	cmd.Env = os.Environ()
	if b := in.Bearer(); b != "" {
		cmd.Env = append(cmd.Env, ct.KeyVar()+"="+b)
	}
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	r.log.Printf("[%s] docker %s", s.Name(), redactedArgs(args))
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(out.String())
		in.set(Crashed, "docker run failed: "+msg)
		r.log.Printf("[%s] docker run failed: %s", s.Name(), msg)
		return fmt.Errorf("docker run failed: %s", msg)
	}
	if in.isStopping() { // unloaded while `docker run` was still creating it: do not leave it running
		r.removeContainer(in, name)
		return nil
	}
	go r.tailContainer(in, name)
	go r.superviseContainer(in, name)
	return nil
}

// nextLabPortLocked hands out host ports for container slots from the lab's base (18100 unless
// configured), skipping ports a loaded slot already holds. Caller holds r.mu.
func (r *Rig) nextLabPortLocked() int {
	base := 18100
	if r.cfg != nil && r.cfg.Lab != nil && r.cfg.Lab.BasePort > 0 {
		base = r.cfg.Lab.BasePort
	}
	used := map[int]bool{}
	for _, in := range r.instances {
		used[in.Port] = true
	}
	p := base
	for used[p] {
		p++
	}
	return p
}

// redactedArgs renders the docker argv for the log with every -e value shortened to its name.
func redactedArgs(args []string) string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		if args[i] == "-e" && i+1 < len(args) {
			kv := args[i+1]
			if k, _, ok := strings.Cut(kv, "="); ok {
				out = append(out, "-e", k+"=…")
			} else {
				out = append(out, "-e", kv)
			}
			i++
			continue
		}
		out = append(out, args[i])
	}
	return strings.Join(out, " ")
}

// tailContainer streams `docker logs -f` into the slot's tail until the container is gone.
func (r *Rig) tailContainer(in *Instance, name string) {
	cmd := exec.Command("docker", "logs", "-f", "--tail", "200", name)
	pr, err := cmd.StdoutPipe()
	if err != nil {
		return
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		return
	}
	sc := bufio.NewScanner(pr)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r")
		if strings.Contains(line, "%|") { // progress bars redraw the same line; keep the log readable
			continue
		}
		in.tail.push(line)
		if in.isStopping() {
			break
		}
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
}

// superviseContainer polls the server's health and the container's running state. Healthy
// when the server answers 200; crashed (with the log's last lines) when the container exits.
// Nothing is restarted: for a lab slot the exit reason is the result.
func (r *Rig) superviseContainer(in *Instance, name string) {
	cl := &http.Client{Timeout: 3 * time.Second}
	target := in.External.String() + in.Slot.Container.HealthPath()
	last := State("")
	for {
		if in.isStopping() {
			return
		}
		running, exitCode, ierr := containerRunning(name)
		var st State
		var why string
		if ierr != nil {
			// The docker CLI did not answer (daemon busy, a timeout): unknown is not crashed.
			// Keep the last state and ask again on the next poll.
			st = last
			if st == "" {
				st = Starting
			}
		} else if !running {
			st, why = Crashed, fmt.Sprintf("container exited (%d): %s", exitCode, in.tail.last())
		} else {
			req, _ := http.NewRequest(http.MethodGet, target, nil)
			if b := in.Bearer(); b != "" {
				req.Header.Set("Authorization", "Bearer "+b)
			}
			resp, err := cl.Do(req)
			if err != nil {
				st = Starting
			} else {
				resp.Body.Close()
				if resp.StatusCode == 200 {
					st = Healthy
				} else {
					st = Starting
				}
			}
		}
		if st != last {
			in.set(st, why)
			switch st {
			case Healthy:
				r.log.Printf("[%s] container healthy (%s)", in.Slot.Name(), in.External)
				in.captureBaseline()
			case Crashed:
				r.log.Printf("[%s] container exited: %s", in.Slot.Name(), why)
			}
			last = st
		}
		if st == Crashed {
			return // the log stays readable in the slot until it is unloaded
		}
		for i := 0; i < 30; i++ { // 3 s between polls, but leave promptly on unload
			if in.isStopping() {
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}
}

// containerRunning asks docker whether the container is still up, and its exit code if not. A
// missing container reads as not running (exit -1); a CLI failure is an error, not a verdict.
func containerRunning(name string) (running bool, exitCode int, err error) {
	cmd := exec.Command("docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", name)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		if strings.Contains(stderr.String(), "No such object") {
			return false, -1, nil
		}
		return false, -1, fmt.Errorf("docker inspect: %v %s", err, strings.TrimSpace(stderr.String()))
	}
	fields := strings.Fields(string(out))
	if len(fields) < 2 {
		return false, -1, fmt.Errorf("docker inspect: unexpected output %q", strings.TrimSpace(string(out)))
	}
	code, _ := strconv.Atoi(fields[1])
	return fields[0] == "true", code, nil
}

// removeContainer stops and removes the slot's container (unload).
func (r *Rig) removeContainer(in *Instance, name string) {
	if out, err := exec.Command("docker", "rm", "-f", name).CombinedOutput(); err != nil {
		r.log.Printf("[%s] docker rm %s: %v %s", in.Slot.Name(), name, err, strings.TrimSpace(string(out)))
	}
}

// ContainerFor reports whether a slot is a container slot and its docker name.
func (in *Instance) ContainerFor() (string, bool) {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.docker, in.docker != ""
}

// Snapshot is the slot's status as the API reports it.
func (in *Instance) Snapshot() Status { return in.snapshot() }

// ServedName is the model name the upstream inside a container slot answers to when it is not
// the slot's alias ("" when they agree or for any other slot kind). The router rewrites the
// request's `model` to it.
func (in *Instance) ServedName() string {
	ct := in.Slot.Container
	if ct == nil || ct.Served == "" || ct.Served == in.Slot.Name() {
		return ""
	}
	return ct.Served
}

// Lab is the config's lab block (nil when absent).
func (r *Rig) Lab() *config.Lab {
	if r.cfg == nil {
		return nil
	}
	return r.cfg.Lab
}

// LabSlot builds the container slot the lab panel asked for, taking everything that touches
// the host (image, mounts, extra docker flags, the key variables, the served name) from the
// config's lab block and only the knobs from the request: alias, GPUs, host port, command,
// pre-command and environment. A client can choose what runs inside the image; it cannot
// choose what the container is given of the host.
func (r *Rig) LabSlot(req config.Slot) (config.Slot, error) {
	lab := r.Lab()
	if lab == nil {
		return config.Slot{}, fmt.Errorf("container slots need a lab block in the config")
	}
	if req.Container == nil {
		return config.Slot{}, fmt.Errorf("not a container slot")
	}
	env := map[string]string{}
	for k, v := range lab.Env {
		env[k] = v
	}
	for k, v := range req.Container.Env {
		env[k] = v
	}
	s := config.Slot{
		Alias: req.Alias, GPUs: req.GPUs, Port: req.Port,
		APIKeyEnv: lab.APIKeyEnv,
		Container: &config.Container{
			Image: lab.Image, Cmd: req.Container.Cmd, Pre: req.Container.Pre, Env: env,
			Mounts: lab.Mounts, Extra: lab.Extra, Port: lab.Port, KeyEnvIn: lab.KeyEnvIn, Served: lab.Served,
		},
	}
	return s, s.Container.Validate(s)
}

// AddContainerForTest registers a container slot that fronts an already-running upstream
// (nothing is launched, no docker): the router's served-name rewrite and the container kind
// can be exercised by tests without a daemon.
func (r *Rig) AddContainerForTest(s config.Slot, upstream string) error {
	s.External = upstream
	in, err := r.addExternal(s, Healthy)
	if err != nil {
		return err
	}
	in.mu.Lock()
	in.docker = containerName(s)
	in.mu.Unlock()
	return nil
}
