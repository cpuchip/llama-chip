# The lab: container slots with editable knobs

A server like vLLM has a dozen launch-time settings (context length, GPU memory fraction, the
speculative draft length, the KV cache tier, an experiment's one-line edit) and no way to feel
them except a shell script per attempt. The lab panel turns each attempt into a **container
slot**: the rig runs the image with `docker run`, the knobs are the container's environment,
the boot log streams into the panel, and once the server answers `/health` the slot is a model
in the chat test like any other, with tok/s on every reply.

## Config

Add a `lab` block; without it the panel stays hidden.

```json
"lab": {
  "image": "your-inference-image:tag",
  "mounts": ["D:\\models:/app/models", "cache-volume:/cache"],
  "extra_args": ["--ipc", "host", "--shm-size", "64m"],
  "api_key_env": "MY_SERVER_KEY",
  "key_env_in": "VLLM_API_KEY",
  "port": 18020,
  "base_port": 18100,
  "env": {"HOME": "/cache", "PORT": "18020"},
  "presets": [
    {"name": "baseline", "cmd": "bash docker/entrypoint.sh single",
     "env": {"MAX_LEN": "100000", "GPU_UTIL": "0.88"}, "note": "what production runs"},
    {"name": "an experiment", "cmd": "bash docker/entrypoint.sh single",
     "pre": "sed -i 's/^SOME_CONSTANT = 16$/SOME_CONSTANT = 32/' /app/some/file.py",
     "env": {"MAX_LEN": "100000", "GPU_UTIL": "0.88"}, "note": "the same, with one line changed"}
  ]
}
```

| field | meaning |
|---|---|
| `image` | the image every lab launch runs |
| `mounts` | `docker -v` entries: model directory, a cache volume |
| `extra_args` | extra `docker run` flags |
| `api_key_env` | host environment variable holding the bearer the server expects; the router sends it on every proxied request |
| `key_env_in` | the container variable that receives that bearer (default `VLLM_API_KEY`) |
| `port` | the server's port inside the container (default 18020) |
| `served` | the model name the image's server answers to (a vLLM launcher hardcodes one); requests addressed to the slot's alias have `model` rewritten to it, so several lab slots of one image can run under different aliases |
| `base_port` | the first host port the panel proposes (default 18100); each launch takes the next |
| `env` | environment every launch gets, under the preset's and the form's |
| `presets` | named starting points: `cmd` (run inside the image), `env` (the knobs), optional `pre` (shell run first, inside the container) and `note` |

The bearer never appears on the docker command line: the key variable is passed by name and
docker copies the value from the rig's environment. The rig log prints the launch with every
`-e` value shortened to its name. That protection covers the bearer only: everything in `env`
(the block's and the form's) is an ordinary container environment, visible in a process
listing and returned by `GET /api/lab`, so a token does not belong there.

A launch through the API is accepted only from this machine (loopback), and it takes what the
container is given of the host (image, mounts, docker flags, the key variables, the served
name) from the lab block, never from the request body: a client chooses what runs inside the
image, not what the container can reach.

A container slot can also be declared directly in `slots` or in a profile:

```json
{"alias": "qwen-lab", "gpus": [1], "port": 18100, "api_key_env": "MY_SERVER_KEY",
 "container": {"image": "your-inference-image:tag", "cmd": "bash docker/entrypoint.sh single",
               "env": {"MAX_LEN": "100000"}, "mounts": ["D:\\models:/app/models"]}}
```

## Using it

1. Pick a preset, a GPU and an alias. Edit the knobs (one `KEY=VALUE` per line) and, if the
   preset has one, the pre-command.
2. **Launch.** The slot appears in the loaded table as kind `container`, state `starting`, and
   the boot log follows in the panel. A boot that dies shows `crashed` with the server's own
   reason (a KV cache that does not fit at that context, say) as the last line.
3. When it is `healthy`, choose the alias in the chat test and talk to it. The reply meta line
   carries the generation tok/s and the prompt tok/s; the live-activity panel shows the same
   per slot. To feel a prefill setting, paste a long document.
4. **Stop & remove** (or unload from the table) removes the container. Nothing is restarted
   automatically: an exited lab container is a result, not a fault.

## API

- `POST /api/load` with a slot carrying `container` launches it (the panel builds this body).
- `GET /api/logs?name=<slot>` returns the slot's recent output and state.
- `GET /api/lab` returns the config's lab block (404 when there is none).
- `POST /api/unload {"name": ...}` removes the container.

GPU pinning uses the card's UUID (`--gpus device=GPU-...` plus `NVIDIA_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES` inside), the same identity the rest of the rig pins by, so a container's
own numbering never picks the wrong card.
