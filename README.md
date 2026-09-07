# llama-chip

A native local LLM rig manager for multi-GPU Windows boxes. It orchestrates
[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` instances with **per-GPU
pinning**, an optional **one-model-across-both-GPUs split**, an **OpenAI-compatible router**,
**auto-restart on crash**, and a **loader UI** — no Docker, no Python.

Built because a dockerized wrapper (FlexLLama) kept dying: its proxy stayed up while a model
runner crashed, and there was no auto-restart. llama-chip is a single Go binary that does the
same job natively, with the pieces that wrapper lacked.

## How it works

llama-chip doesn't build llama.cpp. It **reuses what LM Studio already downloaded** — LM Studio
does a good job of fetching the latest builds for every backend (CUDA, CUDA12, ROCm, Vulkan,
CPU) and the GGUF model library — and drives those binaries directly:

- **Backends** live in `~/.lmstudio/extensions/backends/` (`llama-chip backends` lists them,
  newest first, flagging which are actually downloaded vs stubs).
- **Models** live in `~/.lmstudio/models/` (`llama-chip models` lists the GGUFs).

You select a backend (the latest runnable CUDA12, say), define which models load on which
GPU(s), and llama-chip spawns + supervises the `llama-server` processes and fronts them on one
OpenAI endpoint.

## Status — all four phases working

**Phase 1 — discovery CLI**
```
llama-chip backends   # the llama.cpp builds LM Studio has (cuda12/cuda/rocm/vulkan/cpu), newest first
llama-chip models     # the GGUF models in the LM Studio library
llama-chip gpu        # live NVIDIA GPU state
```

**Phase 2 — the rig** (`llama-chip serve --config config.json`): one `llama-server` per slot,
`CUDA_VISIBLE_DEVICES` per-GPU pinning (+ `--split-mode` for one model across both cards),
health-check + **auto-restart on crash**, OpenAI router on `:8090` (drop-in — routes
`/v1/chat|completions|embeddings` by the `model` field, streams SSE).

**Phase 3 — loader UI + runtime control** at `/`: live GPU bars, loaded-model table, a load
form (model picker + GPU checkboxes + context + KV cache). Backed by a runtime API:
`POST /api/load`, `POST /api/unload`, `GET /api/models|backends|status|gpu`.

**Phase 4 — dynamic model management.** Named **profiles** (GPU layouts) you switch between:
```
POST /api/profile {"name":"diverse"}    # unload all, load that layout
GET  /api/profiles
POST /api/ensure  {"model":"...","gpus":[1],"ctx_size":8192}   # load-by-need; evicts overlapping-GPU slots
```
A profile like `diverse` = qwen on GPU0 + gemma on GPU1 (model diversity); `big-single` = one
model split across both cards. The substrate switches profiles (or calls `/api/ensure`) to load
what a work phase needs — proven: `diverse` loaded two models one-per-GPU, then `single` swapped
to one and freed GPU1.

See `config.example.json` for slots + profiles.

**Phase 5 — runtime management via LM Studio's `lms`.** llama-chip drives LM Studio's own
official downloader instead of reinventing one:
```
llama-chip runtime ls          # installed engines (passthrough to `lms runtime ls`)
llama-chip runtime get -l llama.cpp:cuda12   # list available from LM Studio's source
llama-chip pull [llama.cpp:cuda12]           # download the latest runtime (lms runtime get -y)
```
llama-chip auto-uses the **latest installed runnable** backend (detected by the presence of
`ggml-base.dll` — newer runtimes ship a ~20KB `llama-server.exe` shim that loads
`llama-server-impl.dll`, so exe *size* isn't the signal).

## Vision / multimodal (mmproj)

llama.cpp serves vision/audio models by loading the model GGUF **plus an `--mmproj`
projector** (the multimodal encoder). llama-chip wires it in automatically: when a
model's directory holds a co-located `mmproj-*.gguf` (LM Studio downloads the projector
alongside a vision model), the slot launches with `--mmproj` and the model accepts
OpenAI `image_url` content — **no config needed**. `llama-chip models` flags
vision-capable models (`supports_vision`), and the `:8090` router forwards multimodal
content through untouched.

- Auto-detect is on by default. Set `"no_mmproj": true` on a slot to load text-only,
  or `"mmproj": "<path>"` to point at a specific projector.
- Proven local: the gemma-4 family (12b/26b-a4b) and qwen3.6-35b read images on the
  4090s — read a screenshot's on-screen text back correctly, $0 and private.
- The projector + image tokens add ~1–2 GB VRAM, so trim `ctx_size` if a tight
  multi-model profile OOMs.

Send an image the standard OpenAI way (the router proxies it to the backend as-is):
```json
{"model":"gemma-4-26b-a4b","messages":[{"role":"user","content":[
  {"type":"text","text":"what is in this image?"},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}
```

## Wiring into pg-ai-stewards

llama-chip is the substrate's local inference rig (it replaced the dockerized FlexLLama). The
substrate dials it as the `flexllama` provider — the name is kept so the substrate's
`model_capability` rows and `ingest`/`reason`/`critic` role-aliases need **zero** change:

1. `oss/.env` (the substrate): point the provider at the host —
   ```
   STEWARDS_PROVIDER_FLEXLLAMA_BASE_URL=http://host.docker.internal:8090/v1
   STEWARDS_PROVIDER_FLEXLLAMA_KIND=openai
   STEWARDS_PROVIDER_FLEXLLAMA_DEFAULT_MODEL=qwen3.6-27b
   ```
   then recreate the `pg` service (`docker compose up -d --no-deps --force-recreate pg`,
   data-safe) so the bgworker picks up the provider.
2. Serve the models under the **names the substrate dispatches** — the `dance` profile aliases
   the GGUFs as `qwen3.6-27b` (GPU0) and `gemma-4-26b-a4b` (GPU1). Apply it after `serve`.
3. The substrate (in its container) reaches the rig over `host.docker.internal:8090`, the same
   path its `ollama` embeddings provider already uses.

## External slots — front a server you did not launch (vLLM, say)

A slot can point at an OpenAI-compatible server that llama-chip does **not** start or stop: a vLLM
container, an SGLang box, anything that speaks `/v1/*` and answers `GET /health`. The rig only
polls its health and the router proxies to it, so it shows up in `/v1/models`, in profiles, in
`/api/status` and in the federation roster exactly like a llama-server slot. The alias is required
and must be the model name the upstream serves, because the request body is forwarded as it came.

```json
{ "alias": "qwen3.8-27b", "external": "http://127.0.0.1:18020", "api_key_env": "QWEN_PROD_KEY", "gpus": [0] }
```

- `external` is the server **root** (no `/v1`); the router appends the request path.
- `api_key_env` names an environment variable holding the upstream's bearer (`api_key` is the
  inline alternative; the env var wins when both are set). It is sent as `Authorization` on every
  proxied request and on the health poll.
- `gpus` is declarative here: it tells `ensure` which cards the upstream occupies so an overlapping
  load evicts it from the roster (it does not stop the upstream), and it tells the UI where it lives.
- There is no restart: if the upstream goes away the slot reads `crashed` with the reason until it
  comes back, and `unload` only stops the poller.

## Federation — pool GPUs across machines

llama-chip can pool the GPUs of several machines behind one OpenAI endpoint. Each node serves
its own GPUs locally; a request for a model it doesn't serve is routed to a peer that does, over
a private mesh (NetBird / Tailscale / LAN). **Local-first** — a node always works on its own GPUs
with zero dependency on peers; a peer's models appear only while the mesh reaches it and vanish
the moment it drops. Add a `federation` block to `config.json`:

```json
"federation": {
  "node_name": "home-gpu",
  "peers": [ { "name": "laptop", "url": "http://100.84.0.3:8090" } ]
}
```

Full design, the config reference, the NetBird mesh runbook, and the two-substrate topology are
in **[docs/federation.md](docs/federation.md)**.

For a **shared pool across people/LANs**, run the **hub** (`cmd/llama-hub`, deploy at
`llama.example.com`): an admin mints node/user join tokens and watches a live roster of who has
what model + free GPU; nodes set `federation.hub_url` + a `hub_token` instead of hand-listing
peers, and still route peer-to-peer over the mesh (the hub is control-plane only). See
**[docs/hub.md](docs/hub.md)**.

## Roadmap
- VRAM-aware `ensure` eviction (estimate from file size + ctx) instead of GPU-overlap only.
- Optional: respect lms's *selected* runtime, or pin a version in config (`backend: "cuda12@2.22.0"`).
- A Windows scheduled-task install (`--install-startup`) so the rig is up at logon for the substrate.
- Refine the guess-context KV math for sliding-window (gemma) / compressed-KV architectures.

## Build

```
go build ./cmd/llama-chip
```

Requires Go 1.22+. Windows + NVIDIA today (the GPU helper shells `nvidia-smi`); the backend/model
discovery is cross-platform.
