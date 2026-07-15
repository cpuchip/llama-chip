# llama-chip on NOCIX (CPU-only, Dokploy)

A GPU-less llama-chip node that keeps the federation alive with a small model at a few tok/s on
CPU. Packaged as a docker-compose app for Dokploy on the NOCIX VPS. The image is tiny; the CPU
llama.cpp backend and the model GGUF are pulled into the `/data` volume on **first boot** (both
idempotent, so restarts are fast and don't depend on GitHub/HuggingFace being reachable).

## Deploy

1. Point a Dokploy Compose app at this directory (`deploy/nocix/`), or copy `docker-compose.yml`.
2. Set the environment (below). **Set `FED_TOKEN`** — the node is exposed on a public HTTPS domain
   and this bearer is what gates it. Set `ADVERTISE` to that public URL and (if joining fermion's
   hub) `HUB_URL` + `HUB_TOKEN`.
3. Dokploy maps the domain to container port `8090`. First `up` downloads the model (minutes on a
   shared link); `GET /health` returns 200 once the model is loaded.

## Environment

| Var | Default | Meaning |
|-----|---------|---------|
| `FED_TOKEN` | *(empty)* | Bearer that gates this node's API. **Required on a public deploy** (empty ⇒ ungated, logged loud). |
| `ADVERTISE` | *(empty)* | Public URL fermion/the hub reach this node at, e.g. `https://llama-nocix.cpuchip.net`. |
| `HUB_URL` | *(empty)* | fermion's hub, e.g. `https://llama.cpuchip.net`. Empty ⇒ standalone gated node (no roster). |
| `HUB_TOKEN` | *(empty)* | This node's join token for the hub. |
| `NODE_NAME` | `nocix` | Federation name (how fermion addresses it — must match fermion's `peer_tokens` key). |
| `MODEL_REPO` | `lmstudio-community/gemma-4-E2B-it-GGUF` | HF repo (also the on-disk `publisher/repo` dir). |
| `MODEL_FILE` | `gemma-4-E2B-it-Q4_K_M.gguf` | GGUF filename (also the slot handle's tail). |
| `MODEL_ALIAS` | `gemma-4-e2b` | The name clients use in the OpenAI `model` field. |
| `MODEL_URL` | derived from `MODEL_REPO`/`MODEL_FILE` | Override to serve any GGUF. |
| `CTX_SIZE` / `PARALLEL` / `KV_CACHE` | `8192` / `1` / `f16` | Slot shape. |
| `BACKEND_REF` | `latest` | ggml-org release ref (`latest` or a build tag like `b10012`). |
| `HOST_PORT` | `8090` | Published host port (Dokploy maps the domain to it). |
| `MEM_LIMIT` / `CPUS` | `6g` / `4` | Resource caps (shared VPS). |

The default model is **gemma-4-E2B-it Q4_K_M** (verified real GGUF, ~3.4 GB):
`https://huggingface.co/lmstudio-community/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf`

## First-boot behavior (entrypoint)

1. **Backend** — if no `llama-server` under `/data/backends`, run `llama-chip pull-ggml <BACKEND_REF>`,
   which fetches the ggml-org **ubuntu-x64 CPU** build (`llama-*-bin-ubuntu-x64.tar.gz`) and extracts
   it (symlinks preserved) into `/data/backends/ggml-<tag>/`.
2. **Model** — if `/data/models/<MODEL_REPO>/<MODEL_FILE>` is absent, download it (resumable) from
   `MODEL_URL`.
3. **Config** — render `/data/config.json` with one CPU slot (`"gpus": []`) and a federation block.
4. `exec llama-chip serve`.

## How fermion routes to this node (the peer token)

fermion runs **keyless** for mesh clients but this node is **bearer-gated**, so fermion must attach
*this node's* token when it proxies a chat here. That's a per-peer outbound token on **fermion's**
config, keyed by this node's `NODE_NAME`:

```jsonc
// fermion's config.json — federation block
"federation": {
  "node_name": "fermion",
  "hub_url": "https://llama.cpuchip.net",
  "hub_token": "…",
  "peer_tokens": {
    "nocix": "THE_SAME_VALUE_AS_FED_TOKEN_ON_THIS_NODE"
  }
}
```

**Outbound bearer precedence** (fermion → peer): `peer_tokens[peer]` **>** fermion's own
`federation.token` **>** the caller's incoming `Authorization`. So the per-peer token wins for
`nocix` while every keyless mesh peer keeps legacy passthrough behavior. The same token is used both
to discover the peer (poll, in static mode) and to proxy chats to it (and roster/hub-managed mode,
which is fermion's posture, skips the poll entirely — the hub supplies `nocix`'s models + address).

No fermion restart is needed beyond loading the updated config. Verify end-to-end with:

```bash
curl -s https://<fermion>/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"gemma-4-e2b","messages":[{"role":"user","content":"hello"}]}'
```

## Local proof (before deploying)

Build + run the image on any Linux-capable Docker host, CPU-only, and hit it directly — see the
image README's "oracle" transcript. The NOCIX deploy should never be the first time the image runs.
