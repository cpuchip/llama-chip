#!/usr/bin/env bash
# entrypoint.sh — idempotent first-boot for the CPU-only llama-chip node.
#
# On every boot (all steps skip cleanly if already done, so a restart is fast and does NOT depend
# on GitHub/HuggingFace being reachable):
#   1. pull the CPU llama.cpp backend into /data/backends if missing
#   2. download the model GGUF into /data/models/<repo>/<file> if missing (resumable)
#   3. render /data/config.json from env
#   4. exec `llama-chip serve`
set -euo pipefail

BACKENDS_DIR="${LLAMACHIP_GGML_DIR:-/data/backends}"
MODELS_DIR="${LLAMACHIP_MODELS_DIR:-/data/models}"
CONFIG="/data/config.json"

# --- tunables (env) ---
NODE_NAME="${NODE_NAME:-nocix}"
ADVERTISE="${ADVERTISE:-}"           # PUBLIC url fermion/the hub reach this node at (e.g. https://llama-nocix.cpuchip.net)
HUB_URL="${HUB_URL:-}"               # fermion's hub (e.g. https://llama.cpuchip.net); empty = standalone gated node
HUB_TOKEN="${HUB_TOKEN:-}"           # this node's join token for the hub
FED_TOKEN="${FED_TOKEN:-}"           # bearer that gates this node's PUBLIC surface — SET THIS on a public deploy
LISTEN="${LISTEN:-0.0.0.0:8090}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-5}"

# slot shape (CPU): a few tok/s of a small model is the goal
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL="${PARALLEL:-1}"
KV_CACHE="${KV_CACHE:-f16}"
BACKEND_REF="${BACKEND_REF:-latest}" # ggml-org release ref: "latest" or a build tag like "b10012"

# model (env) — default: gemma-4-E2B instruct Q4_K_M (verified real HF GGUF, ~3.4 GB).
# models.Discover expects a publisher/repo/file.gguf layout, so REPO+FILE define the on-disk path
# and the slot handle; URL is what we actually download.
MODEL_REPO="${MODEL_REPO:-lmstudio-community/gemma-4-E2B-it-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-E2B-it-Q4_K_M.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma-4-e2b}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}}"

mkdir -p "$BACKENDS_DIR" "$MODELS_DIR"

if [ -z "$FED_TOKEN" ]; then
  echo "[entrypoint] WARNING: FED_TOKEN is empty — this node's API is UNGATED. Set FED_TOKEN on any public deploy." >&2
fi

# --- 1) backend: pull the CPU llama-server if not already present (idempotent, offline-safe) ---
if [ -n "$(find "$BACKENDS_DIR" -type f -name llama-server 2>/dev/null | head -n1)" ]; then
  echo "[entrypoint] CPU backend already present in $BACKENDS_DIR — skipping pull"
else
  echo "[entrypoint] pulling CPU llama.cpp backend ($BACKEND_REF) from ggml-org releases ..."
  llama-chip pull-ggml "$BACKEND_REF"
fi

# --- 2) model: download the GGUF if not already present (idempotent, resumable) ---
MODEL_PATH="${MODELS_DIR}/${MODEL_REPO}/${MODEL_FILE}"
if [ -s "$MODEL_PATH" ]; then
  echo "[entrypoint] model already present at $MODEL_PATH — skipping download"
else
  echo "[entrypoint] downloading model → $MODEL_PATH"
  mkdir -p "$(dirname "$MODEL_PATH")"
  # resumable (-C -), retry transient failures, follow redirects (HF → CDN); fail on HTTP errors.
  curl -fL --retry 5 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
       -C - -o "${MODEL_PATH}.part" "$MODEL_URL"
  mv -f "${MODEL_PATH}.part" "$MODEL_PATH"
  echo "[entrypoint] model download complete ($(du -h "$MODEL_PATH" | cut -f1))"
fi

# --- 3) render config.json from env ---
# A single CPU slot ("gpus": [] → CUDA_VISIBLE_DEVICES=""). The federation block is always written:
# with FED_TOKEN set the node is bearer-gated even when standalone; with HUB_URL set it also joins
# fermion's roster so fermion can discover + route to it (attaching this node's token via its
# federation.peer_tokens — see README).
cat > "$CONFIG" <<EOF
{
  "backend": "ggml@${BACKEND_REF}",
  "listen": "${LISTEN}",
  "slots": [
    {
      "model": "${MODEL_REPO}/${MODEL_FILE}",
      "alias": "${MODEL_ALIAS}",
      "gpus": [],
      "ctx_size": ${CTX_SIZE},
      "parallel": ${PARALLEL},
      "kv_cache": "${KV_CACHE}",
      "no_mmproj": true
    }
  ],
  "federation": {
    "node_name": "${NODE_NAME}",
    "advertise": "${ADVERTISE}",
    "token": "${FED_TOKEN}",
    "hub_url": "${HUB_URL}",
    "hub_token": "${HUB_TOKEN}",
    "poll_interval_sec": ${POLL_INTERVAL_SEC}
  }
}
EOF

echo "[entrypoint] config.json:"
cat "$CONFIG"
echo "[entrypoint] starting llama-chip serve on ${LISTEN} ..."
exec llama-chip serve -config "$CONFIG"
