# syntax=docker/dockerfile:1
#
# llama-chip — CPU-only Linux image for a Dokploy docker-compose deploy on a shared VPS.
# Multi-stage: a static Go build of ./cmd/llama-chip, then a slim ubuntu runtime that carries
# only what the pulled ggml-org CPU llama-server needs (libgomp1) plus TLS + curl for first-boot
# downloads. The llama.cpp backend and the model GGUF are NOT baked in — the entrypoint pulls
# them into the /data volume on first boot (idempotent), so the image stays small and the heavy
# bits persist across restarts/redeploys.
#
# Build context is the REPO ROOT (see docker-compose.yml: context ../..).

# ---- build ----
FROM golang:1.24-bookworm AS build
WORKDIR /src
# Only the Go sources are needed (stdlib-only module — no go.sum, no external deps). Copying
# selectively also guarantees the live root config.json is never pulled into the image.
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
# GOWORK=off: build this module standalone (the repo lives inside a go.work workspace on dev boxes;
# the workspace file is not present in the container, but be explicit). CGO off → static binary.
ENV GOWORK=off CGO_ENABLED=0 GOFLAGS=-trimpath
RUN go build -ldflags="-s -w" -o /out/llama-chip ./cmd/llama-chip

# ---- runtime ----
FROM ubuntu:24.04
# libgomp1: the ggml CPU kernels (libggml-cpu-*.so) link libgomp.so.1 (OpenMP).
# ca-certificates + curl: first-boot HTTPS downloads (backend tarball, model GGUF) + healthcheck.
# libssl3/libcrypto/libstdc++ are already in the ubuntu:24.04 base (llama-server links them).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/llama-chip /usr/local/bin/llama-chip
COPY deploy/nocix/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
# Discovery dirs point INTO the persistent volume so the pulled backend + downloaded model survive
# restarts (llama-chip reads these env overrides before any home-dir fallback).
ENV LLAMACHIP_GGML_DIR=/data/backends \
    LLAMACHIP_MODELS_DIR=/data/models \
    HOME=/data
WORKDIR /data
EXPOSE 8090
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
