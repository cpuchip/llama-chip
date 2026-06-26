# llama-hub — the GPU-pool coordinator (control plane). Pure-Go, no CGO, tiny image.
# Build context MUST be the repo root (it needs the whole module): in Dokploy set
#   Build Path / context = .    Dockerfile = cmd/llama-hub/Dockerfile
FROM golang:1.23-alpine AS build
WORKDIR /src
COPY . .
# go.work is gitignored, but force module mode in case the context carries one.
RUN GOWORK=off CGO_ENABLED=0 go build -trimpath -o /llama-hub ./cmd/llama-hub

FROM alpine:3.20
RUN apk add --no-cache ca-certificates && adduser -D -u 10001 hub
WORKDIR /app
COPY --from=build /llama-hub /usr/local/bin/llama-hub
# tokens persist here — mount a volume so they survive redeploys
ENV LLAMA_HUB_LISTEN=:8088 \
    LLAMA_HUB_TOKENS=/data/tokens.json
RUN mkdir -p /data && chown hub /data
USER hub
VOLUME /data
EXPOSE 8088
# set LLAMA_HUB_ADMIN_KEY in the Dokploy env to seed your admin login on first boot
ENTRYPOINT ["llama-hub"]
