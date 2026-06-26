# Joining a shared GPU pool

This guide is for joining an existing llama-chip **pool** — a group of machines that share GPUs
behind one hub. You'll need three things from whoever runs the pool:

1. **The hub URL** — e.g. `https://your-hub.example` (replace `<HUB_URL>` below with it).
2. **A token** — a *node* token if you're contributing GPUs, or a *user* token if you only want to
   use the pool. Mint these in the hub UI (admin) → "Access tokens".
3. **Mesh access** — install the mesh client (NetBird/Tailscale) and join the pool's network, so
   your machine gets a private mesh IP that peers can reach. Ask the pool admin for the enrollment
   command/URL.

> Throughout, `<HUB_URL>` is the pool's hub (the admin gives it to you) and `<mesh-ip>` is your
> machine's address on the mesh (run `netbird status` / `tailscale ip` to find it).

---

## A. Contribute GPUs (a "node")

Use a **node** token. Your GPUs join the pool and you can use everyone else's.

1. Build/install llama-chip and confirm it runs standalone (`llama-chip serve` → the loader UI on `:8090`).
2. Join the mesh, note your `<mesh-ip>`.
3. Add a `federation` block to your `config.json`, and set `listen` so peers can reach you:

```json
{
  "backend": "cuda12",
  "listen": "0.0.0.0:8090",
  "slots": [ { "model": "…a model that fits your GPU…", "alias": "my-model", "gpus": [0] } ],
  "federation": {
    "node_name": "my-box",
    "advertise": "http://<mesh-ip>:8090",
    "hub_url": "<HUB_URL>",
    "hub_token": "lck_…your node token…",
    "poll_interval_sec": 5
  }
}
```

4. `llama-chip serve`. You'll appear in the hub roster within a few seconds. Your `:8090` now serves
   your local models **and** routes any model you don't have to whoever in the pool does — over the
   mesh, peer-to-peer. If the hub or a peer is unreachable, you keep serving your own GPUs (local-first).

## B. Use the pool without GPUs (a "client")

Use a **user** token. You run llama-chip with **no slots** — it becomes a gateway to the whole pool.

```json
{
  "listen": "0.0.0.0:8090",
  "slots": [],
  "federation": {
    "node_name": "my-laptop",
    "advertise": "http://<mesh-ip>:8090",
    "hub_url": "<HUB_URL>",
    "hub_token": "lck_…your user token…"
  }
}
```

`llama-chip serve`, then point any OpenAI-compatible client at `http://localhost:8090/v1`. Requests
route to whichever pooled node serves the model you ask for. (`GET /v1/models` lists everything
currently reachable.)

## C. Let an AI agent use the pool

An AI agent (a coding agent, an autonomous worker, a substrate like pg-ai-stewards) uses the pool
exactly like any OpenAI client — point its provider at a llama-chip endpoint that's joined the pool:

- **base URL:** `http://localhost:8090/v1` (the agent's *local* llama-chip, joined per A or B above —
  from inside a container use `http://host.docker.internal:8090/v1`).
- **model:** the handle the pool serves, e.g. `qwen3.6-35b-a3b`. The local llama-chip routes it to a
  peer that has it loaded.
- **api key:** none needed at the OpenAI layer (the federation/mesh is the boundary). If the pool
  enables a node bearer token, set it as the provider's key.

The agent never talks to the hub directly — it talks to its local llama-chip, which handles
discovery (via the hub) and routing (over the mesh). So an agent gains the whole pool's models by
changing one base-URL.

---

## What's mine to fill in

`<HUB_URL>` and `<mesh-ip>` are placeholders on purpose — drop in the pool's real hub URL and your
own mesh address. When you're signed into the hub UI, the **"How to join"** panel shows this same
snippet with the hub's real URL already filled in (copy-paste, then add your token + mesh IP).

Running the pool yourself? See [hub.md](hub.md) (deploy the hub) and [federation.md](federation.md)
(the mesh).
