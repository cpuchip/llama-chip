# The hub — a shared GPU pool at `llama.example.com`

The hub is the **control plane** for a federated GPU pool: a small service where an admin mints
join tokens and watches a live roster of which nodes are online, what models each serves, and
their free GPU. A group of machines — yours and your co-workers', across different LANs — register
with it and discover each other, then **route to each other peer-to-peer over a mesh**. The hub
never carries inference traffic; it only says who's in the group and where to reach them.

```
                         ┌───────────────────────────────┐
   admin (browser) ────► │  llama.example.com  (the hub)  │   control plane only
   mints tokens, sees    │  • tokens (admin/node/user)    │   (registry + auth + UI)
   the roster            │  • live roster + free GPU      │
                         └───────────────────────────────┘
                              ▲ register/heartbeat ▲           (no model traffic here)
                  ┌───────────┘                    └───────────┐
        ┌─────────┴─────────┐                        ┌─────────┴─────────┐
        │ node: home-gpu  │ ◄═══ inference over ══► │ node: peer-laptop │  data plane =
        │ llama-chip :8090  │     the NetBird mesh     │ llama-chip :8090  │  NetBird mesh (p2p)
        └───────────────────┘                          └───────────────────┘
```

Two layers, kept separate on purpose:
- **Control plane (this hub):** tokens, roster, UI. Small, public, hosted.
- **Data plane (NetBird mesh):** the actual model bytes, peer-to-peer, WireGuard-encrypted, never
  through the hub. See [federation.md](federation.md) for the mesh runbook.

## Tokens (auth)

Three kinds, all bearer tokens (the right primitive — GPU nodes are headless and can't do an OAuth
flow):

| kind | can |
|---|---|
| **admin** | mint/revoke tokens, see everything, manage via the UI |
| **node** | register itself + read the roster (a GPU contributor) |
| **user** | read the roster / route through the pool (a client, no GPUs of its own) |

The **admin key** is seeded once from the `LLAMA_HUB_ADMIN_KEY` env var — that's the key you log
into the UI with, and use to mint everyone else's tokens. A token's plaintext is shown exactly once
(at mint); the hub stores only its SHA-256 hash. Human login via Google/email-password is a later
add — tokens cover everything for now.

## Deploy to `llama.example.com` (Dokploy)

1. **New Compose app** in Dokploy, repo `cpuchip/llama-chip`, compose file `cmd/llama-hub/docker-compose.yml`
   (build context is the repo root — the Dockerfile needs the whole module).
2. **Env (secret):** `LLAMA_HUB_ADMIN_KEY=<a long random string>`. This is your admin login + the
   master mint key. Keep it secret; rotate by changing it and redeploying.
3. **Domain:** add `llama.example.com` → container port **8088** in the Dokploy UI (TLS via the
   panel). Point `llama.example.com` DNS at your VPS.
4. **Volume:** `hub-data:/data` persists `tokens.json` across redeploys (already in the compose).
5. Deploy. Visit `https://llama.example.com`, paste your admin key, and you're in.

> Health check: `GET /healthz` (open). It's the deploy oracle.

## Connecting a node (yours or a co-worker's)

On the machine with GPUs, add a `federation` block with `hub_url` + a node token to `config.json`.
`<HUB_URL>` is your hub (the live UI's "How to join" panel fills it in for you), `<mesh-ip>` is the
machine's mesh address:

```json
{
  "backend": "cuda12",
  "listen": "0.0.0.0:8090",
  "slots": [ { "model": "…fits this box…", "alias": "my-model", "gpus": [0] } ],
  "federation": {
    "node_name": "peer-laptop",
    "advertise": "http://<mesh-ip>:8090",   // this node's MESH address (peers route here)
    "hub_url": "<HUB_URL>",
    "hub_token": "lck_…",                     // a node token minted in the hub UI
    "poll_interval_sec": 5
  }
}
```

Then `llama-chip serve`. The node heartbeats its models + GPU to the hub and pulls the roster; it
routes any model it doesn't serve locally to whichever peer does — over the mesh. Local-first as
always: if the hub or a peer is unreachable, the node keeps serving its own GPUs.

**The full join guide (humans contributing GPUs, clients with none, and AI agents) is in
[joining.md](joining.md)** — generic, with placeholders, so it's safe to hand to anyone joining.

### A co-worker joining, end to end

1. They install **NetBird** and join your network (`netbird up --management-url …`) → they get a
   `100.x` mesh IP. (See [federation.md](federation.md) for the NetBird control-plane setup.)
2. You mint them a **node token** in the hub UI (label it, e.g. "my laptop").
3. They install llama-chip, drop in the `federation` block above with `advertise` = their mesh IP
   and `hub_token` = the token you gave them, and run `serve`.
4. They appear in the roster; their models + GPU are now part of the pool, and they can use yours.

Revoke their token in the UI to remove them.

## Privacy: keep private work on trusted nodes

A shared pool means a private intent could otherwise route to a co-worker's GPU. The
token model reserves a `scope` field for pinning private intents to a trusted node set; **P1 does
not enforce it yet**, so for now: keep private/file_private substrate work on a substrate whose
`flexllama` provider points at a hub group containing **only your own nodes** (a separate group /
hub, or static peers), not the shared co-worker pool. Enforced scoping is the next addition.

## API reference

| endpoint | who | does |
|---|---|---|
| `GET /healthz` | open | liveness |
| `GET /api/me` | any token | your token's kind + label |
| `POST /api/register` | node/admin | upsert this node `{name, mesh_addr, models, gpus}` → returns the roster |
| `GET /api/roster` | any token | the online nodes (TTL-pruned) |
| `GET /api/tokens` | admin | list tokens (hashes never exposed) |
| `POST /api/tokens` | admin | mint `{kind, label}` → returns the plaintext **once** |
| `POST /api/tokens/revoke` | admin | `{id}` → revoke |
| `GET /` | open page | the admin/roster UI (prompts for a key, kept in the browser) |

## Deferred (not in P1)

- **Enforced private-intent scoping** (the `scope` field) — pin private models to trusted nodes.
- **Human auth** — Google / email-password login for the UI (tokens cover machines today).
- **Relay fallback** — for a node that can't join the mesh (chosen data plane is NetBird mesh).
- **Federated placement** — ask a node with free VRAM to load a model on demand.
