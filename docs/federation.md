# Federation — pool GPUs across machines behind one endpoint

llama-chip can pool the GPUs of several machines into one logical rig. Each node keeps running
its own GPUs locally; when a node is asked for a model it doesn't serve itself, it routes the
request to a peer that does — over a private mesh (NetBird / Tailscale / LAN). One OpenAI
endpoint, the whole pool behind it. This is the same trick LM Studio does over Tailscale, built
into llama-chip.

## The design: local-first, no head/worker hierarchy

Every node runs the same llama-chip with a `federation` block listing the peers it can reach.
There is no central "head" that everything depends on:

- A node **always serves its own GPUs locally**, with zero dependency on any peer.
- A peer's models become reachable **only while the mesh reaches that peer**, and vanish the
  instant it drops (the poller marks it offline and evicts its routes).
- So a roaming laptop on its own keeps working on its own GPU; plugged into the mesh it sees the
  whole pool; the home box powering off just removes the big models from the laptop's view — it
  doesn't break anything.

This is exactly the property a mobile work laptop + an intermittently-on home box need. No single
point of failure.

### How routing works

```
client → POST /v1/chat/completions {"model":"qwen3.6-35b-a3b", …}
         │
         ▼
   this node's router
     1. local slot serves "qwen3.6-35b-a3b"?  → reverse-proxy to 127.0.0.1:<port>   (unchanged)
     2. else a reachable PEER serves it?        → reverse-proxy to the peer over the mesh
     3. else                                    → 404
```

Discovery is pull-only gossip: each node polls every peer's `/api/fed/local` (a **local-only**
model list) on an interval and keeps a live `model → peer` map. `/api/fed/local` is deliberately
local-only so that two mutually-peered nodes can't loop. The aggregated view (local + remote) is
exposed to *clients* via `/v1/models`; peers never poll that, so the aggregation never feeds back.

## Config block

Add a `federation` block to `config.json` (omit it, or leave `peers: []`, to run standalone):

```json
{
  "backend": "cuda12",
  "listen": "0.0.0.0:8090",
  "federation": {
    "node_name": "home-gpu",
    "advertise": "http://100.84.0.2:8090",
    "token": "",
    "poll_interval_sec": 5,
    "peers": [
      { "name": "laptop", "url": "http://100.84.0.3:8090" }
    ]
  },
  "profiles": { "...": "..." }
}
```

| field | meaning |
|---|---|
| `node_name` | this node's name (shown in `/api/status` + gossiped to peers) |
| `advertise` | the URL peers use to reach this node (informational; for your own clarity) |
| `token` | optional bearer token. **When set**, non-loopback callers must send `Authorization: Bearer <token>`. Loopback (the local substrate, a local browser) is always exempt, and `/health` is always open. Leave empty to rely on the mesh as the boundary. |
| `poll_interval_sec` | how often to refresh each peer (default 5) |
| `peers[]` | `{ name, url }` per reachable node. `url` is a base URL — a mesh IP, e.g. `http://100.84.0.3:8090`. |
| `hub_url` | optional coordinator (e.g. `https://llama.example.com`) that supplies the roster + issues join tokens, **instead of** static `peers`. See [hub.md](hub.md). |
| `hub_token` | this node's join token, minted in the hub. |

**Two discovery modes:** hand-list `peers` (LAN / small fixed set), **or** set `hub_url` + `hub_token`
and let the hub supply the live roster (a shared pool across people and LANs). Either way the data
path is the same peer-to-peer mesh; only discovery differs. The hub is documented in [hub.md](hub.md).

> **Bind to the mesh.** When federating, set `listen` to `0.0.0.0:8090` (or the mesh interface IP)
> so peers can reach it. NetBird gives each peer a stable private IP and encrypts all traffic
> between peers — the mesh itself is the firewall. The `token` is defense-in-depth on top.

## Reference topology — two machines, one shared pool

Two machines, two substrates, one shared GPU pool:

```
            NetBird mesh (control plane on your VPS)
   ┌─────────────────────────────┐        ┌──────────────────────────────┐
   │ HOME BOX  (Windows, 2×4090)  │        │ LAPTOP  (Ubuntu 24, 3500 Ada) │
   │                              │        │                               │
   │ pg-ai-stewards substrate ────┼──┐  ┌──┼──► pg-ai-stewards substrate    │
   │   (general / public work)    │  │  │  │     (private work; takes over  │
   │                              │  │  │  │      the private loop)         │
   │ llama-chip :8090  ◄──────────┼──┘  └──┼──► llama-chip :8090            │
   │   dance-moe (qwen-35b GPU0,  │  peers │     local 12GB model           │
   │              gemma GPU1)     │◄──────►│     + federates the 4090s      │
   └─────────────────────────────┘        └──────────────────────────────┘
```

- The **home box keeps its substrate** (general/public digesters) and its `dance-moe` rig.
- The **laptop runs a separate substrate** for your private and work intents — and takes over that loop.
- Each substrate dials its **own local** `llama-chip :8090`. Federation makes the *other* machine's
  GPUs reachable: the laptop's `:8090` serves its 12GB model locally and routes `qwen3.6-35b-a3b`
  to the home 4090s when the mesh is up; when you're away with the home box off, the laptop falls
  back to its local model. Nothing in the home substrate's config changes.

### config per machine

**Home box** `config.json` — add to the existing dance-moe config:
```json
"federation": {
  "node_name": "home-gpu",
  "advertise": "http://<home-mesh-ip>:8090",
  "peers": [ { "name": "laptop", "url": "http://<laptop-mesh-ip>:8090" } ]
}
```
and change `listen` to `"0.0.0.0:8090"`.

**Laptop** `config.json` — a small local slot + the home box as a peer:
```json
{
  "backend": "cuda12",
  "listen": "0.0.0.0:8090",
  "slots": [
    { "model": "<a model that fits 12GB>", "alias": "my-model", "gpus": [0], "ctx_size": 32768, "kv_cache": "q8_0" }
  ],
  "federation": {
    "node_name": "laptop",
    "advertise": "http://<laptop-mesh-ip>:8090",
    "peers": [ { "name": "home-gpu", "url": "http://<home-mesh-ip>:8090" } ]
  }
}
```

> The laptop substrate's `flexllama` provider points at the laptop's own `http://localhost:8090`
> (or `host.docker.internal:8090` from Docker). It asks for `qwen3.6-35b-a3b` exactly as today;
> federation routes it to the home 4090s transparently. Role aliases (`reason`/`critic`/`ingest`)
> resolve by the model name, so no substrate change is needed beyond the provider base URL.

## NetBird mesh — control plane on your VPS (your hands)

NetBird is a self-hostable WireGuard mesh (the open Tailscale alternative). One-time setup:

1. **Stand up the control plane** on `your-vps.example.com` (your VPS). The quickest path is the
   official `getting-started-with-zitadel` compose bundle:
   ```bash
   curl -fsSL https://github.com/netbirdio/netbird/releases/latest/download/getting-started-with-zitadel.sh | bash
   ```
   It brings up the management, signal, and Zitadel IdP services behind Caddy/TLS. Point a
   subdomain (e.g. `mesh.example.com`) at the box and let it issue certs. (Dokploy can also host it,
   but the bundled compose is simplest for the control plane itself.)
2. **Enroll each machine** as a peer:
   ```bash
   # install
   curl -fsSL https://pkgs.netbird.io/install.sh | sh         # linux (laptop)
   #  Windows (home box): download the NetBird MSI from netbird.io and install
   # join the mesh
   netbird up --management-url https://mesh.example.com
   ```
   Each peer gets a stable `100.x.x.x` IP. `netbird status` lists peers + their mesh IPs.
3. **Put those mesh IPs** into each machine's `federation.peers[].url` and `advertise`.
4. **(Optional) NetBird ACLs**: restrict the mesh so only your peers can reach `:8090`. Combined
   with the `token`, that's two layers over WireGuard's encryption.

### Data-safety note (private intents / file_private)

Routing private work from the laptop substrate to the home 4090s sends that data **over the mesh to
your own GPU** — WireGuard-encrypted, peer-authenticated, never leaving hardware you own. That
satisfies the file_private rule (the constraint is *never route private data to a train-on-data
cloud provider*; your own GPUs over an encrypted mesh are fine). Do **not** add a cloud provider as
a federation peer for private intents.

## Laptop substrate bootstrap (Ubuntu 24 — your hands)

The pg-ai-stewards OSS image is Debian-based, so it ports straight to Ubuntu:

1. Install Docker + the NVIDIA Container Toolkit (so the bridge/daemons can see the 3500 Ada if
   needed; inference itself goes through llama-chip, not the containers).
2. Clone the OSS repo + the private overlay; bring up the compose stack (its own DB, bridge, UI,
   persona-host) — a fresh substrate, separate DB from the home box.
3. Apply the overlay chain (incl. the private pipeline overlay). Point the `flexllama` provider at
   the laptop's `http://host.docker.internal:8090`.
4. Install + run llama-chip on the laptop (build `GOWORK=off go build -o llama-chip ./cmd/llama-chip`),
   with the federation config above.
5. **Migrate the private workload**: enable its cron pipeline on the laptop substrate; disable it on the home box.
   The pipeline + overlay are portable — it's the same core, just a different DB.

## Verifying it works

The unit oracle is `internal/fed` + `internal/router` (`GOWORK=off go test ./...`). To prove it
live across two processes (or against an existing rig), run a second llama-chip with a federation
block and no local slots, pointed at the rig:

```bash
# head with zero local GPUs, peering an existing rig on :8090
llama-chip serve --config fed-head.json   # listen :18090, peers → http://127.0.0.1:8090

curl http://127.0.0.1:18090/api/status          # federation.peers shows the rig online + its models
curl http://127.0.0.1:18090/v1/models           # aggregated: the rig's models, owned_by "peer:<name>"
curl http://127.0.0.1:18090/v1/chat/completions \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":2000}'
#   → routed to the peer, real completion comes back (give thinking models ≥2000 tokens)
```

> Proven 2026-06-22: a zero-GPU head federated the live `dance-moe` rig and served a real
> `qwen3.6-35b-a3b` completion by routing to it (and a model nobody serves returned 404). The
> `/v1/models` fallback also covers a peer running a pre-federation llama-chip build.

## Managing the pool — remote control + node-pinned chat

A node's loader UI doesn't just *show* the pool (the **Pool** panel renders every node's GPUs +
models from `/api/pool`); it can **manage any node in it** over the mesh. Each pool card has
**＋ Load…** (opens a form populated with *that node's* GGUFs, profiles, and GPUs), an **Unload all**
button, and an **✕** on each loaded-model chip. Self routes to the local `/api/*` endpoints; a peer
routes through one proxy:

```
GET|POST /api/remote?node=<name>&op=<op>[&…]
```

The named node is resolved to its mesh address (hub roster or static peers) and the call is
forwarded to that node's matching management endpoint with the federation bearer attached; the
peer's response streams straight back. Allowed `op`s:

| op | method | proxies to | use |
|---|---|---|---|
| `status` `gpu` `models` `backends` `profiles` `live` `guess-context` | GET | `/api/<op>` | read a peer's state / populate its forms |
| `load` `unload` `unload-all` `profile` `ensure` | POST | `/api/<op>` | load/unload/switch on the peer |

```bash
# read the home box's loaded models from the laptop
curl "http://localhost:8090/api/remote?node=home-gpu&op=status"
# free the home box's GPUs from anywhere in the pool
curl -X POST "http://localhost:8090/api/remote?node=home-gpu&op=unload-all"
```

> A peer's management endpoints are gated by **its** federation `token` over the mesh — the same
> token the proxy forwards — so a shared-secret cluster authenticates and an open mesh-only cluster
> needs none. The mesh (peer-authenticated, encrypted) is the boundary either way.

### Node-pinned chat (test a specific remote)

The **Chat test** picker lists this node's models first, then each remote node as an optgroup. A
remote pick sends the chat with `?node=<name>`, which **pins** the request to that node even when a
model of the same name is loaded locally — so you can test a chosen remote directly:

```bash
curl -X POST "http://localhost:8090/v1/chat/completions?node=home-gpu" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":2000}'
#   → forced to home-gpu (the ?node= hint is stripped before forwarding, so no double-hop)
```

> Proven 2026-06-23: a scratch node with **zero models** managed the live `dance-moe` rig over the
> federation — read its status/GGUFs via `/api/remote`, opened a remote Load form, and ran a
> `?node=alpha` chat that `qwen3.6-35b-a3b` answered. Unknown node → 404, unknown op → 400.

## Deferred (not in this version)

- **Automatic** federated placement — a node *deciding on its own* which peer has free VRAM and
  loading there (`/api/ensure` as a scheduler). The manual hook exists today (`/api/remote?op=ensure`
  or `op=load` against a chosen node); auto-selection across the pool is the follow-up. Not needed
  for `dance-moe`, since the home box keeps those models loaded and routing covers it.
