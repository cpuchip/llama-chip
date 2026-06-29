# Spec — custom `llama-server` backends (cut the LM Studio runtime tether)

**Status:** P0 verified + **E1 (per-slot backend override) + E2 (managed `pull-ggml` + `ggml@<tag>`
resolution) BUILT + verified** 2026-06-28 (E2 download path proven by a real b9747 pull; a cudart-asset-match
bug was caught + fixed). **E4 `runner` field — design now grounded in Unsloth Studio, not yet built.**
**Date:** 2026-06-28.
**Origin:** the model-rig session of 2026-06-28 — three new coding models served fine on the rig
(Qwen3.5-9B-MTP, Nemotron Cascade 2, EXAONE 4.5), but **North Mini Code (`cohere2moe`) and DiffusionGemma
(block-diffusion) crashed**: LM Studio's bundled `llama.cpp` (cuda12@2.22.0) doesn't have those
architectures yet, and llama-chip can only use what LM Studio downloaded.

## Why

llama-chip's design (README "How it works") is to **reuse what LM Studio already downloaded** — both the
**models** (`~/.lmstudio/models/`) *and* the **`llama.cpp` backends** (`~/.lmstudio/extensions/backends/`).
The model half is great; the backend half is a **tether**: LM Studio's compiled `llama.cpp` builds lag
upstream `ggml-org/llama.cpp` by weeks, so a brand-new architecture (`cohere2moe`, a new diffusion arch,
the next big model) **crashes until LM Studio ships a newer runtime**. llama-chip was born to escape a
dying *wrapper* (FlexLLama); this finishes the job by escaping the *runtime cadence* — letting a slot run
on a `llama-server` **we** control, pulled straight from `ggml-org` releases (published ~daily, far ahead
of LM Studio).

Upstream facts (verified 2026-06-28): `ggml-org/llama.cpp` is MIT; its [GitHub Releases] publish prebuilt
**`llama-server.exe` + CUDA DLLs** per build (e.g. `llama-b9747-bin-win-cuda-12.4-x64.zip` +
`cudart-llama-bin-win-cuda-12.4-x64.zip`, build b9747 = 2026-06-21). LM Studio's runtimes are *its own*
recompiles of the same source, distributed from `files.lmstudio.ai` — strictly behind upstream.

## What already works (grounded in the code)

The custom-backend path is **already ~90% built** — `resolveBackend` accepts an explicit directory:

```go
// internal/rig/rig.go
func resolveBackend(spec string) (backends.Backend, error) {
    if strings.ContainsAny(spec, "/\\") {                 // a path, not a variant name
        srv := spec + sep + backends.ServerBinName()      // <dir>/llama-server.exe
        if _, err := os.Stat(srv); err == nil {
            return backends.Backend{Variant: "custom", Dir: spec, Server: srv, Complete: true}, nil
        }
        return ..., fmt.Errorf("no %s in %s", ...)
    }
    return backends.Latest(backends.Variant(spec))         // else: the LM Studio path (default)
}
```

And `config.Backend` already documents it: *`"cuda12" (default) | "cuda" | "rocm" | "vulkan" | "cpu" | or
an explicit dir`*. The launcher (`internal/rig/rig.go launch`) puts the backend's DLL dirs on `PATH`
(Windows) / `LD_LIBRARY_PATH` and sets `cmd.Dir = Backend.Dir`, so a **self-contained release directory
(binary + all DLLs co-located) runs as-is** — `DLLDirs()` returns `[Dir]` for a `custom` variant (its
`vendorDir()` only applies to the LM Studio split layout).

**So the manual recipe works today:**
1. Download from [ggml-org releases](https://github.com/ggml-org/llama.cpp/releases) the matching pair for
   this box (2× RTX 4090, CUDA 12): `llama-bNNNN-bin-win-cuda-12.4-x64.zip` **and**
   `cudart-llama-bin-win-cuda-12.4-x64.zip`.
2. **Extract BOTH zips into one directory** (so `llama-server.exe`, `ggml-*.dll`, *and* `cudart64_12.dll`
   / `cublas64_12.dll` are all co-located — that's the difference from LM Studio's `backends/vendor/`
   split, which a `custom` backend doesn't look for).
3. Set `"backend": "C:/.../llama-cpp-b9747-cuda12"` in `config.json` (rig-global), or test one slot (see E1).

**P0 of this spec is to *verify* that recipe** — the inverse hypothesis: North Mini Code (`cohere2moe`)
crashes on LM Studio's cuda12@2.22.0 but **serves** on a fresh ggml-org build (which has merged PR #24260).
If P0 passes, the feature already exists and the rest is ergonomics.

## Enhancements (net-new)

### E1 — per-slot backend override (the valuable one)

Today `Backend` is **rig-global**: `config.Backend` → `resolveBackend` once → every slot shares it. That
forces an all-or-nothing choice. Add an **optional per-slot** backend so a mixed rig is possible — the
stable dance models on LM Studio's vetted cuda12, one slot on a fresh ggml build for a bleeding-edge arch:

```jsonc
// config.Slot gains:
"backend": "C:/.../llama-cpp-b9747-cuda12"   // optional; falls back to config.Backend, then "cuda12"
```

Build: add `Backend string` to `config.Slot`; in `rig` constructor, resolve each slot's backend
(`slot.Backend` → else `config.Backend`), so `Instance.Backend` is per-slot. Small, surgical — the launch
path already takes `in.Backend`. **This unblocks "test a new arch on one slot without restarting the whole
rig on a different backend"** — exactly the friction in the 2026-06-28 session.

### E2 — managed `pull-ggml` + `backend: "ggml@<build>"`

So nobody hand-downloads two zips. Mirror the existing `llama-chip pull` (which wraps `lms runtime get`)
with a sibling that pulls from **ggml-org** instead:

```
llama-chip pull-ggml [bNNNN|latest]   # download the matching win-cuda-12.4 binary + cudart zips,
                                       # extract BOTH into ~/.llama-chip/backends/ggml-bNNNN/
llama-chip backends                    # now also lists ggml-managed builds (newest first)
```

Then `resolveBackend` learns a third form: `"ggml@latest"` / `"ggml@b9747"` → resolve (and, if absent,
pull) the managed dir. Pick the asset by OS + CUDA family (`win-cuda-12.4` for this box). This is pure
convenience over E1's manual dir; ship it second.

### E3 — discovery niceties (optional)

`llama-chip backends` already lists LM Studio builds; have it also scan `~/.llama-chip/backends/` and flag
custom/ggml builds, so `backends` is the one place to see every runnable `llama-server`.

### E4 — the `runner` field (non-`llama-server` engines: block-diffusion / DiffusionGemma)

**DiffusionGemma is NOT solved by E1–E3.** Block-diffusion LLMs are served by a *different binary* than
`llama-server` (e.g. `llama-diffusion-*`, llama.cpp PR #24423), with a different generation protocol
(multi-canvas diffusion, not token-by-token). The `runner` field is how a slot dispatches to such an engine
while the OpenAI router stays unchanged. (Promotable to its own spec once we actually build it.)

**Grounded design — Unsloth Studio already solves this exact problem** (explored 2026-06-28; we mimic the
*design*, not the code — see the license caveat below). Their block-diffusion path
(`studio/backend/core/inference/llama_cpp.py`) does five things llama-chip should copy as a *pattern*:

1. **Route by GGUF metadata, before arch resolution.** Read the GGUF header; treat the slot as diffusion if
   `general.architecture` starts with `"diffusion"` **or** the file carries a `diffusion.canvas_length` key.
   Deciding *before* normal arch-resolution means an unknown/bleeding-edge arch doesn't fall back to
   `llama-server` and crash. (Unsloth: `_is_diffusion`.)
2. **A different binary behind the SAME interface.** The diffusion runner presents the *same* `/v1` +
   `/health` as `llama-server`, "so the rest of Studio is unchanged." → llama-chip's router and health-check
   need **zero** changes; only the launch differs.
3. **Runner discovery = env override OR adjacent to `llama-server`.** Find it via an env var
   (`DG_VISUAL_BIN`-style) **or** next to `llama-server` in the same backend dir (`build/bin`, `Release`).
   Absent → actionable error, never a silent downgrade. Fits llama-chip's self-contained backend dir: the
   runner ships in the same `ggml-bNNNN/` extract once such builds exist.
4. **Launch shape:** `runner --gguf <path> --host 127.0.0.1 --port <p> --gpu <id> --maxtok <n>`; child dies
   with the parent (no orphaned GPU process); read the auto-sized context back from stdout.
5. **Dispatch short-circuit:** after the metadata read, `if isDiffusion { return startRunner(...) }` *before*
   the `llama-server` branch; resolve the `llama-server` binary lazily so its not-found error is deferred (a
   diffusion GGUF doesn't need it).

**llama-chip shape:** a `"runner"` field on `config.Slot` (default empty = `llama-server`), or auto-detected
from GGUF metadata per (1). `Backend` still supplies the *dir*; `runner` selects *which binary in it*. The
OpenAI router is untouched (2). Build when a diffusion `llama-server`-adjacent binary actually exists in a
ggml-org (or our own) release — **until then, run DiffusionGemma in Unsloth Studio** and optionally federate
llama-chip's router to its OpenAI endpoint.

## Patterns to adopt from Unsloth Studio (explored 2026-06-28)

Unsloth Studio's backend is a Python peer to llama-chip — same job (manage llama.cpp installs + GGUFs, serve
OpenAI-compat). Its lifecycle code (`studio/install_llama_prebuilt.py`,
`studio/backend/core/inference/llama_cpp.py`) is worth mimicking by *design*:

1. **Runner dispatch (E4 above)** — the highest-value steal; it *is* the runner field.
2. **Integrity-verified pulls + dual source.** Their fork publishes a schema-versioned `manifest.json` +
   separate `sha256.json` per release; the host picks its artifact from the manifest's GPU/arch coverage and
   **refuses any asset absent from the checksum manifest**. Vanilla ggml-org has no manifest → selected by
   filename regex (what `pull-ggml` does today). Adopt-later: if we ever publish our own backend builds, ship
   a manifest+sha; for ggml-org pulls, add a sha256 check if/when they publish one.
3. **Capability probe before arg-building.** Parse `<server> --help`, cache on `(path, mtime)`, feature-gate
   flags per build — so a flag present on a fresh ggml build but not on LM Studio's older one is detected,
   not assumed. Directly relevant since llama-chip juggles backends of different vintages.
4. **Binary-discovery ladder + owned-process reaping.** An ordered env→install-tree→PATH search that
   distinguishes a *transiently locked* binary (Windows AV / in-flight install) from a *missing* one, and
   only kills servers under a known install root. Both map straight onto a Go multi-backend manager.
5. **Model registry — confirmation NOT to over-build.** Unsloth keeps no alias catalog: a model is an HF
   `repo_id` + a quant "variant" → blob-hash → `huggingface_hub` snapshot into the stock HF cache, plus a
   small default-seed list and a path→clean-name normalizer for `/v1/models`. Validates llama-chip's lean
   approach; if we add model-pulling, copy this shape, not a heavy catalog.

**⚠ License caveat (load-bearing).** Unsloth Studio's **entire backend** — every file above (inference /
serving / diffusion / installer) — is **AGPL-3.0**, *not* "Apache core / AGPL UI" as first assumed. Only the
`unsloth/` training tree is Apache-2.0, and that's the part least useful here. **Mimic the designs/
architecture (which are not copyrightable) in our own Go; do not translate or lift those files.** The
runner-dispatch idea, manifest+sha scheme, discovery ladder, and HF-as-registry approach are all expressible
independently.

## Build plan + oracle

- **P0 — verify the existing path.** Download a fresh ggml-org `win-cuda-12.4` build (+ cudart) into one
  dir; `"backend": "<dir>"`; load **North Mini Code**. **Oracle = the inverse hypothesis:** it crashes on
  LM Studio cuda12@2.22.0 (already observed) and **serves** on the ggml build. If it serves, the core
  feature is proven and we just add ergonomics. (If even the fresh build lacks `cohere2moe`, build the PR
  branch from source — documents the build-from-source fallback.)
- **P1 — E1 per-slot backend.** `Slot.Backend` + per-slot resolve. Test: dance-moe on cuda12 + North Mini
  on the ggml dir, both healthy at once. Keep a unit test on the resolver (variant vs path vs `ggml@`).
- **P2 — E2 managed `pull-ggml`** + `ggml@<build>` resolution + `backends` listing.
- **P3 — the `runner` field (E4)** for `llama-diffusion-*` / non-`llama-server` engines — design now grounded
  in Unsloth Studio's block-diffusion path (route-by-GGUF-metadata → same `/v1` interface → binary adjacent
  to `llama-server`). Build when a diffusion `llama-server`-adjacent binary ships in a ggml-org (or our own)
  release; until then DiffusionGemma runs in Unsloth Studio.

## Risks / compatibility

- **Default unchanged.** `config.Backend` defaults to `"cuda12"` (LM Studio path); custom/ggml is strictly
  opt-in. The substrate's `dance-moe` profile must still boot identically — regression-test it after E1.
- **CUDA family must match** the box (cuda12 ↔ the `cuda-12.4` asset; not the cuda-13.3 one). `pull-ggml`
  picks it; the manual recipe must say it.
- **Self-contained dir requirement** (both zips into one folder) — document loudly; the 0xC0000135
  "DLL not found" crash is the failure mode if `cudart` is missing (same class as the LM Studio vendor-dir
  note already in `backends.go`).
- **Ethos:** this *deepens* llama-chip's reason for being — full independence from any wrapper, now for the
  engine too. LM Studio stays the (excellent) model downloader; the runtime is ours when we want the edge.
</content>
