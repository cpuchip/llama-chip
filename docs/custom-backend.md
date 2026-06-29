# Spec — custom `llama-server` backends (cut the LM Studio runtime tether)

**Status:** P0 verified + **E1 (per-slot backend override) BUILT** 2026-06-28; E2 (managed `pull-ggml`)
is next. **Date:** 2026-06-28.
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

## Out of scope → a separate, bigger spec

**DiffusionGemma (block-diffusion) is NOT solved by this.** It needs a *different binary* —
`llama-diffusion-cli` (llama.cpp PR #24423) — with a different generation protocol (multi-canvas diffusion,
not token-by-token), so it isn't `llama-server`-shaped and won't answer `/v1/chat/completions` the normal
way. That's a future **per-slot `runner` field** (`"runner": "llama-diffusion-cli"`) + an OpenAI-API
translation shim — its own design pass. **For now, run DiffusionGemma in Unsloth Studio** (it bundles the
diffusion runner and exposes an OpenAI endpoint llama-chip's router could even federate to).

## Build plan + oracle

- **P0 — verify the existing path.** Download a fresh ggml-org `win-cuda-12.4` build (+ cudart) into one
  dir; `"backend": "<dir>"`; load **North Mini Code**. **Oracle = the inverse hypothesis:** it crashes on
  LM Studio cuda12@2.22.0 (already observed) and **serves** on the ggml build. If it serves, the core
  feature is proven and we just add ergonomics. (If even the fresh build lacks `cohere2moe`, build the PR
  branch from source — documents the build-from-source fallback.)
- **P1 — E1 per-slot backend.** `Slot.Backend` + per-slot resolve. Test: dance-moe on cuda12 + North Mini
  on the ggml dir, both healthy at once. Keep a unit test on the resolver (variant vs path vs `ggml@`).
- **P2 — E2 managed `pull-ggml`** + `ggml@<build>` resolution + `backends` listing.
- **P3 (separate spec) — the `runner` field** for `llama-diffusion-cli` / non-`llama-server` engines.

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
