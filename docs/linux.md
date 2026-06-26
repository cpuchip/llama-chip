# Running llama-chip on Linux

llama-chip was written for Windows multi-GPU boxes; it builds and runs on Linux with a small
set of OS-conditional fixes (all in this tree). Validated 2026-06-22 on Ubuntu + an RTX 3500
Ada laptop GPU, driving LM Studio's Linux llama.cpp backends, and federated to a remote 4090
node ("alpha") via the hub.

## What was Windows-specific (now fixed, OS-conditional)

`internal/backends/backends.go`
1. **Backend dir prefix.** `parseDirName` hardcoded `llama.cpp-win-x86_64-`; LM Studio names the
   Linux dirs `llama.cpp-linux-x86_64-…` (and macOS `…-mac-…`). Now built from `osToken()`
   (win/linux/mac) → all backends discover correctly. *This was the blocker* — without it
   `llama-chip backends` reported "no backends found" and `serve` couldn't resolve one.
2. **Server binary + ggml core lib names.** `llama-server.exe` / `ggml-base.dll` → `ServerBinName()`
   (`llama-server` on Linux) and `coreLibName()` (`libggml-base.so` on Linux, `.dylib` on macOS).
   The "runnable" check now finds the real engine lib on every OS.

`internal/rig/rig.go`
3. **Runtime library path.** The launcher put the backend's lib dirs on `PATH` (Windows DLL
   resolution). On Linux the `.so` deps (ggml-cuda, cudart/cublas) resolve via **`LD_LIBRARY_PATH`**
   (macOS: `DYLD_LIBRARY_PATH`); sibling libs also via the binary's `$ORIGIN` rpath (hence
   `cmd.Dir`). Now OS-conditional. Also the explicit-dir backend resolver uses `ServerBinName()`.

No build tags needed — all `runtime.GOOS` switches in shared files; the binary is portable.

## Paths on Linux

LM Studio keeps its **data** under `~/.lmstudio/` even when the **app** is installed elsewhere
(e.g. `/opt/LM-Studio`): backends in `~/.lmstudio/extensions/backends/`, GGUFs in
`~/.lmstudio/models/`, the `lms` CLI at `~/.lmstudio/bin/lms`. So the default discovery paths
work as-is. If your data dir differs, override per the existing config knobs:
`backends_dir` / `models_dir` in config.json, or `LLAMACHIP_BACKENDS_DIR`.

## Build & run (Linux)

```
go build ./cmd/llama-chip
./llama-chip backends   # cuda12/cuda/vulkan/cpu, newest first — all "yes" runnable
./llama-chip gpu        # nvidia-smi works cross-platform
./llama-chip serve --config config.json
```

`nvidia-smi` is already cross-platform, so GPU state works unchanged.

## Validated end-to-end on Linux
- discovery (`backends`/`models`/`gpu`) ✓
- local spawn: `POST /api/load` → `llama-server` healthy on the local GPU (the LD_LIBRARY_PATH fix) ✓
- federation gateway (`slots: []` + `hub_url`/`hub_token`): peer models appear in `/v1/models`
  as `owned_by: peer:<node>`, chat routes over the mesh ✓
- as a pg-ai-stewards `flexllama` provider (`host.docker.internal:8090/v1`): the substrate
  dispatched a chat that routed through llama-chip to alpha's 4090s and returned ✓
