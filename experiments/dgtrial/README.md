# dgtrial — DiffusionGemma × pg-ai-stewards (detached trial)

A little Go program that drives Unsloth's prebuilt `llama-diffusion-gemma-visual-server` over its
stdin/stdout protocol and runs **real pg-ai-stewards work-item stages** through **DiffusionGemma**
(`unsloth/diffusiongemma-26B-A4B-it-GGUF`, Q4_K_M, one RTX 4090, ~263 tok/s).

It is **detached**: the fixtures under `fixtures/` are snapshots of real work-item stages (the binding
question + the substrate model's own output as a reference). Nothing here touches the live substrate. The
point is to measure whether DiffusionGemma is worth wiring into the substrate's dispatch **before** building
that plumbing.

## How DiffusionGemma would integrate into pg-ai-stewards

The substrate already dispatches model calls to OpenAI-compatible endpoints (LM Studio, the llama-chip
router on `:8090`, remote providers). DiffusionGemma's runner is **not** `llama-server` and does **not**
speak HTTP — it's a stdio worker. So the integration path is:

**llama-chip E4 (the `runner` field).** A llama-chip slot with `"runner": "diffusion-gemma"` spawns the
visual-server and fronts it with an OpenAI `/v1` adapter on the router (`:8090`) — the OpenAI-translation
front this trial's driver prototypes. Then the substrate dials `model: "diffusiongemma"` like any other
model: `list_models` gains a row, and a pipeline stage can request it via `model_override`. **No substrate
change** beyond a model-registry entry — the adapter lives in llama-chip, where the diffusion knowledge
already is.

This trial is the **de-risking step before E4**: it answers "is DiffusionGemma good enough at substrate work
to be worth the runner plumbing?" The driver here (`main.go`) is the E4 adapter's seed — it already does the
hard part (spawn, READY handshake, request/stream/parse over stdio); E4 just wraps it in an HTTP server.

### Which stages suit it

DiffusionGemma is fast, reasons (a `<|channel>thought` trace), codes, **and tool-calls natively**
(Google model card: "Function Calling — native support for structured tool use"; proven live, fixture 03 —
it emits `<|tool_call>call:NAME{args}<tool_call|>` through the visual-server harness with tools merely
inlined in the system prompt). So it fits a wide range of stages:

- ✅ **plan**, **critique**, **synthesize / draft**, **code-gen** — self-contained generation.
- ✅ **agentic / tool-using stages** — it returns structured tool calls, so a stage that needs
  `doc_search` / `doc_get` / file tools / web can run on it. The E4 adapter parses the `<|tool_call>…`
  tokens into OpenAI `tool_calls`, executes, and feeds the result back (standard local-model tool loop).
- The only real gap is **harness plumbing**, not model capability: the visual-server's request schema
  (`{seed, n_blocks, messages}`) has no `tools` slot, so the E4 adapter must format tool definitions into
  the prompt (or extend the request to pass them to the GGUF chat template) and parse the tool-call tokens
  from the output. Inlining tools in the system prompt already works (fixture 03).

The seeded fixtures: a **plan** stage and a **content** stage (real doc-builds, with `qwen3.6-35b-a3b`
references pulled from the live substrate), plus a **tool-call** capability probe (fixture 03).

## Run

```sh
go build -o dgtrial.exe .
./dgtrial.exe                 # loads the 16GB GGUF once, runs every fixture, writes results.md
# flags: -server -model -fixtures -out -ngl -gpu
```

Requires the CUDA-13 runtime DLLs co-located with the server (workspace memory
`reference_local_coding_models_rig`: the CUDA-12 ones load but silently fall back to **CPU**).

## Findings

First run (2026-06-29, one RTX 4090, Q4_K_M, 9s model load — see `results.md` for full outputs):

| Fixture | Stage | DiffusionGemma | vs substrate (`qwen3.6-35b-a3b`) |
|---|---|---|---|
| doc plan | `plan` | 1133 tok, **144 tok/s**, 7.8s | structurally complete, all 5 areas; **but terminology drifted** (guessed STA attributes "Daring/Control", a non-existent "Ops" dept) — no corpus grounding |
| faction table | `build-content` | 716 tok, **236 tok/s**, 3.0s | competitive — **corrected 2 of qwen's errors** (Ferengi/Breen = Alpha Quadrant, not Gamma) while making its own (Klingon → Alpha, should be Beta); ~20× faster, terser |

**Verdict.** DiffusionGemma is a strong, fast candidate across **most generation AND agentic stages** —
plan, draft, critique, code-gen, quick factual content, *and* tool-using work. Caveats for integration, all
**harness** rather than model limits:

1. **No corpus grounding → terminology drift** on domain specifics *when it generates blind*. For a `plan`
   stage that's fine (the `build` stage re-grounds via `doc_search`). For accuracy-critical output, feed the
   retrieved context **inline** — or let it tool-call to fetch (it can; see #4).
2. **Strip the thought channel.** Raw output is `<|channel>thought … <channel|><final answer>`; split on
   `<channel|>` and keep the tail (the E4 adapter should do this).
3. **No `tools` slot in the visual-server request** (`{seed, n_blocks, messages}`). The E4 adapter formats
   tool defs into the prompt (or extends the request to pass them to the chat template) and parses the
   tool-call tokens. Inlining tools in the system prompt already works today.
4. **Tool-calling is real** (corrects an earlier wrong claim): given a tool it emits
   `<|tool_call>call:get_current_weather{city: "Tokyo"}<tool_call|>` — proven live (fixture 03). So it can
   run `gather` / tool-`build` stages, not just self-contained ones.

It earns the E4 runner plumbing. The driver here already does the stdio handshake/stream/parse; E4 wraps it
in an OpenAI `/v1` server (translating `<|channel>` / `<|tool_call>` markers into `content` + `tool_calls`)
so the substrate dials `model: "diffusiongemma"` like any other model.
