# Model insights — which local model for which role

What we've actually measured on the 2×RTX-4090 rig (llama-chip serving LM Studio's CUDA
llama-server), and what role each model earns. Updated 2026-06-21.

The substrate dispatches by **role**, not by model name — `ingest` / `reason` / `critic`
(see the overlay `role-aliases.sql`). This doc says which model should sit behind each role,
and why, from head-to-head tests rather than vibes.

## TL;DR — role assignments

| Role (substrate alias) | Best model | Why |
|---|---|---|
| **ingest / gather** | **gemma-4-26B-A4B** (MoE) | Tightest, most concise briefs; fast; cheap KV → fits huge multi-source context |
| **reason / synthesize** | **qwen** — `qwen3.6-35B-A3B` (MoE) for speed, `qwen3.6-27B` (dense) for max depth | Most depth + crispest structure on hard reconciliation tasks |
| **critic / judge** | **qwen** | Most thorough + precise; names the fallacy, more lexical detail |

This **confirms the current dance** (gemma = ingest, qwen = reason/critic). The one open
upgrade is swapping the dense qwen-27b → the MoE qwen-35b-a3b for reason/critic (see below).

## Measured throughput (single request, one card, parallel:1)

| Model | Quant | tok/s | Notes |
|---|---|---|---|
| gemma-4-26B-A4B | Q4_K_M | **~161** | MoE, ~4B active — the gather model |
| qwen3.6-35B-A3B | Q4_K_M | **~183** | MoE, ~3B active |
| qwen3.6-27B | Q4_K_M | **~47** | dense — current reason/critic |
| gemma-4-12B | Q8_0 | ~58 | dense |

**The headline: MoE models run 3–4× faster than dense models of similar capability on this
rig**, because only ~3–4B params are active per token. They also *reason more* (longer hidden
chains) yet still finish faster in wall-clock. (Note: the dance runs the gather MoE at
`--parallel 2`, which splits one card across two request slots, so per-request throughput there
is roughly half these single-slot numbers — still far above the dense models.)

## Experiment 1 — qwen3.6-27b (dense) vs qwen3.6-35B-A3B (MoE)

Same three role tasks, both Q4, one card each.

- **Speed:** 35b-A3B **183 tok/s** vs 27b-dense **47 tok/s** — ~3.9× faster.
- **Quality:** comparable. On the hardest task (reconcile "saved by grace, not works" with
  "faith without works is dead"), both were excellent; the MoE's *"Paul combats legalism;
  James combats antinomianism … Paul defines salvation's basis, James its evidence"* was the
  crispest framing, while the dense model went slightly deeper. No task where dense clearly won.
- **Verdict:** the **MoE (qwen-35b-a3b) is the better default for reason/critic** — same-or-crisper
  output at ~4× the speed. **The VRAM caveat turned out to be a non-issue** (probed 2026-06-21): its
  GQA makes KV cheap (~13.6 MiB/1k tokens at q8), so it loads and serves at its **full trained 256k
  context on one 4090** — 192k leaves a ~1 GB cushion, matching the dense 27b's current 192k. So the
  swap is a clean win: same context, ~4× the speed. (256k itself fits but with a tight ~530 MiB
  cushion — keep it ≤192k for safety under inference.)

## Experiment 2 — gemma-4-26B-A4B (MoE) vs gemma-4-12b (dense)

- **Speed:** 26B-A4B **160 tok/s** vs 12b-Q8 **58 tok/s** — ~2.7× faster.
- **Quality:** comparable; the MoE was sharper on critique (named *why* an etymology→theology
  leap fails), tied on synthesis and extraction.
- **Verdict:** the MoE is worth it — faster at equal-or-better quality. The dense 12b is fine for
  simple extraction/summarization if you ever need to free VRAM. The original "12b hung 30 min"
  scare was a bad config (QAT quant @ 512k context), not the model — and the gather tool-grant has
  since been trimmed 159→15 tools, so the pressure that drove the swap is mostly gone. The MoE
  stays the default because it's faster, not because 12b is unusable.

## Experiment 3 — qwen vs gemma, per role

Same three tasks across qwen-27b, qwen-35b-a3b, and gemma-26b-a4b.

- **Gather (synthesize many sources → tight brief):** **gemma wins.** Its briefs were the most
  concise and complete (all four sources woven, ~1,250 tokens in 8 s) where the qwens were more
  expansive. A gather brief should be tight — gemma's instinct fits.
- **Reason (reconcile a real tension):** **qwen wins.** Both qwens had more theological depth and
  a crisper cause-vs-evidence structure; gemma was good but a notch more formulaic.
- **Critic (catch planted errors):** **qwen wins (gemma close).** The qwens were more thorough and
  named the fallacy precisely ("etymological fallacy," specific lexical equivalents); gemma caught
  the same core error but with less detail.

## Caution — models confabulate specifics, even the critic

In the critic task, *all three* models correctly demolished the planted fallacy but each invented
or mis-stated a historical detail about Tyndale (one said he "didn't coin" atonement, another moved
the date to 1530, another said it "predates" him — Tyndale is generally credited with the ~1526
coinage). **Take model-asserted dates, names, and numbers as unverified.** This is exactly why the
substrate's quote/fact gate (read the source, then assert) stays mandatory — the local models are
strong at *reasoning over* given material and weak at *recalling* specifics.

## VRAM + context cheat-sheet (24 GB card, Q4 weights, q8 KV)

| Model | Weights+buffers | Realistic max context on one card |
|---|---|---|
| gemma-4-26B-A4B | ~17.3 GB | ~213k/slot × 2 slots (KV is very cheap on this MoE) |
| qwen3.6-27B (dense) | ~17 GB | ~192k |
| qwen3.6-35B-A3B (MoE) | ~21 GB | **up to 256k** (trained max) on one card — KV is cheap (~14 MiB/1k); 192k leaves ~1 GB cushion |
| gemma-4-12B | ~7.5 GB (Q4) / ~12 GB (Q8) | very large; tiny footprint |

Two models can't share one card: the **weights** don't fit (e.g., 26b ~17 GB + 12b ~7.5 GB > 24 GB),
and reducing context only shrinks KV, not weights. Each model needs its own card; a third
simultaneous model needs a third GPU (a convenience for diversity, not a bottleneck fix — the MoEs
are fast enough that the 2-card dance has headroom).

## Parallelism + context windows (`--parallel`)

**How `--parallel` and context interact (this is the mental model):** `--ctx-size` is the **TOTAL**
KV cache, split evenly across `--parallel` slots. Each slot is an independent conversation with its
own KV cache; **per-request context = ctx_size ÷ parallel**. A slot's context can't exceed the
model's trained max, but the *total* you allocate (`per-slot × parallel`) can exceed a single
model's max because it's N separate sequences — that's what you saw on the 12b. So to give two
concurrent requests 131k each, set `ctx_size 262144, parallel 2`.

⚠️ **llama-chip's `ctx_size` is the total** (it passes it straight to `--ctx-size`). FlexLLama's
config `n_ctx` was *per-slot* and it multiplied. Porting the dance 1:1 silently **halved** gemma's
per-request context (212992 total ÷ 2 = 106k/slot instead of the intended 213k). Fixed
2026-06-21 — the dance gemma is now `ctx_size 425984, parallel 2` = 213k/slot.

**Does parallel-2 actually help? Yes (measured 2026-06-21):**

| | qwen-35b-a3b | gemma-26b-a4b |
|---|---|---|
| solo (1 req) | 178 tok/s | 151 tok/s |
| parallel-2 aggregate (2 concurrent) | 281 tok/s (**1.58×**) | 240 tok/s (**1.59×**) |
| per-request under load | ~144 each | ~131 each |

So two concurrent requests finish for the price of ~1.25 — real continuous-batching benefit. Each
individual request is a bit slower, but a judge and a gather no longer block each other.

**Do we get true parallelism across *both* cards? Yes — 0.94×.** Four concurrent requests (2 per
card) delivered **488 tok/s aggregate**, 94% of the sum of each card's independent parallel-2
throughput (522). The 6% loss is minor batching/host overhead; the cards genuinely run at once.

**"2× both" recommended config** (both MoE, parallel-2):
- GPU0: qwen-35b-a3b `ctx 262144, parallel 2` = 131k/slot (~486 MiB free — tight; drop to ~120k/slot for cushion)
- GPU1: gemma-26b-a4b `ctx 425984, parallel 2` = 213k/slot (~936 MiB free)

**gemma-4-31b (dense) note:** it's *dense*, not MoE — the gemma MoE is 26b-a4b. The 31b is slower
(41 tok/s solo vs the MoE's 161) and its KV is heavier; at parallel-2 it tops out ~64k/slot
comfortably (vs the MoE's 213k/slot). Single-slot it holds the full 256k (sliding-window KV plateaus
~128k) but with a tight ~482 MiB cushion. **For the parallel gemma slot the 26b-a4b MoE wins
decisively** — far more per-request context and ~4× the speed.

## Reproduce

`/tmp/roles.py <alias>...` (and `/tmp/ab.py`) drive llama-chip directly: load the models one per
card via `/api/load`, then POST the same prompts to each `/v1/chat/completions` and diff outputs +
timings. Pause the substrate first (UI → **Free GPUs**, or set `autonomy_paused`), run, then restore
the `dance` profile and resume. Reasoning models need `max_tokens ≥ ~3000` or they return empty
`content` (thinking-only).
