# Module 05 — Inference & Serving Mastery

## Why this module matters

At senior level, "serving" means making one endpoint fast and cheap. At principal level, it means owning the org's entire latency/cost/quality frontier: every model in production, what each costs per request, what SLO it carries, and whether the portfolio as a whole is on the efficient frontier or 3× off it. Most organizations cannot answer "how many models do we serve and what does each cost?" — and the engineer who produces that answer, then acts on it, has just done the most valuable serving work of the year without tuning a single kernel. The ML System Design course covers the mechanics of batching, caching, and cascades per-system; this module is about the org-level decisions: which serving stacks to kill, which to consolidate, when self-hosting flips economical, and how to design SLOs that survive contact with finance.

## 1. The serving portfolio view

The first principal move on any serving problem is to zoom out from the endpoint to the inventory. Build this table — one row per model in production:

```text
model | owner team | framework | QPS (p50/peak) | latency SLO | actual p99 |
hardware | $/month | $/1k req | last retrain | on-call rota | tier (user-facing / internal / batch)
```

Expect the exercise to take two to four weeks at a 200-engineer company, because the information does not exist in one place. You will find models nobody remembers deploying, endpoints serving 0.2 QPS on a dedicated GPU ($2,000/month for a model that could run on a $30/month CPU box), three teams independently running sentence-transformer embedding services, and at least one "temporary" Flask server that has been temporary for three years. Sculley et al.'s "Hidden Technical Debt" (NeurIPS 2015) called this configuration and glue-code debt; a decade later the serving layer is still where most of it accumulates.

Why this table is the first move, not an audit formality:

- **It converts serving from anecdote to economics.** "Serving is expensive" is a complaint; "$340k/month across 23 endpoints, 60% of it on GPUs averaging 11% utilization" is a business case.
- **It reveals the consolidation targets.** Sort by $/1k requests. The top rows are almost always low-QPS models on over-provisioned hardware — the cheapest wins available.
- **It surfaces the on-call tax.** Count distinct serving stacks. Each one is a separate deploy pipeline, dashboard set, failure taxonomy, and 2 a.m. page that only one person knows how to handle.

A useful planning number: in orgs of 50–500 engineers that grew ML organically 2019–2024, expect 5–10 distinct serving stacks and GPU utilization of 10–25% fleet-wide. Getting fleet utilization from 15% to 50% is a 3.3× cost reduction with zero model changes — no optimization project inside a single endpoint comes close.

The table is not a one-time audit; it becomes a standing quarterly review with four questions per row:

```text
1. Is actual p99 within SLO — and is the SLO still the right one? (SLOs rot both ways)
2. Is $/1k req trending down, flat, or up — and why?
3. Does this model still earn its serving cost? (kill candidates surface here)
4. Is it on the platform yet? If not, what's the date?
```

Twenty minutes per quarter per row, and the org's serving posture stops being a mystery. The review artifact — one page, the table plus deltas — is also how you make the cost frontier legible to directors and finance, which is what unlocks budget for the consolidation work below.

## 2. Consolidation economics

The steady state to aim for is a **two-tier platform**: one CPU-oriented tier for classic ML (sklearn/XGBoost/small Torch — Triton, KServe, or a thin FastAPI+ONNX Runtime layer behind a shared deploy pipeline) and one GPU tier for deep/generative models (vLLM or Triton+TensorRT-LLM), plus managed APIs as a third "tier zero" for frontier-quality long-tail traffic. N bespoke stacks → 2 platforms. The math:

**Ops load.** Each bespoke stack costs roughly 0.3–0.5 FTE/year in maintenance: dependency upgrades, CVE patching, deploy-pipeline fixes, dashboard rot, tribal-knowledge transfer when its one expert leaves. Seven stacks ≈ 2.5–3.5 FTE ≈ $500–700k/year fully loaded, before you count the incident cost of stacks nobody knows well.

**On-call.** Distinct stacks cannot share a rotation meaningfully — a pageable rotation needs 4–6 people who can actually debug the thing. Seven stacks either means seven shallow rotations (nobody can debug what they're paged for) or one rotation with a 7× wide surface (every incident starts with 30 minutes of "how does this one deploy?"). One platform means one runbook, one failure taxonomy, one deep rotation.

**GPU bin-packing.** This is the quietly dominant term. Bespoke stacks pin dedicated GPUs per model; a shared GPU tier packs models together. Concretely:

```text
Bespoke:  5 models × 1 dedicated A10G each, avg 15% util  → 5 GPUs, ~$4,300/mo (on-demand)
Packed:   same 5 models on 2 A10Gs via Triton multi-model
          (or MIG slices on an A100), 60% util             → 2 GPUs, ~$1,700/mo
```

Fleet-wide at even modest scale (30–50 GPUs), bin-packing plus autoscaling typically recovers 40–60% of GPU spend. The enablers are multi-model serving (Triton instance groups, vLLM served with multiple LoRA adapters over one base), MIG partitioning for small models on big GPUs, and a scheduler that treats GPUs as a pool, not as pets.

**What consolidation is not.** It is not a rewrite of every model. The platform absorbs models via a thin contract — a standard input/output schema, a container interface, a registry entry — and models migrate in economic order (highest $/req and highest incident-count first). Any migration plan that starts with "first we rewrite everything in framework X" is a red flag; you'll see the correct ordering in the worked example.

## 3. LLM serving: the judgment layer

The ML System Design course covers what continuous batching and paged attention *are*. Principal-level judgment is knowing when each technique pays and what it costs to adopt.

### Engine and hosting selection

Three viable postures in 2026, and the decision is per-workload, not per-company:

- **Managed API** (frontier or hosted open-weights): wins below roughly $10–20k/month equivalent token spend, for spiky traffic, and for tasks needing frontier quality. Zero ops. You pay 5–20× per token over well-utilized self-hosting and you get no data flywheel from fine-tuning your own weights.
- **vLLM (self-hosted)**: the default open-source engine. Continuous batching, paged KV cache, prefix caching, speculative decoding, broad model support, OpenAI-compatible API. Fastest path from "we have a fine-tuned model" to production. Slightly behind TensorRT-LLM on absolute peak throughput for some model/GPU pairs.
- **TensorRT-LLM (+ Triton)**: squeezes the last 10–40% of throughput on NVIDIA hardware via compiled kernels. Costs you an engine-build step per model/GPU/quantization combo and a smaller talent pool. Justified when GPU spend is large enough that 20% of it exceeds the engineering cost — as an order of magnitude, above ~$100k/month on one workload.

The principal-level default: managed API for the long tail and for product exploration; vLLM for the high-volume fine-tuned paths; TensorRT-LLM only where a single workload's GPU bill justifies the toolchain. Revisit yearly — the engines converge fast.

### The technique inventory, with when-it-pays conditions

- **Continuous batching**: table stakes; 5–10× throughput over static batching at interactive SLOs. You get it free in any modern engine. If any team still serves an LLM with `model.generate()` behind Flask, that is a consolidation target, not a tuning target.
- **Prefix caching**: pays whenever requests share long stable prefixes — system prompts, few-shot blocks, agent preambles, multi-turn history. TTFT reductions of 3–10× on hits. The engineering action is prompt *structure* (stable prefix first, volatile suffix last), which is a code-review standard you can set org-wide for the cost of a doc.
- **Speculative decoding**: 1.5–3× decode speedup when a draft model (or n-gram/Medusa-style heads) predicts the target's tokens well — high acceptance on natural language and code, lower on unusual formats. Costs draft-model memory and tuning time. Adopt on latency-critical, high-volume paths; skip on throughput-oriented batch workloads where you'd rather spend the FLOPs on batch size.
- **Quantization ladder**: BF16 → FP8 → INT4 (AWQ/GPTQ-class). FP8 on Hopper/Ada is the 2026 default: ~2× throughput, near-zero measured quality loss on most tasks. INT4 roughly halves memory again and speeds memory-bound decode, with task-dependent quality risk — fine for casual chat and routing, dangerous for math, code, and long-form extraction.
- **Disaggregated prefill/decode**: separate GPU pools for prefill (compute-bound) and decode (memory-bound), connected by KV-cache transfer. Smooths TTFT under mixed workloads and lets you size pools independently. Pays at large scale — think >50 GPUs on one model family with strict TTFT SLOs; below that, the added system complexity (KV shipping, two autoscaling loops) outweighs the gain.

### The quality-gate protocol for quantization

Never let "we quantized it and it looks fine" through review. The protocol:

```text
1. Freeze an eval set (500–2,000 examples) covering your real task slices,
   including the hard slices (long context, numeric, non-English).
2. Run FP16/BF16 baseline → record per-slice metrics.
3. Run the quantized candidate on identical inputs, temperature 0.
4. Gate: aggregate metric within 1% relative of baseline AND no slice
   degrades more than 3% relative. Slices matter — INT4 losses hide in
   the tails, not the average.
5. Log the comparison artifact in the model registry next to the weights.
   "Which quantization is prod running and what did it cost us?" must be
   answerable in one query.
```

The same protocol gates any serving-side change that can alter outputs: engine upgrades, tensor-parallel degree changes, sampling defaults, speculative-decoding enablement (verify acceptance doesn't skew distributions on your task).

## 4. SLO design for model serving

Principal engineers write the SLOs; seniors meet them. Rules that survive contact with production:

**p50 is for capacity planning; p99 is the SLO.** Users experience the tail. A 300 ms p50 with a 4 s p99 is a bad product; queueing theory guarantees the tail blows up first as utilization rises, so the p99 SLO is what constrains your utilization ceiling (in practice you cannot run a latency-SLO'd GPU service much above 60–70% sustained utilization — see §6).

**Streaming splits latency into TTFT and TPOT.** Time-to-first-token is what the user perceives as responsiveness; time-per-output-token is reading speed. A sane chat SLO: TTFT p99 ≤ 800 ms, TPOT ≥ 30 tok/s sustained. Never write a single "total latency" SLO for a streaming endpoint — a 20-second full generation is fine if the first token lands in half a second and tokens flow faster than reading speed.

**Budget the pipeline, not the model.** For a RAG endpoint with a 1.5 s TTFT target:

```text
auth + routing            50 ms
embedding + vector search 120 ms
reranker                  80 ms
context assembly          10 ms
LLM queue + prefill       900 ms   ← the budget's owner
guardrail (concurrent)    0 ms on the critical path
network + client          150 ms
buffer                    190 ms
```

Publishing the budget converts "the endpoint is slow" arguments into "retrieval is 60 ms over budget" tickets. Each line gets an owner. This is a one-page doc with outsized leverage.

**SLOs are per-tier, not global.** User-facing sync: tight p99. Internal tools: relaxed p99, tighter cost. Batch: throughput and completion-deadline SLOs, no latency SLO at all — which is exactly why batch should never share a GPU pool's SLO class with interactive traffic (run it as preemptible filler to soak idle capacity instead).

## 5. Autoscaling GPU services

GPU autoscaling breaks every CPU-era instinct:

- **Cold starts are minutes, not seconds.** Node provision (30 s–5 min if capacity exists at all) + image pull (LLM images run 10–20 GB) + weight load (a 70B in FP8 is ~70 GB; from remote storage this is minutes) + engine warmup/compile. Realistic cold start for a big model: **5–15 minutes**. Mitigations, in order of cost-effectiveness: bake weights into the image or a pre-attached volume; keep a warm pool of drained-but-loaded replicas; stream weights (Run:ai model streamer, safetensors lazy load); keep scale-down slow and scale-up eager.
- **Scale on queue depth or in-flight concurrency, never CPU.** GPU serving saturates while host CPU idles at 10%. The right signals: request queue length, batch occupancy, KV-cache utilization, TTFT trend. KEDA-on-queue-depth or an engine-aware autoscaler; HPA-on-CPU is a standing incident invitation.
- **Over-provisioning is an insurance premium — price it.** If demand can spike 2× in one minute and cold start is ten minutes, headroom is your only defense. The decision is explicit: N warm replicas × $/hr versus (probability of spike × SLO-breach cost). For a revenue-critical endpoint, 30–50% headroom is normal and correct; for an internal tool, run hot and let the queue absorb spikes. Writing this tradeoff down — with numbers — is the difference between a capacity strategy and a thermostat.
- **Scheduled scaling beats reactive scaling** when traffic is diurnal (most consumer traffic is). Pre-scale 30 minutes before the morning ramp; let reactive scaling handle only the residual.

A policy sketch worth standardizing across the GPU tier, so every service isn't hand-rolling its own:

```text
autoscale policy (GPU inference tier):
  scale-up signal:    queue_depth > 2 × in-flight capacity for 30 s
                      OR TTFT p95 > 0.8 × SLO for 60 s
  scale-down signal:  batch occupancy < 30% for 15 min   # slow down, eager up
  min replicas:       tier-0: peak_forecast × 1.3        # priced headroom
                      tier-2: 0                          # cold start accepted
  schedule:           +N replicas at 07:30 local, 30 min before ramp
  batch filler:       preemptible batch jobs soak idle capacity, evicted on scale-up
```

## 6. Multi-region and failover

Model serving failover has two properties normal services don't: replicas are expensive (a second region of warm GPUs can double serving cost) and "failover" can silently change model behavior (a different region running a different engine version or quantization is a *different model*).

The decision framework, per endpoint tier:

- **Tier 0 (revenue-critical, user-facing):** active-active in 2 regions, each sized to ~70% of global peak (so one region's loss degrades to brownout, not blackout). Version-pin engines and weights across regions; add cross-region output-parity checks to the deploy gate.
- **Tier 1 (user-facing, degradable):** active-passive with a *cheaper* fallback rather than a full replica — fail over to a managed API or a smaller model. A cascade (§7) gives you this for free: the fallback path is just a routing-table change. Quality degrades a measured amount; cost doesn't double.
- **Tier 2 (internal/batch):** queue and retry. Regional GPU capacity shortages (common for H100-class during demand spikes) are a real failover trigger, not just region outages — your "failover" plan is also your "AWS has no capacity in us-east-1 today" plan.

Data residency (Module 01 of the ML System Design course covers the six crossing surfaces) constrains all of this: an EU-pinned workload cannot fail over to us-east-1, so its failover plan must be intra-region redundancy plus a degraded EU-local fallback model.

## 7. Cascades and routing: the primary cost dial

Once the portfolio exists, the single highest-leverage recurring decision is *which requests go to which model*. The cascade formula and the confidence-threshold economics:

```text
blended_cost/req = Σ_i share_i × cost_i
quality(τ)       = share_small(τ) × q_small + (1 − share_small(τ)) × q_big
```

where τ is the confidence threshold controlling escalation. The senior framing is "cascades save money." The principal framing: **τ is a product-owned dial with a dollar value per notch, and someone must own it.** Tighten τ (escalate more): quality up, cost up. Loosen: reverse. Concretely, on a 10M req/day support workload where the small model handles 80% at $0.4/1M output tokens and the frontier tier costs $15/1M, moving escalation share from 20% → 15% saves ≈ $110k/year — and whether that 5% of traffic actually needed frontier quality is an eval question, answerable with a weekly sampled A/B of escalation decisions.

Operational requirements that distinguish a real cascade from a demo: a calibrated confidence signal (logprob-based scores are miscalibrated out of the box — recalibrate on your task); a routing decision log joined to outcomes (so you can audit τ); and a monthly review where the τ owner looks at the quality/cost frontier and moves the dial deliberately. Routing by task type (classifier → model) and by user tier (free vs paid) compose with confidence routing; the design is covered in the ML System Design course, the *governance of the dial* is the principal's job.

## 8. Capacity math you must be able to do live

The self-hosted token cost formula, from Module 01 of the ML System Design course, restated because everything here builds on it:

```text
$/1M output tokens = GPU_$/hr × num_GPUs / (throughput_tok_s × 3600) × 1e6 / utilization
```

Worked, with 2026 planning numbers (state your own when they differ):

```text
Llama-class 8B, FP8, vLLM, 1× H100 @ $2.50/hr, 6,000 tok/s peak decode:
  at 100% util: 2.50 / (6000 × 3600) × 1e6 = $0.116 /1M tokens
  at  50% util: $0.23 /1M
  at  15% util: $0.77 /1M

70B, FP8, TP=4 on 4× H100 ($10/hr), 2,800 tok/s:
  at 100% util: 10 / (2800 × 3600) × 1e6 = $0.99 /1M
  at  40% util: $2.48 /1M   ← now within 2–4× of a managed mid-tier API,
                              before you pay the ops team
```

Two lessons the numbers force. First, **utilization sensitivity dominates everything**: the same hardware is 5× cheaper or more expensive per token depending only on how full you keep it, which is why bin-packing, batch-as-filler, and consolidation beat kernel-level tuning. Second, **always add the people**: a self-hosted GPU tier needs on the order of 1.5–3 FTE (serving infra + on-call share + eval maintenance) ≈ $400–700k/year. Below roughly $50k/month in equivalent API spend on a workload, self-hosting that workload alone rarely pencils; the portfolio move is to self-host a shared platform across many workloads so the FTE cost amortizes.

Capacity sizing in the other direction — from demand to GPUs:

```text
GPUs = peak_QPS × avg_output_tokens / (tok_s_per_GPU × target_util)
     e.g. 40 QPS × 400 tok / (6000 × 0.6) ≈ 4.4 → 5 GPUs + headroom policy
```

Do this arithmetic in meetings, out loud, with stated assumptions. It is the fastest credibility-builder a principal has, and it ends more bad projects than any design review.

## Worked example

**Scenario.** You join a 300-person fintech ("Ledgerline") as its first principal ML engineer. ML grew organically across four teams. Nobody can say what serving costs. Your first-quarter mandate: rationalize serving.

**Step 1 — Inventory (weeks 1–3).** Interviews plus cloud-bill archaeology produce the table:

```text
#  model                 stack              QPS      SLO        hardware        $/mo     $/1k req  on-call
1  fraud-score (XGB)     sklearn+Flask      120      50ms p99   3× c6i.2xl      $760     $0.002    payments
2  txn-categorizer (XGB) sklearn+Flask       45      200ms      2× c6i.2xl      $510     $0.004    payments
3  doc-extract (LayoutLM)custom Torch srv     8      2s         2× A10G          $1,700   $2.5      docs
4  kyc-face (CNN)        custom Torch srv     3      1s         1× A10G          $850     $3.3      risk
5  embed-search (BERT)   custom Torch srv    25      300ms      2× A10G          $1,700   $0.8      search
6  support-assist (8B FT)vLLM               15 peak  TTFT 1s    2× H100 ded.     $7,300   $5.6      platform
7  contract-QA           frontier API       0.5      5s         —                $11,000  $250      nobody
GPU fleet utilization: 12–18%. Distinct deploy pipelines: 6. Rotations: 4 (one is a single person).
Total: ~$23,800/month serving + est. 2.5 FTE maintenance across stacks.
```

The table alone changes the conversation: 46% of serving spend is one low-QPS API workload (#7), and five GPUs run under 20% utilization.

**Step 2 — Target architecture.** Two tiers plus tier-zero API:

- **CPU tier:** Triton (or KServe) serving ONNX-exported models — absorbs #1, #2. One deploy pipeline, one dashboard, autoscaled on request concurrency.
- **GPU tier:** one shared pool — Triton multi-model on 2× A10G absorbs #3, #4, #5 (bin-packed, MIG where helpful); vLLM on 2× H100 keeps serving #6 and gains prefix caching + FP8 (quality-gated per §3) to absorb headroom.
- **Tier zero:** frontier API retained for #7's hardest 20% — the other 80% of contract-QA queries route to the fine-tuned 8B on the vLLM tier behind a confidence threshold, after a 4-week eval shows parity on the routine-clause slice.

**Step 3 — Migration order.** Economic order, not convenience order: (1) #7 cascade first — biggest $ win ($11k → ~$3.4k/month), no infra build required since #6's vLLM tier exists; (2) #3/#4/#5 onto the shared GPU pool — kills three bespoke Torch servers and frees 3 A10Gs ($2,550/month) plus the single-person rotation; (3) #1/#2 onto the CPU tier last — they're cheap and stable, they move for on-call unification, not dollars. Each migration ships behind a shadow-traffic parity gate (mirrored requests, output diff < tolerance for 1 week) before cutover, with instant DNS-level rollback.

**Step 4 — Projected outcome (quarter-end review deck, one slide).**

```text
Serving $:   $23.8k → $13.1k/month  (−45%): API cascade −$7.6k, GPU bin-packing −$2.6k, CPU right-sizing −$0.5k
Ops:         6 pipelines → 2; 4 rotations → 1 platform rotation (6 people, real depth)
             est. maintenance 2.5 FTE → 1 FTE  (~$300k/yr freed)
Quality:     gated — every migration behind frozen-eval parity (§3 protocol) + 1-week shadow diff
Risk:        cascade τ owned by contract-QA PM, reviewed monthly with cost/quality frontier chart
```

Note what made this work: no model was retrained, no kernel was tuned, and the largest single win was a routing decision. That is what serving mastery looks like at this level.

## Exercise

**Task.** Produce a serving-portfolio inventory and consolidation proposal for the following org, as a 2–3 page decision doc plus a spreadsheet.

*MedScribe*, a 400-person clinical-documentation company: (a) an ASR model (custom Torch server, 4× A10G, 20 QPS peak, diurnal 6:1 peak:trough); (b) a medical-NER tagger (sklearn-era CRF on Flask, 30 QPS, CPU); (c) a note-summarization fine-tuned 13B (vLLM, 4× H100 dedicated, TTFT SLO 1.5 s, 8 QPS peak); (d) a coding-suggestion model (frontier API, $28k/month, 1 QPS, quality-critical); (e) two internal embedding services run by different teams (each 1× A10G, <5 QPS); (f) a batch re-summarization job that rents 8 H100s every night at 2 a.m. HIPAA applies: PHI cannot leave the compliance boundary, which currently rules out the frontier API for (d) — the team "scrubs" inputs with a regex script nobody has audited.

**Deliverables.**

1. The inventory table (estimate missing numbers; state every assumption with a sentence of justification).
2. A target two-tier architecture with hardware counts, utilization projections, and autoscaling signals per service.
3. A migration order with rationale per step, quality gates, and rollback story.
4. Projected $/month before/after and FTE/on-call impact.
5. One paragraph on the HIPAA question for (d): options (self-host on the GPU tier, a BAA-covered API, fix and audit the scrubber), your recommendation, and what evidence would change it.

**You're done when:** every model has a row with $/1k req computed; the proposal names the first migration and why it's first; every migration step has an explicit quality gate and rollback; the batch job (f) appears in your utilization plan (hint: it should be soaking your interactive tier's idle capacity, not renting its own fleet); and the doc's projected savings are arithmetic someone else can check, not vibes.

**Self-check questions.**

1. Which single row in your inventory has the highest $/1k requests, and is your first migration aimed at it? If not, what justifies the different order?
2. What utilization did you assume for the consolidated GPU tier, and what happens to your savings number if it comes in 20 points lower?
3. For the 13B summarizer, would FP8 quantization pass your quality gate on medical text? What slices would you check before believing the aggregate metric?
4. Your autoscaler for the ASR service — what signal does it scale on, and what is its cold-start time? Does the diurnal 6:1 ratio call for scheduled pre-scaling?
5. If the nightly batch job preempts interactive headroom and a 3 a.m. traffic spike hits, what breaks first, and does your design detect it before users do?
