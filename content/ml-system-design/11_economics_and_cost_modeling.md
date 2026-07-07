# Module 11 — Economics and Cost Modeling

## Why this module matters

The build-vs-buy argument is decided with math, not intuition — and most junior engineers lose it because they can't produce the numbers fast enough under pressure. "The API is probably fine" is not an answer; "$0.90/1M self-hosted vs $10/1M API, crossover at 60% GPU utilization, which we hit at 300k req/day" is. This module is the consolidated cost reference for the entire course. Every chapter that touches pricing points here rather than duplicating tables; this is the one place to update when vendor prices shift. Read it after the serving and inference chapters — the utilization and throughput numbers from those chapters are inputs to every formula here.

## 1. Cloud GPU pricing reference

**These are representative order-of-magnitude figures as of mid-2026. Check current prices before any build decision — cloud GPU prices have been declining and promotional pricing changes frequently.**

### On-demand prices (per GPU per hour, approximate)

| GPU | Typical cloud hosts | On-demand $/GPU-hr |
|-----|--------------------|--------------------|
| A100 40GB SXM | AWS p4d, GCP a2-standard | ~$2–3 |
| A100 80GB SXM | AWS p4d, GCP a2-ultragpu | ~$3–4 |
| H100 80GB SXM | AWS p5, GCP a3-highgpu, Azure NDH100v5 | ~$6–10 |
| H200 141GB SXM | AWS p5e, GCP a3-ultragpu | ~$10–15 |

The A100-to-H100 jump is 2–3× in price but 2–3× in BF16 throughput (and 3–4× in FP8 throughput with Transformer Engine) — per-token cost is roughly similar between the two tiers once you account for model speed and KV-cache fit. H100 wins when your model barely fits on A100 or when you need FP8 training.

### Spot vs on-demand vs reserved

| Pricing model | Discount vs on-demand | Constraint |
|--------------|----------------------|-----------|
| On-demand | — | None; pay as you go |
| Spot / preemptible | 60–80% cheaper | Can be interrupted; need checkpointing; not for latency-SLO serving |
| 1-year committed-use / reserved | 30–40% off | Locked to region + GPU SKU; right-size carefully |
| 3-year reserved | 50–60% off | Long lock; risky on rapidly-changing GPU tiers |

**Rule of thumb:** training jobs should always target spot/preemptible — the savings compound over a training run lasting days to weeks. Serving clusters should run reserved instances for the baseline (the load you reliably sustain 24/7) and on-demand/spot for burst. Never run inference serving entirely on spot without a fallback: one preemption during a latency-SLO request causes a hard timeout rather than a graceful retry.

### Full-node pricing context

Most cloud instances sell GPUs as nodes: an 8×H100 SXM node at $8/GPU-hr = **$64/hr** ($46k/month on-demand, ~$28k/month at 1-year reserved). That is the planning unit for a 70B+ serving cluster — not per-GPU arithmetic. At 8×A100 and $3.50/GPU-hr: $28/hr = ~$20k/month on-demand.

## 2. API pricing reference

**These are representative ranges for major hosted inference APIs as of mid-2026. Vendor pricing changes; treat these as order-of-magnitude planning numbers and verify before committing to a cost model.**

| Tier | Example model class | Input $/1M tokens | Output $/1M tokens |
|------|--------------------|--------------------|---------------------|
| Frontier | GPT-4o / Gemini Ultra / Claude 3.5-class | $2–15 | $8–60 |
| Mid-tier | GPT-4o-mini / Gemini Flash / Claude Haiku-class | $0.10–0.50 | $0.40–2 |
| Small/hosted open | Hosted Llama-3-8B-class | $0.05–0.20 | $0.10–0.50 |
| Fine-tuned premium | Any tier with vendor fine-tuned weights | 1.5–3× base rate | 1.5–3× base rate |

The input/output asymmetry is consistent across providers: output tokens cost **3–6× more** per token than input tokens because generation is autoregressive (one forward pass per output token) while prefill is batched. Always decompose in your cost model — forgetting the asymmetry underestimates costs by 2–3× on output-heavy workloads. The fine-tuned premium reflects the vendor's per-model infra overhead; whether it's worth it versus self-hosting a fine-tuned open model is precisely the crossover analysis in the next section.

Reasoning model APIs (o1/R1-class) generate long thinking traces before the visible answer; the trace counts as output tokens even when hidden from the user. At 10–50k thinking tokens per hard query at $20–60/1M output, a single reasoning call costs **$0.20–$3.00** — comparable to a human support ticket. Budget explicitly for this; naive cost models miss it entirely.

## 3. The build-vs-buy crossover — worked example

This is the calculation the foundations chapter calls "the crossover argument" and the serving chapter waves at. Here is the full derivation.

### The formula

$$
\text{cost}_{\text{self-hosted per 1M}} = \frac{\text{GPU cluster } \$/\text{hr}}{\text{throughput}_{\text{tok/s}} \times 3600 \times \text{utilization} \times 10^{-6}}
$$

Self-hosting beats API when $\text{cost}_{\text{self-hosted per 1M}} < \text{API output } \$/\text{1M}$. Rearranging for the minimum utilization required to break even:

$$
\text{utilization}_{\text{crossover}} = \frac{\text{GPU cluster } \$/\text{hr}}{\text{throughput}_{\text{tok/s}} \times 3600 \times 10^{-6} \times \text{API output } \$/\text{1M}}
$$

### Worked example: 70B model, one 4×H100 cluster

**Setup.** You are spending $10/1M output tokens on a frontier API for a product that generates 300-token responses. You are evaluating self-hosting a comparable 70B open model on one 4-GPU H100 node, reserved at ~$7/GPU-hr all-in (reserved discount plus amortized networking):

- Cluster cost: 4 × $7 = **$28/hr**
- Throughput: a well-tuned 4-way-TP 70B FP8 replica sustains roughly **1 500 decode tok/s** within interactive SLOs — this is a planning number from the serving chapter; measure your own stack
- Tokens delivered per hour at utilization U: 1500 × 3600 × U = 5.4M × U tokens/hr
- Self-hosted $/1M output: $28 / (5.4 × U) = **$5.19 / U per 1M tokens**

Setting self-hosted cost = API cost to find the crossover utilization:

$$
\frac{\$5.19}{U} = \$10 \quad\Rightarrow\quad U = 0.52 \quad\Rightarrow\quad {\sim}52\%\ \text{GPU utilization}
$$

**Interpretation:** if this 4×H100 node sustains more than ~52% utilization serving output tokens, self-hosting beats the API on pure compute cost. Below 52%, the API is cheaper — and easier, since it comes with no ops burden.

### Converting utilization to QPS

At 52% utilization → 1500 × 0.52 ≈ **780 decode tok/s** required. At 300 tokens/response: **780 / 300 ≈ 2.6 req/s ≈ 225k req/day**. That is the minimum throughput floor that justifies one 4×H100 node against a $10/1M API. Below ~200k req/day, API wins on economics alone.

### What the naive math misses

The utilization calculation is necessary but not sufficient. Four factors routinely flip the decision in ways the spreadsheet doesn't capture:

1. **Utilization is spiky.** Web products rarely sustain 50%+ utilization smoothly — they have 10× intraday variance. Averaging a 52% break-even over a day that peaks at 200% and troughs at 5% gives you a bad average answer. Model the distribution, not the mean; spot autoscaling helps but adds operational complexity.

2. **Ops cost is real.** The self-hosted path requires a serving team (or at minimum one engineer owning it), eval infrastructure to catch model regressions after upgrades, on-call rotations, and upgrade treadmill costs when the base model improves. A conservative estimate for a 2-person part-time burden at $300k loaded engineering cost: **~$50k/month in hidden ops overhead**, which needs to be added to the GPU bill before comparing to the API.

3. **The data flywheel.** Self-hosting means you *own the traffic*: every request and its feedback is yours to log, analyze, and fine-tune on. API usage leaves that data with the vendor or requires explicit logging infrastructure and privacy review. At high volume, proprietary interaction data is the moat — a model continuously improved on your traffic compounds over time in a way that an API relationship does not.

4. **Latency and control.** Managed APIs add a network round trip, are subject to provider-side rate limits and degradations, and cannot be customized below the prompt level. If your SLO requires sub-100ms TTFT or requires a custom inference kernel, self-hosting is required regardless of the cost math.

**Summary rule:** API first, self-host when you hit three of four: (1) token volume clears the utilization crossover, (2) you have traffic data worth training on, (3) you have a serving team or can hire one, (4) latency/control requirements rule out managed APIs.

## 4. TCO worksheet

A one-sheet template for a self-hosted serving decision. Fill in every row; missing rows are usually where the surprise costs are hiding.

| Cost category | Monthly estimate | Notes |
|--------------|-----------------|-------|
| **Compute — on-demand** | $X | Peak-burst capacity above reserved baseline |
| **Compute — reserved** | $Y | Baseline always-on replicas (1-yr or 3-yr) |
| **Compute — spot** | $Z | Training jobs, batch inference (assume 70% of on-demand price) |
| **Storage — model weights + checkpoints** | $A | Typically $0.02–0.05/GB/month on object storage; a 70B FP16 checkpoint ≈ 130 GB |
| **Storage — vector index / feature store** | $B | Scales with corpus; HNSW in-memory vs disk-ANN changes this dramatically |
| **Egress** | $C | ~$0.05–0.10/GB to internet; surprisingly large on RAG systems streaming large documents |
| **GPU interconnect (NVLink/InfiniBand)** | $D | Often bundled; call it out separately for multi-node training |
| **Observability / tracing** | $E | LLM tracing stacks (LangSmith-class) charge per span or per token; budget $500–5k/month depending on volume |
| **Model evaluation infra** | $F | LLM-as-judge eval pipelines consume tokens; estimate from eval cadence × token cost |
| **Ops headcount (loaded)** | $G | At least 0.25 FTE to own a serving stack; 1 FTE for a serious deployment |
| **Tooling / licenses** | $H | Monitoring, secrets management, CI/CD runners; often $1–5k/month at small teams |
| **Total** | **$X+Y+Z+A+B+C+D+E+F+G+H** | |
| **API equivalent** | **tokens/month × blended $/1M** | For direct comparison |
| **Break-even months** | **(TCO setup cost) / (monthly API savings)** | Include migration engineering cost |

The three rows most often left blank on first drafts: egress (non-trivial on document-heavy RAG), observability (token-hungry eval pipelines add up), and ops headcount (zero-ops is a fantasy). Fill all three before declaring self-hosting cheaper.

## 5. Cost per freshness tier (feature platform reference)

The data-engineering chapter defers here for operating-cost comparisons across freshness tiers. The rule: **each tier costs roughly an order of magnitude more to operate than the previous one.** Here is why, with the line items.

| Freshness tier | Latency of feature update | Primary infrastructure | Relative monthly cost per feature |
|----------------|--------------------------|----------------------|----------------------------------|
| **Batch / static** | Hours to daily | Spark/BigQuery job + object storage | 1× (baseline) |
| **Near-real-time** | Minutes (micro-batch) | Flink or Spark Structured Streaming + Kafka + low-latency online store (Redis/DynamoDB) | ~5–15× |
| **Real-time / streaming** | Seconds | Flink sliding-window aggregates over Kafka, low-latency online store with sub-10 ms p99 | ~20–50× |

The cost drivers that cause the jump:
- **Compute:** streaming jobs run 24/7; batch jobs run once per cycle. A daily Spark job that runs for 2 hours costs 2 GPU-hours/day; an equivalent Flink streaming job costs 24 GPU-hours/day — 12× before optimization.
- **Storage write amplification:** streaming writes individual records continuously; object storage and columnar stores are optimized for bulk writes. Real-time writes to Redis/DynamoDB at high cardinality generate far more I/O ops, and cloud databases bill per I/O.
- **Operational complexity:** streaming topologies require backpressure tuning, consumer lag monitoring, and schema-evolution discipline; the engineering burden is real and does not amortize quickly across few features.
- **Online store cost:** an in-memory Redis cluster serving low-latency lookups at scale costs ~$0.50–2/GB/month in managed form (ElastiCache, MemoryDB) — an embedding table at 1 billion items × 128 floats × 4 bytes ≈ 512 GB → $250–1k/month just for the online store.

**Decision rule:** default to batch for any feature whose value doesn't degrade meaningfully on a day-old snapshot; instrument and measure the model quality delta before upgrading to streaming. The cost jump is almost never worth it for static user attributes (account age, device type) and almost always worth it for velocity features (transactions in the last 5 minutes, recent click sequence for recsys). State this reasoning explicitly in any feature-platform interview question.

## Going deeper

- Cloud pricing calculators (AWS, GCP, Azure) are the authoritative source — this chapter's numbers are representative baselines, not quotes. Run the calculator with your exact GPU count, region, commitment term, and egress assumptions before a real build decision.
- The serving chapter's capacity planning section (the "10k concurrent users" worked example) is the upstream calculation that feeds into the GPU count here — the two calculations should be done together.
- Operator cost is consistently the most underestimated line item in junior engineers' TCO models; reading public post-mortems from teams who tried to self-host and then migrated back to APIs reveals the hidden costs that spreadsheets miss.
- For training cost specifically: the training chapter's FLOPs arithmetic (6ND rule, MFU estimation) translates directly to GPU-hours, which translates to dollars at the spot rates above.

## Project 11 — Build a cost model for a real product

Pick any system you have designed in a previous module's project (support assistant, RAG pipeline, serving benchmark). Produce a one-sheet TCO using the worksheet above for both the API path and the self-hosted path at three traffic levels: 10k, 100k, and 1M req/day. For each level, state whether the economics favor API or self-hosting and why. Plot the monthly cost of each path as a function of req/day and mark the crossover. Include the ops overhead estimate as a separate line and show how it moves the crossover. Stretch: add a third column for a hybrid cascade (mid-tier API for easy requests, self-hosted small model for the high-volume narrow path) and show that the cascade is almost always the optimal shape at moderate scale.

## Interview Q&A

**Q1. At what request volume does self-hosting a 70B model beat a frontier API, and what does the math look like?**

**A.** State the formula first: self-hosted $/1M = cluster $/hr ÷ (throughput_tok_s × 3600 × utilization × 10⁻⁶). Then plug in numbers: a 4×H100 node at ~$28/hr running a well-tuned 70B FP8 model at ~1 500 decode tok/s has a break-even utilization of $28 / (5.4M × U) = API $/1M. Against a $10/1M API, U_crossover = 52% — achieved at roughly 780 decode tok/s ≈ 2.6 req/s ≈ 225k req/day of 300-token responses. Below that volume, idle GPUs make self-hosting more expensive. Then name what the naive math misses: utilization is spiky (a day-average of 52% can still mean under-utilization most hours), ops headcount adds $30–50k/month hidden cost, and the data-flywheel argument matters more at scale than the per-token spread. The complete answer is: API below ~200–500k req/day; self-host once traffic sustains GPU utilization above ~50–60% and you have a team to operate it.

**Q2. A product manager asks why real-time features cost so much more than batch features. Give the engineering explanation.**

**A.** Three reasons. First, compute stays on 24/7: a batch job runs for 2 hours daily, but an equivalent Flink streaming job consumes compute every hour of the day — an order-of-magnitude more GPU/CPU hours per feature. Second, online store cost: real-time features require a sub-10 ms lookup store (Redis, DynamoDB) rather than object storage; in-memory storage costs $0.50–2/GB/month in managed form, versus cents/GB for object storage. A realistic embedding table — say 100M users × 64 float features — is ~25 GB at $50/month in Redis, versus a negligible batch read cost from Parquet. Third, operational burden: streaming topologies demand consumer-lag monitoring, schema-evolution discipline, and on-call coverage, which is an engineering-headcount cost that doesn't appear in the infrastructure bill but is real. The rule of thumb: each freshness tier costs roughly 10–20× more to operate than the previous one; instrument and confirm the model quality delta before upgrading, because for static features the delta is usually small and the cost is not.

**Q3. Your team is debating fine-tuned API vs self-hosted fine-tuned open model. Walk through the decision framework.**

**A.** Four axes. (1) Token economics: compute the crossover utilization as above; self-hosting wins only above ~50–60% sustained GPU utilization, which typically means 200k+ req/day for a mid-size cluster. Fine-tuned API premium is 1.5–3× the base rate, which raises the crossover — the math may still favor self-hosting at lower volume if the premium is steep. (2) Data residency and IP: fine-tuning an API means sending proprietary training data to the vendor; some teams are blocked by security or legal review. Self-hosting keeps both the training data and the resulting weights in-house. (3) Quality ceiling: fine-tuned open models are often competitive with fine-tuned API models for narrow tasks after enough task-specific data — but starting from a weaker base model and closing the gap requires genuine fine-tuning expertise (SFT → DPO or RLVR, eval-driven iteration), not just uploading a JSONL file. (4) Ops reality: a self-hosted fine-tuned model requires serving infrastructure, regression testing on each base-model upgrade, and someone to own the model quality over time. The standard 2026 answer: prototype on the API to confirm product-market fit; collect interaction data; fine-tune an open model for the high-volume narrow path once you have both the traffic and the data to improve the model; keep the API for the long-tail edge cases the narrow model handles poorly.

## You can now

- compute the self-hosted \$/1M-token cost and the break-even GPU utilization for a given cluster cost, throughput, and API price from memory.
- convert a utilization crossover into a QPS and requests/day threshold, and state the traffic volume at which self-hosting overtakes an API.
- name the four factors the naive crossover math misses — spiky utilization, hidden ops headcount, the data flywheel, and latency/control requirements — and adjust the build-vs-buy decision accordingly.
- fill a full TCO worksheet and flag the three rows juniors leave blank (egress, observability, ops headcount).
- explain why each feature-freshness tier costs roughly an order of magnitude more than the previous one, and decide when the streaming upgrade is actually worth it.
