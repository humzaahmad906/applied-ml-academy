# Module 05 — LLM Serving Systems

## Why this module matters

"How would you serve this model to 100k concurrent users?" is now a standard interview deep-dive, and serving is where the 2023→2026 evolution has been most dramatic: from "wrap the model in FastAPI" to a specialized systems discipline with its own papers, engines, and architecture patterns. The good news: it's built from a small number of ideas, all downstream of the prefill/decode asymmetry from the inference chapter.

## 1. The core engine ideas (in the order they arrived)

- **Continuous (in-flight) batching** (Orca, 2022): schedule at the *iteration* level, not the request level — new requests join the running batch the moment any sequence finishes, instead of waiting for the whole batch to drain. This alone gave order-of-magnitude throughput gains over static batching and is table stakes in every engine.
- **PagedAttention** (vLLM, SOSP 2023): manage KV cache like virtual memory — fixed-size blocks allocated on demand with an indirection table — eliminating the fragmentation and worst-case preallocation that wasted most KV memory. More usable KV memory → bigger batches → more throughput.
- **Prefix caching / RadixAttention** (SGLang, 2024): store KV blocks in a radix tree keyed by token prefix; requests sharing a prefix (system prompts, few-shot blocks, multi-turn history, agent loops) skip recomputing it. For agentic workloads, **KV-cache hit rate is now a primary design metric** — prompts are deliberately structured (stable preamble first, volatile content last) to maximize it.
- **Chunked prefill** (Sarathi lineage): split long prefills into chunks interleaved with decode steps, so one giant prompt doesn't stall every other request's token cadence — fixes tail ITL without extra hardware.
- **Structured/constrained decoding:** grammar-constrained generation (xgrammar, Outlines; compressed-FSM tricks) masks invalid tokens each step → guaranteed-valid JSON/schemas at near-zero overhead. Production tool-calling depends on it.
- **Multi-LoRA serving** (S-LoRA/Punica lineage): one base model + hundreds of hot-swapped adapters batched together — the standard way to serve many fine-tuned variants cheaply.

## 2. Prefill/decode disaggregation — the 2026 architecture

Prefill is compute-bound, decode is bandwidth-bound; co-locating them makes them interfere (a long prefill stalls everyone's decode). **Disaggregation** runs separate prefill and decode worker pools, transferring the KV cache between them, so each pool scales and parallelizes independently and tail latencies become controllable. The idea evolved quickly: first a goodput-optimal formulation, then KV-cache-centric architectures built around a distributed KV store; by 2025–26 it went mainstream — the major serving engines (vLLM, SGLang) ship it natively, dedicated orchestration layers (NVIDIA Dynamo, and Kubernetes-native equivalents like llm-d) provide KV-aware routing, KV transfer over RDMA-class paths, and autoscaling above any engine, with large deployments running disaggregated serving in production.

The engineering crux is **KV transfer**: a 70B-class model accumulates ~0.3 MB/token, so a 4k prompt ≈ >1 GB that must move prefill→decode within the latency budget — demanding NVLink/RDMA-class interconnect or smart placement. Related: **KV-cache-aware routing** (route a request to the replica already holding its prefix) and multi-tier KV storage (HBM→DRAM→SSD, LMCache lineage) for long-conversation reuse.

Decision rule worth stating in interviews: chunked prefill + prefix caching solves most workloads on co-located instances; disaggregate when you have long prompts + strict ITL SLOs + scale to justify the infrastructure.

## 3. Engines and the serving stack

- **vLLM** — broadest model/hardware support, the default. **SGLang** — RadixAttention lineage, exceptionally strong on prefix-heavy/structured/agentic workloads and large-scale MoE deployments. **TensorRT-LLM** — peak NVIDIA performance at the cost of compilation rigidity. All three now offer the same headline features (continuous batching, paged KV, quantization, spec decode, prefix caching, disaggregation); differentiation is workload- and ops-fit.
- Above the engine: an **orchestration layer** (Dynamo, llm-d, or a routing proxy like LiteLLM/custom) handling replica routing, autoscaling, failover, and multi-model/multi-tenant policy. **MoE serving** adds wide expert-parallel deployments (many GPUs serving one giant sparse model, all-to-all token routing) — DeepSeek-V3-class models made this a standard pattern.
- **SLO vocabulary:** TTFT p50/p99, ITL/TPOT p99, and **goodput** — throughput counting only requests that met both SLOs. Optimizing raw throughput while violating ITL is the classic rookie chart.

## 4. Serving reasoning models

o1/R1-class models — and their successors — change the serving economics in a way that many interviewers are now probing explicitly. The shift is from **prefill-dominated** to **decode-dominated** workloads: a reasoning model emits a thinking trace before producing its visible answer, and that trace runs 10–100× longer than the final response. A query that costs 500 output tokens on a standard model might cost 5 000–50 000 on a reasoning model, with the thinking trace constituting 90%+ of the output budget. The decode phase, already the bottleneck in standard serving, becomes the dominant cost and latency driver.

**Thinking budgets as an API surface.** Modern reasoning model APIs expose budget controls — `max_completion_tokens`, "reasoning effort" knobs, and in research-style deployments, **"Wait"-token budget forcing** (appending "Wait" to the assistant turn forces the model to continue deliberating before concluding). These parameters are first-class serving knobs: tighter budgets reduce cost and latency but sacrifice answer quality on hard questions. The per-product design question is what budget to set per query class — and that requires understanding the accuracy/cost curve empirically, not guessing.

**Routing reasoning vs fast models by query hardness.** The right architecture is almost never "always use a reasoning model." Instead: a lightweight classifier (or the model's own confidence/perplexity on a short fast pass) estimates query difficulty — math proofs and multi-step logic go to reasoning models; factual recall and simple extraction go to the cheap fast path. This is the cascade pattern (defined below) applied along a reasoning/non-reasoning axis. Calibrating the classifier is the hard part: miscalibration in the "send everything to reasoning" direction destroys unit economics; in the other direction, it degrades quality on the hard queries that most need it.

**The test-time-compute paradox.** More reasoning tokens do not monotonically improve accuracy. On easy queries, extended thinking traces can **degrade accuracy** — the model second-guesses correct initial answers, introduces spurious intermediate steps, or overcomplicates simple lookups. This is the "overthinking" failure mode studied in multiple 2025 papers. Adaptive budget methods (DynaThink, ST-BoN lineage) dynamically allocate reasoning compute per query rather than applying a flat budget — the serving infrastructure must support variable-length decode budgets and per-request early-stopping signals. For interview purposes: stating "I'd route easy queries away from reasoning models and use adaptive budgets on the remainder" is the senior answer; stating "I'd use the reasoning model for everything" signals unfamiliarity with the economics.

**Interview angle:** if the question is "design serving for a math-tutor product," the 2026-standard answer expects you to: (1) distinguish reasoning model queries (multi-step proofs) from non-reasoning queries (formula lookup), (2) route them to separate pools or at least separate budget configurations, (3) name that the decode phase dominates capacity planning, so your GPU count estimate is 10–50× higher than for a comparable non-reasoning workload at the same QPS, and (4) discuss thinking-budget guardrails to prevent tail latencies from blowing out when users submit hard problems.

## 4a. The GenAI gateway pattern

Above the serving engine sits a layer that most mature teams have built and that junior candidates rarely mention: the **GenAI gateway**. The now-standard pattern is a single internal API surface that fronts multiple backends — hosted frontier APIs, cloud-managed models, self-hosted open models, and internal fine-tuned models — with unified interfaces for all consumers inside the company.

What the gateway provides: **failover** (if the primary provider returns a 429 or goes down, the gateway retries against a secondary without the calling service knowing); **rate limiting and quota enforcement** per team or use case; **per-use-case cost tracking** (so the finance team can chargeback model costs to the product teams consuming them — otherwise costs are invisible until the bill arrives); **A/B routing** at the model level (send 5% of traffic to a new model variant); and **policy enforcement** (content filtering, PII scrubbing, audit logging in one place rather than in every calling service).

The gateway is not a serving engine — it does not do batching, KV management, or quantization. It sits above all of that, as an HTTP proxy with routing logic. Think of it as the service mesh of the model layer. In interviews, naming it demonstrates that you've thought about multi-model operations, not just single-model serving. It is also the natural home for the cost tracking and cascade routing discussed below.

## 4b. Cascades and model routing as a named pattern

Cascades appear in the cost math of the foundations chapter, the document-extraction design in the interview playbook, the reranking funnel in the retrieval chapter, and the recommendation pipeline in the classic-ML chapter. They deserve a named, reusable definition.

**The cascade pattern:** route requests through a sequence of models of increasing capability and cost, escalating only when the cheaper model's confidence falls below a threshold. The mechanics:

1. A **fast, cheap model** handles the request and produces a confidence score along with its output.
2. If confidence ≥ threshold → serve the cheap result.
3. If confidence < threshold → escalate to the next tier.
4. Repeat up to a frontier model or human review.

**Confidence calibration is the routing currency.** The cascade works only if the cheap model's confidence scores are *calibrated* — a confidence of 0.8 should mean the model is right ~80% of the time, not 50% or 95%. Uncalibrated confidence turns the cascade into a random splitter. Calibration methods: temperature scaling (a single scalar applied to logits post-training, usually sufficient for classification heads), Platt scaling, or held-out calibration sets with isotonic regression. The calibration check is part of the eval harness, not an afterthought.

**What to set the threshold at:** this is a product/business decision, not purely an ML decision. It encodes: how much quality degradation is acceptable on the cheap path, what the cost ratio between tiers is, and what the acceptable error rate on the escalated tail is. State this explicitly in an interview — it shows you understand that the cascade is a joint engineering-product-finance decision.

**Reuse the pattern across contexts:** document triage, query routing in RAG (cheap dense retrieval → expensive cross-encoder), agent tool selection (local lookup → API call → human), and reasoning-vs-fast model routing are all the same cascade shape. Naming it once and applying it everywhere is the senior move.

## 5. Long-context serving

1M-token context windows are now a product feature, not a research artifact. They introduce a serving problem that doesn't exist at 4k or 32k context, and interviewers at companies building document-processing or agent-loop products are starting to probe it.

**The memory problem.** KV cache at 1M tokens for a 70B-class model is roughly 320 KB/token × 1M = **~320 GB per user session** in FP16 — larger than the model weights themselves and far exceeding the HBM of any single GPU. Even a 7B model at 1M context accumulates ~2–4 GB of KV per user. Multi-user serving with long-context sessions saturates memory before it saturates compute.

**The prefill latency problem.** Prefilling 1M tokens is a compute-bound operation that can take **minutes** on current hardware, even with FlashAttention. A user uploading a 500-page PDF and asking a question cannot wait 2 minutes for the first response token. This drives the need for **chunked prefill** (interleave prefill chunks with ongoing decode, so other users' ITL doesn't spike) and **async prefill** (fill the KV cache in the background before the user asks, if the document is known in advance).

**Mitigations, in order of maturity:**

- **Prefix caching:** if multiple users query the same long document, compute its KV once and share it. Effective when the corpus is shared (code repos, policy documents, knowledge bases); useless when each session has a unique personal context.
- **Chunked prefill:** prevents one long document from blocking all other requests' decode steps — standard in all major engines, should always be on for long-context workloads.
- **Context parallelism / ring attention:** split the long sequence across multiple GPUs, each attending to a ring-passed slice. This is the inference-time equivalent of sequence parallelism in training. Required when the prefill doesn't fit in one GPU's compute budget at interactive latency.
- **Tiered KV storage (HBM → DRAM → SSD):** hot KV blocks live in HBM; cold/idle-session blocks are evicted to DRAM or NVMe and paged back on demand. Several production KV-store implementations offer this multi-tier model. Effective for long-running sessions with idle periods (e.g., a user who opened a 200k-token document but is reading slowly).
- **RAG as the alternative:** the RAG-vs-long-context decision is not primarily about capability — it is a **cost and quality trade**. RAG costs prefill only for the retrieved chunks (hundreds of tokens) rather than the whole corpus. The caveat: "lost in the middle" degradation means long-context answers can be *worse* than targeted retrieval answers when the relevant information is buried in the middle of a million-token context. Longer context ≠ better answers. The synthesis: retrieval narrows to the relevant passages; long context lets you be generous with those passages without having to over-chunk.

**Reference architecture: KV-cache-centric disaggregation.** The most capable long-context serving designs, proven in production systems handling over 100B tokens per day, push disaggregation one step further: **disaggregate not just prefill from decode, but also separate the KV cache into a pooled, multi-tier distributed store** independent of the compute nodes. KV blocks are stored in a cluster-wide pool (HBM + DRAM + SSD tiers), routed by prefix hash, and transferred to whatever compute node needs them via a high-bandwidth interconnect. Requests are routed to nodes that already hold the relevant KV prefix (cache-centric routing). Measured results in these systems range from 59% to nearly 500% capacity gains depending on traffic mix, with the largest gains on workloads with high KV reuse. This is the disaggregated-prefill-decode idea from the earlier disaggregation section, extended to multi-tier storage and pooled KV — the pattern to reach for when doing long-context inference at scale.

**Numbers to have ready:** a 1M-token session costs roughly $1–10 in prefill compute on a frontier API (depending on model size and provider); amortized over a multi-turn conversation, the per-turn cost is dominated by the initial fill. Prefix caching can reduce this to near-zero for the second and subsequent turns if the document hasn't changed — which is why cache-aware session design (stable document prefix, appended conversation) is economically critical, not just a performance trick.

## 6. Capacity planning & cost math (interview gold)

Worked example — "serve a 70B chat model, 10k concurrent users": assume each active user generates a request every ~30 s, 1k-token prompts, 300-token responses → ~330 req/s, ~110k decode tok/s + 330k prefill tok/s. On H100-class hardware with a well-tuned engine, a 4-way-TP 70B replica sustains roughly 1.5–3k decode tok/s within interactive SLOs (order-of-magnitude planning number — measure your own) → ~40–70 replicas of 4×H100 before caching; system-prompt prefix caching might cut prefill cost 50–80%. Then $/1M tokens = (GPU-$/hr × GPUs) / (tok/s × 3600 / 10⁶) — practice producing this chain fluently, stating every assumption.

Cost levers ranked: quantization (FP8 weights+KV ≈ near-free 1.5–2×), prefix caching (workload-dependent, can be huge), batching/goodput tuning, cheaper hardware per phase (disaggregation enables compute-heavy GPUs for prefill, bandwidth-heavy for decode), spec decode for latency-bound low-batch services, and cascades (small model first, escalate hard cases).

The full API-vs-self-host TCO crossover math, per-provider pricing tables, and the build-vs-buy decision model live in the economics chapter (module 11) — the capacity numbers above are inputs to that analysis, not a replacement for it. Container builds, GPU node pools, Kubernetes autoscaling policies, and production monitoring infrastructure for the serving layer are covered in the deployment chapter (module 12).

## Tool Survival Guide: vLLM

**Architecture internals.** vLLM's `LLMEngine` is the top-level coordinator: it accepts requests, runs the `Scheduler` each forward-pass iteration, and dispatches work to one or more `Worker` processes. The `Scheduler` decides — at every decode step — which sequences to prefill, which to continue decoding, and which to preempt (swap to CPU or recompute later); it enforces `--max-num-seqs` and the per-iteration token budget. Each `Worker` maps to one GPU (or one tensor-parallel rank), owns the model shards, and manages its assigned KV-cache blocks. The `BlockSpaceManager` (the KV-cache manager) maintains per-sequence block tables, allocates physical blocks on demand, shares prefix blocks copy-on-write, and triggers preemption when free blocks run low. This decomposition answers most tuning questions: the scheduler is the bottleneck when `--max-num-seqs` is too low; the block manager is the OOM trigger when `--gpu-memory-utilization` is set too aggressively or `--max-model-len` is unnecessarily large.

**Key launch flags** (representative as of 2026 — verify against current vLLM docs before deploying):

| Flag | Default | What it controls | Tuning guidance |
| ------ | --------- | ----------------- | ----------------- |
| `--max-num-seqs` | 256 | Maximum concurrent sequences in the scheduler | Raise if GPU utilization is low and memory headroom allows; lower first if you see OOM under sustained load |
| `--max-num-batched-tokens` | auto | Total token budget per forward pass (prefill + decode combined) | With chunked prefill this is the chunk-size ceiling — 512–2048 bounds TTFT p99 on mixed workloads |
| `--gpu-memory-utilization` | 0.90 | Fraction of HBM reserved for KV-cache blocks after model weights | OOM despite 0.90 → lower to 0.85; activations need headroom too — never set to 1.0 |
| `--enforce-eager` | False | Disables CUDA graph capture; runs eager mode | Use only when debugging CUDA errors; costs ~10–20% throughput — remove before any benchmark |
| `--max-model-len` | model default | Maximum total sequence length (prompt + completion) | Shorter = more KV blocks available = larger effective batch; must be ≥ your longest expected request |
| `--enable-chunked-prefill` | False | Interleaves prefill chunks with decode iterations | Turn on for any mixed short/long workload; pair with `--max-num-batched-tokens` to control chunk size |
| `--enable-prefix-caching` | False (auto-on in recent builds) | Automatic radix-tree KV prefix caching | Verify it is active; required for agentic and multi-turn workloads to see cache benefit |
| `--kv-cache-dtype` | auto | Data type for stored KV blocks | `fp8` roughly halves KV memory, enabling ~2× larger batches at near-identical output quality |

**Tuning for three common workloads.** Chat / short prompts (≤1k tokens, latency-sensitive): start with `--max-num-seqs 256 --gpu-memory-utilization 0.90 --enable-prefix-caching --kv-cache-dtype fp8`. If TTFT p50 is fine but ITL p99 drifts under load, the decode batch is compute-saturated — lower `--max-num-seqs` until ITL stabilizes. If GPU utilization is low, raise it. Long-context / RAG (3k–32k prompts): add `--enable-chunked-prefill --max-num-batched-tokens 2048`; ITL p99 is the primary signal. Set `--max-model-len` to what your workload actually needs — unnecessary length headroom consumes KV blocks that could serve more sequences. Agentic / multi-turn: pin sessions to the same replica or use a shared KV store; system-prompt bytes must be bit-identical across every step. Track the `vllm:gpu_prefix_cache_hit_rate` Prometheus metric — below 0.50 on a workload with a shared preamble is a routing problem, not a serving problem.

**Failure-mode table.**

| Symptom | Likely cause | Fix |
| --------- | ------------- | ----- |
| OOM at startup or under load despite `--gpu-memory-utilization 0.90` | Activation memory + KV reservation exceeds HBM | Lower to 0.85; reduce `--max-model-len`; confirm `--enforce-eager` is off (graph capture uses extra memory at init time) |
| GPU utilization <50%, low throughput | Batch chronically too small — scheduler not filling the forward pass | Raise `--max-num-seqs`; verify `--max-num-batched-tokens` is not artificially capping the iteration |
| TTFT p99 spikes on mixed short/long traffic | A long prefill request blocking the decode queue | Enable `--enable-chunked-prefill --max-num-batched-tokens 1024`; watch TTFT p99 drop |
| Prefix-cache hit rate near zero on a workload with a shared preamble | Session-affine routing broken, or preamble bytes non-deterministic | Switch to hash-based routing on session ID; audit system-prompt for injected timestamps, UUIDs, or trailing-space drift |
| Throughput regresses after adding a second GPU | AllReduce overhead exceeds the parallelism gain | Only use tensor parallelism when the model does not fit on a single GPU; 2-way TP on a model that fits on 1 GPU often reduces throughput |

## Operations

Concrete alert thresholds for a production LLM serving system — calibrate the exact values against your own p50 after a stable week of traffic, but these are the right quantities to instrument from day one.

| Metric | Alert threshold | What it indicates |
| -------- | ---------------- | ------------------- |
| TTFT p99 | > 2× 7-day baseline | Scheduler congestion, swap/recompute cascade under memory pressure, or KV-transfer latency spike in disaggregated setups |
| ITL p99 | > 2× 7-day baseline | Decode batch saturating memory bandwidth; concurrent sequence count or batch token budget needs tuning |
| KV-cache hit rate | < 0.40 | Session-affine routing degraded, or system-prompt bytes are non-deterministic (timestamps, UUIDs injected into the preamble) |
| Cost per request | > 2× 7-day rolling baseline | Runaway context growth (agent loop not compacting), reasoning-model misrouting, or cascade confidence threshold drifted |
| Scheduler queue depth | > 30 s of estimated service time | Replicas undersized for current load — trigger autoscale or rate-limit upstream before latency SLOs break |

Each alert should carry: metric value, affected replica set, request volume in the last 5 minutes, and whether prefix-cache hit rate changed concurrently. The KV-cache hit rate alert is the easy-to-miss one — it degrades cost silently without spiking latency until cache thrash forces swap/recompute and the latency alert fires too.

## Going deeper

- The four engine ideas — continuous (in-flight) batching, paged KV allocation, radix-tree prefix caching, and chunked prefill — are the foundation; study a mature open-source engine's implementation of each to see how they compose.
- Prefill/decode disaggregation and its goodput-optimal formulations are the current architectural frontier; the KV-transfer problem (gigabytes per long request across the interconnect) is the crux to understand.
- Reasoning-model serving flips the economics from prefill-dominated to decode-dominated; the thinking-budget knobs and the "overthinking" failure mode are both worth understanding empirically rather than by rule of thumb.
- The best way to internalize serving is to benchmark one: the Project below has you produce a latency-throughput frontier and a $/1M-token figure on real hardware.

## Project 05 — Benchmark a serving stack properly

Serve an 7–8B instruct model with vLLM on whatever GPU you have. (1) Build a load generator (or use `vllm bench serve` / genai-perf) replaying a realistic mixed workload: 70% short chat (300-token prompt), 30% long RAG-style (3k-token prompt), Poisson arrivals. (2) Produce the **latency–throughput curve**: sweep arrival rate, plot TTFT p50/p99 and ITL p99 vs achieved throughput; identify the goodput knee under SLOs of TTFT<1.5 s, ITL<80 ms. (3) Ablate: prefix caching on/off (give all requests a shared 800-token system prompt and report hit rate + TTFT delta), chunked-prefill on/off (watch ITL p99 under the long-prompt mix), FP8 KV cache on/off (max batch size delta). (4) Compute $/1M output tokens at the knee using a cloud GPU price. (5) Stretch: repeat on SGLang and explain any differences you observe. Write it up as if for an internal platform decision — this exact artifact is portfolio gold.

### Step-by-step implementation

**Step 1 — Launch the server.**

```bash
# Representative flags as of 2026 — verify against current vLLM docs.
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --dtype bfloat16 \
    --port 8000
```

Confirm it is healthy: `curl -s http://localhost:8000/health` should return `{"status":"ok"}`. Check startup logs for "KV cache dtype: fp8" and "Prefix caching enabled: True" before starting the sweep — a misconfigured server silently runs without caching and your ablation results are garbage.

**Step 2 — Run the load generator.**

Save as `load_generator.py`. Drives Poisson arrivals at a configurable rate, mixes 70% short / 30% long prompts, and measures TTFT and mean inter-token latency from the streaming SSE response. Install deps first: `pip install httpx numpy`.

```python
"""load_generator.py — asyncio Poisson load generator for a vLLM OpenAI-compat endpoint."""
import asyncio
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Stable 800-token system preamble — identical bytes on every request so the
# radix cache can share KV blocks across the entire run. Volatile content (the
# user turn) is appended last; the preamble must never contain timestamps or
# per-request IDs, which would defeat caching entirely.
SYSTEM_PROMPT = "You are a helpful assistant. " * 60  # approximately 800 tokens

SHORT_PROMPTS = [
    "What is the boiling point of water in Celsius?",
    "Name the three branches of the US government.",
    "What is the time complexity of binary search?",
    "Define gradient descent in one sentence.",
    "What does the acronym REST stand for?",
]

LONG_PROMPTS = [
    (
        "Explain the transformer architecture in detail, covering self-attention, "
        "multi-head attention, positional encoding, the feed-forward sublayer, and "
        "how these components interact during both training and inference."
    ),
    (
        "Walk through the steps for fine-tuning a pre-trained language model on a "
        "domain-specific classification task — from dataset preparation through final "
        "evaluation — and name the most common failure modes at each stage."
    ),
]

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass
class RequestResult:
    ttft: float         # seconds to first token
    itl_mean: float     # mean inter-token latency (seconds); per-request mean, not a global percentile
    total_time: float
    prompt_type: str    # "short" | "long"
    error: str = ""


async def send_request(
    client: httpx.AsyncClient,
    prompt: str,
    prompt_type: str,
) -> RequestResult:
    t0 = time.perf_counter()
    first_token_at: Optional[float] = None
    token_times: List[float] = []

    try:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 150,
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                token_times.append(now)
    except httpx.HTTPStatusError as exc:
        return RequestResult(0.0, 0.0, 0.0, prompt_type, error=str(exc))
    except httpx.RequestError as exc:
        return RequestResult(0.0, 0.0, 0.0, prompt_type, error=str(exc))

    total = time.perf_counter() - t0
    ttft = (first_token_at - t0) if first_token_at is not None else total
    gaps = [b - a for a, b in zip(token_times[:-1], token_times[1:])]
    itl_mean = statistics.mean(gaps) if gaps else 0.0
    return RequestResult(ttft=ttft, itl_mean=itl_mean, total_time=total, prompt_type=prompt_type)


async def run_load(
    arrival_rate: float,
    duration: float = 60.0,
    short_fraction: float = 0.70,
) -> List[RequestResult]:
    """Poisson arrivals at arrival_rate req/s for duration seconds."""
    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        tasks = []
        while time.perf_counter() - start < duration:
            await asyncio.sleep(random.expovariate(arrival_rate))
            is_short = random.random() < short_fraction
            prompt_type = "short" if is_short else "long"
            prompt = random.choice(SHORT_PROMPTS if is_short else LONG_PROMPTS)
            tasks.append(asyncio.create_task(send_request(client, prompt, prompt_type)))
        return list(await asyncio.gather(*tasks))


async def sweep(rates: List[float], duration: float = 60.0) -> List[dict]:
    rows = []
    for rate in rates:
        logger.info("arrival_rate=%.1f req/s — running for %.0fs", rate, duration)
        results = await run_load(rate, duration)
        good = [r for r in results if not r.error]
        if not good:
            logger.warning("All requests failed at rate=%.1f — skipping", rate)
            continue
        ttfts = [r.ttft for r in good]
        itls = [r.itl_mean for r in good if r.itl_mean > 0]
        row = {
            "target_rate": rate,
            "achieved_throughput": len(good) / duration,
            "ttft_p50": float(np.percentile(ttfts, 50)),
            "ttft_p99": float(np.percentile(ttfts, 99)),
            "itl_p99": float(np.percentile(itls, 99)) if itls else 0.0,
            "n_requests": len(good),
            "n_errors": len(results) - len(good),
        }
        logger.info(
            "  TTFT p99=%.3fs  ITL p99=%.0fms  throughput=%.1f req/s",
            row["ttft_p99"], row["itl_p99"] * 1000, row["achieved_throughput"],
        )
        rows.append(row)
    return rows


if __name__ == "__main__":
    rates = [1, 2, 4, 8, 12, 16, 20, 25, 30]  # req/s — adjust to your hardware
    data = asyncio.run(sweep(rates, duration=60.0))
    with open("sweep_results.json", "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results written to sweep_results.json")
```

**Step 3 — Measure prefix-cache hit rate.**

Query vLLM's Prometheus endpoint during or after a run. Save as `prefix_cache_hit.py`:

```python
"""prefix_cache_hit.py — read GPU prefix-cache hit rate from vLLM's Prometheus endpoint."""
import logging

import httpx

logger = logging.getLogger(__name__)


def get_prefix_cache_hit_rate(
    metrics_url: str = "http://localhost:8000/metrics",
) -> float:
    """
    Parse vLLM's Prometheus text endpoint for the GPU prefix-cache hit rate.
    Metric name is representative as of 2026 — run
    `curl -s http://localhost:8000/metrics | grep cache` to confirm the current name.
    Returns NaN if the metric is absent (e.g. caching disabled).
    """
    resp = httpx.get(metrics_url, timeout=5.0)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        if line.startswith("vllm:gpu_prefix_cache_hit_rate") and not line.startswith("#"):
            return float(line.split()[-1])
    return float("nan")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rate = get_prefix_cache_hit_rate()
    logger.info("GPU prefix-cache hit rate: %.3f", rate)
```

A hit rate above 0.7 on this workload confirms the 800-token system preamble is being cached and reused. Below 0.4 on a workload with a shared preamble almost always means the bytes are not identical across requests — check for injected timestamps, UUID nonces, or encoding differences.

**Step 4 — Plot the latency–throughput curve and identify the goodput knee.**

```python
"""plot_latency_throughput.py — latency-throughput frontier with goodput knee annotation."""
import json
import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TTFT_SLO_S = 1.5    # seconds
ITL_SLO_S = 0.080   # seconds (80 ms)


def find_goodput_knee(rows: list) -> Optional[dict]:
    """Highest-throughput point where both TTFT p99 and ITL p99 SLOs are satisfied."""
    compliant = [
        r for r in rows
        if r["ttft_p99"] <= TTFT_SLO_S and r["itl_p99"] <= ITL_SLO_S
    ]
    if not compliant:
        logger.warning("No operating point satisfies both SLOs. Loosen thresholds or reduce load.")
        return None
    return max(compliant, key=lambda r: r["achieved_throughput"])


def plot(rows: list, out_path: str = "latency_throughput.png") -> None:
    throughput = [r["achieved_throughput"] for r in rows]
    ttft_p50 = [r["ttft_p50"] for r in rows]
    ttft_p99 = [r["ttft_p99"] for r in rows]
    itl_p99_ms = [r["itl_p99"] * 1000 for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(
        "vLLM — Latency–Throughput Frontier\n8B instruct · 70% short / 30% long prompts",
        fontsize=12,
    )

    ax1.plot(throughput, ttft_p50, "o-", color="steelblue", label="TTFT p50")
    ax1.plot(throughput, ttft_p99, "s--", color="tomato", label="TTFT p99")
    ax1.axhline(TTFT_SLO_S, color="tomato", linestyle=":", linewidth=1.2,
                label=f"SLO {TTFT_SLO_S}s")
    ax1.set_ylabel("TTFT (s)")
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(throughput, itl_p99_ms, "^-", color="darkorange", label="ITL p99")
    ax2.axhline(ITL_SLO_S * 1000, color="darkorange", linestyle=":", linewidth=1.2,
                label=f"SLO {ITL_SLO_S * 1000:.0f} ms")
    ax2.set_ylabel("ITL p99 (ms)")
    ax2.set_xlabel("Achieved throughput (req/s)")
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    knee = find_goodput_knee(rows)
    if knee is not None:
        kx = knee["achieved_throughput"]
        for ax in (ax1, ax2):
            ax.axvline(kx, color="green", linestyle="--", linewidth=1.2)
        ax1.text(
            kx + 0.2, ax1.get_ylim()[1] * 0.85,
            f"Knee\n~{kx:.1f} req/s",
            color="green", fontsize=8,
        )
        logger.info(
            "Goodput knee: %.1f req/s  |  TTFT p99 %.2fs  |  ITL p99 %.0fms",
            kx, knee["ttft_p99"], knee["itl_p99"] * 1000,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", out_path)


if __name__ == "__main__":
    with open("sweep_results.json") as f:
        rows = json.load(f)
    plot(rows)
```

**Expected output (representative — numbers vary by GPU and quantization):**

```text
2026-07-02 10:00:05 INFO arrival_rate=1.0 req/s — running for 60s
2026-07-02 10:01:05 INFO   TTFT p99=0.31s  ITL p99=18ms  throughput=1.0 req/s
...
2026-07-02 10:16:05 INFO arrival_rate=16.0 req/s — running for 60s
2026-07-02 10:17:05 INFO   TTFT p99=1.19s  ITL p99=61ms  throughput=15.7 req/s
2026-07-02 10:17:05 INFO arrival_rate=20.0 req/s — running for 60s
2026-07-02 10:18:05 INFO   TTFT p99=2.54s  ITL p99=93ms  throughput=18.1 req/s
2026-07-02 10:18:05 INFO Goodput knee: 15.7 req/s  |  TTFT p99 1.19s  |  ITL p99 61ms
```

The characteristic shape: TTFT and ITL are flat up to the knee, then rise sharply. The knee is where the system saturates — this is your operating point for the $/1M token calculation. Above it you get more throughput only by violating SLOs.

**Troubleshooting table.**

| Symptom | Cause | Fix |
| --------- | ------- | ----- |
| All requests time out at high arrival rate | Server queue grows unbounded; replicas can't keep up | Lower arrival rate; check server logs for OOM-induced preemption (swap events) |
| TTFT is suspiciously flat across all rates | `--enforce-eager` left on; CUDA graph capture disabled | Remove `--enforce-eager` from the launch command |
| Hit rate reported as NaN from metrics endpoint | Metric name changed in current vLLM version | `curl -s http://localhost:8000/metrics \| grep cache` to find the active metric name |
| `itl_p99=0` for every row | Streaming not working — full response arrives at once | Confirm `"stream": True` in the request payload; check for a proxy stripping chunked encoding |
| Sweep throughput lower than vLLM's own bench | Client-side timing includes HTTP/TLS overhead; `max_tokens` differ | Compare at equal `max_tokens`; measure at the server's Prometheus `throughput_tokens_total` counter for a server-side ground truth |

## Interview Q&A

**Q1. Why does continuous batching dominate static batching for LLM serving?**
**A.** Static batching forms a batch, runs it to completion, then forms the next. Because output lengths vary wildly, the whole batch waits for its longest sequence — GPUs idle on padding/finished slots, and new arrivals queue behind the entire batch (terrible TTFT). Continuous batching schedules per *iteration*: at every decode step, finished sequences exit and queued requests join immediately, keeping the batch full at all times. Result: near-elimination of padding waste, dramatically better throughput at equal latency, and decoupling of one request's length from another's queueing delay. It requires engine-level support — per-iteration scheduling and a KV cache that can grow/free per sequence — which is precisely what PagedAttention's block-based allocation enables; the two innovations are complementary halves of the modern engine.

**Q2. What problem does PagedAttention solve, exactly?**
**A.** KV-cache memory management. Naively, each request preallocates contiguous KV memory for its *maximum possible* length — but actual lengths vary, so most of that reservation is never used (internal fragmentation), and differing lifetimes shred the heap (external fragmentation); measured waste in pre-vLLM systems was a large majority of KV memory. PagedAttention applies the OS virtual-memory idea: KV is stored in fixed-size blocks (e.g., 16 tokens), allocated on demand, mapped through per-sequence block tables, with no contiguity requirement. Waste collapses to under one block per sequence, so far more sequences fit in HBM → larger effective batch → higher throughput. The block indirection also enables copy-on-write sharing for prefixes and cheap preemption/swap — it's the substrate prefix caching and disaggregated KV transfer are built on.

**Q3. When would you disaggregate prefill and decode, and what's the main engineering challenge?**
**A.** Disaggregate when (1) prefill–decode interference is your binding constraint — long, variable prompts causing ITL p99 spikes for everyone else — and chunked prefill tuning can't hold the SLO; (2) you want to scale/provision the phases independently (prompt-heavy RAG traffic needs prefill compute; chat-heavy traffic needs decode bandwidth — even different GPU SKUs per phase); (3) traffic volume justifies the operational complexity. The crux is **KV-cache transfer**: hundreds of KB per token means gigabytes per long request moving prefill→decode inside the latency budget, demanding RDMA/NVLink-class paths and a dedicated KV transfer engine, plus routing that's aware of where KV lives. Below that scale, the honest answer is co-located instances with chunked prefill + prefix caching — and saying so demonstrates judgment, not ignorance.

**Q4. Your agent product has 20-step tool loops re-sending a growing transcript each step. How do you make serving efficient?**
**A.** This workload is prefix-caching paradise and cache-miss hell, depending entirely on design. (1) **Maximize prefix stability:** system prompt and tool definitions first and byte-identical across steps; append-only transcript; no timestamps/randomness in the preamble — every step then hits the radix cache and only the newest turn is prefilled. (2) **Session-affine, KV-aware routing:** pin a session's steps to the replica holding its prefix (or use a shared/tiered KV store) — a random load balancer destroys the hit rate. (3) **Structured decoding** for tool calls (grammar-constrained JSON). (4) **Context engineering as a serving optimization:** compaction/summarization when the transcript exceeds budget — but note compaction *invalidates the cache*, so do it at deliberate breakpoints. (5) Track **KV-cache hit rate as a first-class product metric**; in agentic systems it's often the single biggest cost lever. SGLang-style RadixAttention or vLLM automatic prefix caching both serve this; measure on your trace.

**Q5. Sketch the capacity plan for serving an 8B model to 1M DAU chat product.**
**A.** Assumptions out loud: 1M DAU → ~5% peak-hour concurrency = 50k active; each sends a message every 40 s → 1.25k req/s; 800-token average context after history, 250-token responses → ~310k decode tok/s and ~1M prefill tok/s peak. Single H100-class GPU serving 8B FP8 with a tuned engine: order 5–10k decode tok/s within chat SLOs (state it as a planning number to be measured) → ~40–70 GPUs for decode-equivalent load before optimizations. Then optimize: shared-system-prompt prefix caching kills most prefill; FP8 KV + quantized weights raise per-GPU batch; autoscale on goodput with ~30% headroom for spikes and replica failure; multi-region by data residency. Add the cost line: at $2–3/H100-hr, ~$3–5k/day ≈ fractions of a cent per DAU — then the punchline interviewers love: the same product on a frontier API at ~$1–3/1M tokens would cost 5–20× more at this volume, which is the build-vs-buy crossover argument from the foundations chapter.
