# Module 05 — LLM Serving Systems — Part 2 of 2: Tooling, Benchmarking, and Interview Prep

This is part 2 of the LLM Serving Systems lesson. Part 1 covered the core engine ideas, disaggregation, reasoning-model serving, cascades, long-context serving, and capacity planning; here we get hands-on — a vLLM tuning survival guide, production alerting, a full benchmarking project, and the interview questions that test all of it.

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

## You can now

- Explain the four engine innovations — continuous batching, PagedAttention, prefix caching, and chunked prefill — and trace how each one addresses a specific bottleneck in LLM serving.
- Design a prefill/decode disaggregated serving architecture, state when it is worth the operational complexity, and identify KV-cache transfer as the binding engineering constraint.
- Configure a vLLM deployment for chat, long-context, and agentic workloads using the correct scheduler flags and quantization settings, and interpret the Prometheus metrics that expose misconfiguration.
- Apply the cascade pattern across query routing, RAG retrieval tiers, and reasoning-vs-fast model selection — and articulate why confidence calibration is the routing currency and why the threshold is a joint ML-product-finance decision.
- Produce a capacity plan and cost-per-1M-tokens estimate for a production serving system, naming every assumption, and enumerate the levers — quantization, prefix caching, batching, disaggregation, cascades — that move the number.
