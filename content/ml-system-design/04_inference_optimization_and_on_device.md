# Module 04 — Inference Optimization & On-Device ML

## Why this module matters

Inference is where ML systems spend their money and where users feel quality. The entire discipline reduces to one mental model — the **roofline** — plus a toolbox of techniques (quantization, distillation, speculative decoding, better attention) for moving along it. On-device deployment is the extreme version of the same problem and is one of the fastest-growing specializations in 2026 (privacy, latency, and cost all push the same direction).

## 1. Inference arithmetic — memorize this section

- **FLOPs:** a forward pass costs ≈ **2N FLOPs per token** (N = active params; for MoE use active, not total).
- **KV cache per token** = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes. Example, Llama-3-70B (80 layers, 8 KV heads via GQA, head_dim 128, FP16): 2×80×8×128×2 ≈ **320 KB/token** → a 4k-token context ≈ 1.3 GB *per request*. This single number explains half of LLM serving design.
- **The two phases:** **Prefill** processes the whole prompt in parallel — big matmuls, **compute-bound**. **Decode** generates one token at a time — every step must read all weights + the KV cache from HBM, so it's **memory-bandwidth-bound**.
- **Decode speed upper bound (batch=1):** tokens/sec ≈ memory bandwidth / bytes-of-weights-read. A 7B model in FP16 (14 GB) on a GPU with 1 TB/s ≈ ~70 tok/s ceiling; quantize to 4-bit (3.5 GB) and the ceiling becomes ~280 tok/s. This is why **quantization speeds up decode** — it's a bandwidth play, not a FLOPs play.
- **Arithmetic intensity & batching:** batching B requests reuses each weight read across B tokens, raising arithmetic intensity until you cross from bandwidth-bound to compute-bound. Throughput rises ~linearly with batch size until that knee; per-token latency degrades slowly until it doesn't. All serving design (next chapter) is about operating near the knee.
- **Latency metrics:** TTFT (time to first token, dominated by prefill ∝ prompt length), TPOT/ITL (per-token decode latency), and goodput (requests/sec meeting both SLOs).

### Foundations Box: the roofline / bandwidth model of decode

The per-token-decode bytes-moved equation has two terms:

```text
bytes_per_token = (params × bytes_per_param)                                         # weights
                + (2 × n_layers × n_kv_heads × head_dim × bytes_kv × seq_len)        # KV cache
```

At short context the KV term is small and weights dominate; as context grows the KV term can match or exceed the weight term (see the Q2 answer below for the 8B 32k calculation). The decode throughput ceiling is therefore `tokens/s ≤ HBM_bandwidth_GB/s / bytes_per_token`.

Worked example — 8B model on an A100 80GB SXM4 (2.0 TB/s measured HBM bandwidth), short context so weights dominate:

| Precision       | Weight bytes | Theoretical ceiling | Typical measured |
|-----------------|--------------|---------------------|------------------|
| BF16            | 16.0 GB      | 125 tok/s           | 75–95 tok/s      |
| INT4 (GPTQ/AWQ) | 4.0 GB       | 500 tok/s           | 280–380 tok/s    |

On an H100 SXM (3.35 TB/s) scale both rows proportionally. Measured numbers land at 60–70% of the ceiling because KV-cache reads grow with context length, kernel-dispatch overhead is non-zero, and INT4 dequantization adds a small FLOPs tax. If measured tok/s is less than 50% of the theoretical ceiling, diagnose the gap before attributing slowness to the model.

Measure achieved bandwidth directly — time N decode steps and compute bytes moved:

```python
import time
import logging
import torch

logger = logging.getLogger(__name__)


def measure_decode_bandwidth(
    model: torch.nn.Module, tokenizer, prompt: str, n_new_tokens: int = 128
) -> dict:
    """Return tok/s and achieved HBM bandwidth (GB/s) for n_new_tokens decode steps."""
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warm-up: fills the KV cache for the prompt prefix, primes CUDA kernels.
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_generated = out.shape[-1] - inputs["input_ids"].shape[-1]
    tok_s = n_generated / elapsed
    # Lower bound: counts weight reads only; undercounts KV traffic at long context.
    achieved_bw_gb_s = (model_bytes * n_generated) / elapsed / 1e9
    logger.info(
        "%.1f tok/s | achieved bandwidth %.0f GB/s (model %.1f GB)",
        tok_s, achieved_bw_gb_s, model_bytes / 1e9,
    )
    return {"tok_per_sec": tok_s, "achieved_bw_gb_s": achieved_bw_gb_s}
```

Compare `achieved_bw_gb_s` to the GPU's spec bandwidth. The ratio is the serving-stack bandwidth utilization. Common gaps: FlashAttention not engaged (critical at long context), INT4 kernel falling back to a slow dequant path, or KV-cache growth beyond what the short-context table above assumes.

## 2. Quantization

- **Post-training quantization (PTQ):** weight-only INT4/INT8 — **GPTQ** (error-compensating rounding), **AWQ** (protect the ~1% activation-salient channels), plus the llama.cpp **GGUF k-quant/i-quant** family for CPU/edge. Weight-activation: **SmoothQuant** (migrate activation outliers into weights) enabling INT8/FP8 matmuls. **FP8** is now the default serving precision on Hopper/Blackwell (near-lossless, ~free speedup); **NVFP4/MXFP4** 4-bit floating formats are the Blackwell-era frontier. **KV-cache quantization** (FP8/INT4) attacks the other memory consumer and is often the difference between fitting a batch or not.
- **QAT:** fine-tune with fake-quantization in the loop (TorchAO) when PTQ at the target precision loses too much. The professional workflow: build a **per-layer sensitivity map** (quantize one layer-group at a time, measure task-metric delta). Empirically MLP blocks are tolerant; attention input projections and SSM/linear-attention output projections in hybrid architectures are high-sensitivity — keep those in 8-bit/BF16 in a mixed-precision recipe rather than forcing uniform 4-bit.
- **Evaluation discipline:** perplexity is necessary but insufficient — always evaluate the quantized artifact on *your task metric*; small perplexity deltas can hide large drops on long-tail capabilities (math, low-resource languages).

### Foundations Box: quantization

**PTQ vs QAT — the decision.** PTQ uses a calibration set (typically 512–1024 samples) and no training loop — hours to run, no training infrastructure. QAT inserts fake-quantization nodes into the forward pass and fine-tunes until the network adapts to the quantization grid; the network can recover quality but costs a full fine-tuning run. Default to PTQ; escalate to QAT only when PTQ measurably hurts *your task metric* and you have the fine-tuning budget. In an interview, naming this decision axis — not just listing both techniques — is the senior signal.

**Weight-only vs weight+activation.** Weight-only INT4 quantizes stored weight tensors; activations stay BF16/FP16 at runtime. This is safer because activation distributions vary per input and contain per-channel outliers that can be orders-of-magnitude larger than the median — SmoothQuant migrates those outliers into weights before quantizing; AWQ's per-channel scaling protects the most activation-salient channels. For **decode throughput**, weight-only is the correct target: bandwidth savings come from bytes read per decode step, and between steps activations don't reside in HBM.

**Why 4-bit is the bandwidth sweet spot.** BF16 = 2 bytes/weight; INT4 = 0.5 bytes/weight — a 4× reduction in bytes read per decode step and therefore a 4× higher throughput ceiling. FP8/INT8 give ~2×; going below 4-bit (INT3, INT2) collapses quality on general-purpose models today. The INT4 weight-only crossover is the practical operating point for 2026 serving.

**Parity-check discipline.** After any quantization step, before shipping: run a fixed held-out batch through both the original and quantized model and check the maximum absolute logit difference.

```python
import logging
import torch

logger = logging.getLogger(__name__)


def quantization_parity_check(
    fp_model: torch.nn.Module,
    quant_model: torch.nn.Module,
    input_ids: torch.Tensor,
    tol: float = 0.5,
) -> float:
    """Return max absolute logit diff between fp_model and quant_model on input_ids."""
    fp_model.eval()
    quant_model.eval()
    with torch.inference_mode():
        fp_logits = fp_model(input_ids).logits
        q_logits = quant_model(input_ids).logits
    diff = (fp_logits - q_logits).abs().max().item()
    if diff > tol:
        logger.warning("Parity FAILED: max abs diff %.4f > tol=%.4f", diff, tol)
    else:
        logger.info("Parity OK: max abs diff %.4f", diff)
    return diff
```

The 0.5 tolerance on raw logits is a starting point — tighten it for safety-critical outputs. A failure means uniform INT4 is too aggressive somewhere; use the sensitivity map (Project 04 step 2) to identify which layer groups to keep at 8-bit.

**Where accuracy actually degrades.** Not all layers quantize equally: lm_head and the first embedding layer are high-sensitivity (distribution shifts there compound through sampling); Q/K projections in attention are more fragile than V or the MLP blocks (quantization errors in Q/K distort attention patterns); the final 2–3 transformer blocks before lm_head accumulate errors from earlier layers. MLP blocks in the middle of the network are the most tolerant — start a mixed-precision recipe there and promote layers to higher precision only where the sensitivity map shows degradation.

## 3. Faster decoding

- **Speculative decoding:** a cheap drafter proposes k tokens; the target model verifies them in one parallel forward pass; rejection sampling keeps the output distribution *exactly* the target's. Speedup ≈ (expected accepted length)/(cost ratio). Variants: independent small draft model; **Medusa** (extra decoding heads); **EAGLE-1/2/3** (feature-level drafting — current best, ~3–6× on common workloads); **MTP-based self-speculation** (DeepSeek-V3 lineage) where training-time multi-token-prediction heads double as the drafter — elegant because the drafter is distribution-matched by construction. Caveat: speedups shrink at high batch sizes (the GPU is already compute-busy verifying) — speculative decoding is primarily a *latency* tool for low-batch regimes, which is exactly the on-device regime.
- **Attention efficiency:** **FlashAttention-2/3** (exact attention, IO-aware tiling; FA-3 exploits Hopper's TMA/FP8), **GQA** (fewer KV heads — now universal), **MLA** (latent KV compression), sliding-window + attention-sink patterns, and **hybrid linear-attention/SSM architectures** (Mamba-2, GatedDeltaNet lineage) that replace most attention layers with constant-size recurrent state — O(1) KV memory per layer, transformative for long context on small devices, but with a systems tax: exotic ops have immature support in export-based toolchains (ONNX-style), so framework choice (see the on-device section below) can be dictated by architecture.

### How-to: speculative decoding in practice

**Getting a drafter.** EAGLE-2/3 heads are published on HuggingFace for common base models — search `eagle2-<model-name>` or `eagle3-<model-name>`; vLLM and SGLang load them directly via `--speculative-model`. For a model without published heads, use a small same-family model as an independent drafter (a 1B or 3B variant for a 7B target). To train EAGLE heads from scratch: roughly 12–24 GPU-hours of supervised training on a representative corpus, fine-tuning only the draft head weights while the target model is frozen — the heads are distribution-matched by construction to the target.

**Acceptance rate — the one number.** The acceptance rate α = (tokens accepted) / (tokens proposed) determines whether speculative decoding is worth running. At γ=5 proposed tokens and α=0.75, each verification call returns 1 + 0.75×4 ≈ 4 tokens on average — same bandwidth cost as one non-speculative decode step but ~4× the output. Measure it via vLLM's `spec_decode_draft_acceptance_rate` metric (exposed at the `/metrics` endpoint), or manually: for each verification call, count how many proposed tokens were accepted before the first rejection and average over requests. A healthy rate is >0.60 for instruction-tuned tasks at low temperature; below 0.45 the drafter is domain-mismatched and likely not worth the overhead.

**The bandwidth model of why it works.** Without speculation: T tokens cost T × W bandwidth reads (W = bytes to sweep all weights once). With speculation at γ=5, α=0.70: each verification call costs ~1.1W (target forward pass plus cheap drafter overhead) and returns ~3.8 tokens on average — roughly 0.29W per output token, a ~3.4× improvement. This advantage collapses at high batch sizes because the target-model forward pass is no longer bandwidth-limited — the GPU is compute-saturated during verification and the parallel check stops being free. Speculative decoding is a latency tool for batch≤4; it does not reliably improve throughput at batch≥16.

**vLLM config (representative flags — check current docs; flag names change between minor releases):**

```bash
# EAGLE head as drafter
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model lmsys/eagle2-llama3.1-8b \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 1

# Independent small-model drafter for a larger target
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5
```

Monitor `spec_decode_draft_acceptance_rate` and `spec_decode_efficiency` in the metrics endpoint after serving 1k+ requests. If acceptance rate is consistently below 0.45 on production traffic, switch drafters before concluding speculative decoding doesn't work for your use case.

## 4. On-device inference

- **Stacks:** **llama.cpp/GGUF** (CPU+GPU everywhere, the lingua franca), **MLX** (Apple-silicon-native, Python/Swift, lazy unified-memory arrays — the cleanest path for hybrid/novel architectures on Apple hardware), **Core ML/ANE** (Apple's NPU — fast but rigid op support), **ExecuTorch** (PyTorch's mobile/edge runtime), **MNN** (Alibaba's mobile engine, strong on Android), **ONNX Runtime** (broad but struggles with novel architectures), TensorRT/TensorRT-LLM (NVIDIA edge incl. Jetson).
- **The on-device design loop:** memory budget first (a phone gives you 2–6 GB realistically → 1–4B params at 4-bit), then bandwidth-derived tok/s ceiling, then thermal sustainability (sustained decode throttles), then battery. Prefill is the UX killer on mobile (long prompts on weak compute) — mitigate with prompt caching, smaller contexts, and NPU prefill/GPU decode splits.
- **The selection matrix in practice:** target hardware × architecture support × quantization formats × update/distribution story. E.g., a hybrid-attention VLM may run beautifully via MLX on iOS while the Android path through ONNX is blocked on unsupported ops, forcing either MNN custom ops or an architecture swap — evaluating *exportability before fine-tuning* is the senior move.

## 5. Designing multimodal and VLM systems

VLMs are now an interview topic in their own right, and the document-processing pipeline is the canonical "design this" question at companies doing document AI, medical imaging, and content understanding. The core insight is that multimodal inference is not just "LLM with an image attached" — it introduces a distinct compute profile, a vision token budget problem, and a heterogeneous hardware scheduling challenge that the interview is probing.

### Architecture: encoder + connector + LLM

The dominant VLM architecture has three stages:

1. **Vision encoder** (or encoder-free patch embedding): a ViT-class model processes the image into a sequence of visual features. CLIP-ViT and SigLIP variants are the common choices for the encoder; encoder-free approaches (e.g., treating image patches directly as tokens via linear projection) are used in some lightweight models to reduce latency. The encoder is compute-heavy and parallelizable — unlike autoregressive decode, the entire image is processed in one forward pass with no sequential dependency.
2. **Connector / projector**: a lightweight MLP or cross-attention block that maps vision encoder outputs into the LLM's token embedding space. This is where the vision token count is determined — the design choice here directly controls cost.
3. **LLM**: processes the combined vision tokens + text tokens autoregressively. From the serving engine's perspective, the visual tokens are just a long prefix — they consume KV cache exactly like text tokens. All the prefill/decode mechanics from this chapter and the serving chapter apply.

**For on-device VLMs** (covered in detail in the on-device section below), the vision encoder is often the largest component by latency on mobile hardware. The iOS/Android export considerations discussed there constrain architecture choice — the same decisions apply at server scale but with more room to maneuver.

### Vision token budgets

Visual tokens are expensive. A naive encoding of a 1080p image using a standard ViT-Large patch size can produce **1 000–3 000 visual tokens**, each of which costs KV cache space and prefill compute identical to a text token. At a frontier API rate of $2/1M tokens, a single high-resolution image costs $0.002–$0.006 in visual tokens alone — multiply by document page count and request volume to see why token budget management is a first-class design concern.

**Tiling strategies** trade resolution against cost: a high-resolution image is split into N tiles, each encoded independently at a fixed resolution, with a low-resolution thumbnail also encoded for global context. GPT-4o and Claude's vision implementations use variants of this; it allows the model to "see" fine detail in specific regions while keeping the base cost predictable. The design knobs are tile count (more tiles = higher resolution = more tokens = more cost) and thumbnail resolution. For most document-processing tasks, 2–6 tiles per page is the practical sweet spot — enough for printed text, not needed for handwriting recognition which requires higher density.

**Encode only what matters**: for multi-page documents, don't encode all pages as high-resolution images upfront. Use a cheap classifier or OCR confidence score to route pages — text-dense pages at low resolution; pages with tables, figures, or stamps at higher resolution. This is the cascade pattern applied to visual token allocation.

### Heterogeneous compute: the two-pool architecture

Vision encoding and autoregressive LLM decode have different compute profiles:

- **Vision encoding**: compute-heavy, massively parallelizable across images, no sequential dependency — runs efficiently on GPUs with high FLOP utilization, and can be batched aggressively across concurrent requests.
- **LLM decode**: memory-bandwidth-bound, sequential token generation — GPU utilization is low at batch=1 and rises with batch size.

Running both on the same GPU pool means one workload underutilizes the resource the other needs. The production pattern at scale is a **separate vision-encoding pool** (optimized for throughput/compute) feeding encoded representations to the **LLM serving pool** (optimized for KV memory and continuous batching). At smaller scale, co-location with careful batching works; separate the pools when vision encoding latency is in the critical path for TTFT SLOs.

**Preprocessing pipeline**: image resize, tile splitting, and normalization are CPU-bound operations that run before the GPU pipeline. At high request rates (document processing at millions of pages/day), this CPU preprocessing becomes a bottleneck. The mitigation: a dedicated CPU preprocessing fleet that parallelizes image prep and queues encoded tensors to the vision-encoder GPU pool — the same producer-consumer pattern as audio preprocessing in speech pipelines.

### Multimodal eval

Multimodal systems require per-modality evaluation and a specific hallucination check that text-only evals miss:

- **Per-modality metrics**: OCR accuracy (character error rate), field extraction precision/recall (per field type, sliced by document category), layout understanding (table structure accuracy, bounding-box agreement).
- **Visual hallucination checks**: the model confidently describing content not present in the image — a distinct failure mode from textual hallucination. Detection: a reference-free checker that asks a smaller model "does this description contain any claims that are not visually verifiable in the image?" Run on a sample of production outputs, calibrated against human review.
- **Resolution/tiling sensitivity**: models are often significantly worse on low-resolution or poorly cropped images than on clean scans. Eval must include image quality slices (sharpness, exposure, skew angle) — aggregate metrics hide per-quality regressions.

### Cost profile differences from text-only

VLM requests cost more per request but the budget breakdown shifts: input tokens dominate (1 000+ visual tokens vs perhaps 200 text tokens of context), output tokens are relatively cheap (a structured extraction output is typically 50–300 tokens). This inverts the text-API cost structure where output costs 3–5× per token. For VLM document pipelines: optimize aggressively on input token count (tiling, resolution, selective high-res encoding), and accept that output cost is not the bottleneck.

### Worked mini-design: document-processing pipeline

This is the now-canonical interview question. "Design a system that processes 10M PDFs/day and extracts structured fields" appears across multiple mock questions in the interview playbook. The VLM-aware answer:

**Ingest**: PDFs arrive via queue (Kafka/SQS); a renderer converts pages to images (CPU fleet, parallelized, poppler/pdfium); a triage step classifies page type (text-dense vs mixed vs image-heavy vs handwritten) — cheap classifier or heuristic on text-layer richness.

**Preprocessing**: resize to max dimension, split into tiles based on page type (1 tile for simple text pages, 4–6 for mixed/tabular), normalize. This is the CPU preprocessing fleet.

**Cascade tier 1 — cheap path**: for text-dense pages with a clean OCR text layer, skip the VLM entirely — run a fine-tuned text-only model on the extracted text. Zero vision tokens, order-of-magnitude cheaper. Route 60–70% of pages here for typical business document mixes.

**Cascade tier 2 — VLM path**: pages where OCR text is insufficient (tables, stamps, handwriting, embedded images with data). Run through the vision-encoder + LLM pipeline. Fine-tuned 7B VLM at FP8 with tiling.

**Cascade tier 3 — frontier VLM API**: low-confidence outputs from tier 2, complex layouts, or rare document types. 2–5% of pages, cost-managed by the cascade.

**Output**: structured JSON per page → field aggregation across pages → schema validation → confidence scores attached.

**Numbers**: 10M pages/day ≈ 115 pages/s sustained. Tier 1 handles ~80 pages/s on a CPU fleet. Tier 2 at 35 pages/s on a small GPU pool (8×H100, ~4 pages/s/GPU at batch 8 with tiling). Tier 3: ~5 000 API calls/day. Show the cost line: tier 1 at effectively zero marginal GPU cost; tier 2 at ~$0.01–0.05/page GPU cost; tier 3 at API rates. The cascade exists because tiers 2 and 3 costs are 10–100× tier 1.

**Cross-reference**: on-device VLM deployment considerations appear in the on-device section above — the token budget, encoder separation, and tiling decisions are identical; only the hardware constraints differ.

## Going deeper

- The inference-arithmetic section is the load-bearing one: the roofline, the 2N-FLOPs-per-token rule, the KV-cache formula, and the bandwidth-derived decode ceiling. Everything else in this chapter is a technique for moving along that roofline.
- The quantization literature (error-compensated INT4 rounding, activation-salient channel protection, activation-outlier migration, FP8/FP4 serving formats) rewards hands-on benchmarking — the Project below has you reproduce the bandwidth model against measured tokens/sec.
- Speculative decoding and its feature-level and self-speculative variants, plus IO-aware exact attention, are the decode-latency toolbox; study their acceptance-rate and batch-size caveats.
- For the on-device track, the export toolchains differ sharply in their support for novel/hybrid architectures — verify exportability on the exact architecture before committing to a base model.

## Project 04 — Quantization & speculative decoding lab

(1) Take an ~3–8B instruct model; produce GPTQ-INT4, AWQ-INT4, and GGUF Q4_K_M artifacts. Benchmark all three + the BF16 baseline on (a) perplexity, (b) a task eval you care about (e.g., GSM8K subset or a doc-extraction set), (c) tokens/sec at batch 1 and batch 8. Verify the bandwidth model: predicted tok/s = bandwidth/model-bytes vs measured. (2) Build a per-layer sensitivity map: quantize layer groups one at a time and chart task-metric degradation; identify which projections are fragile. (3) Run speculative decoding (EAGLE head or a 0.5B drafter via vLLM/SGLang) and measure speedup vs acceptance rate at temperature 0 and 0.8. (4) Bonus, on-device: run the same model via llama.cpp or MLX on a laptop/phone and measure sustained (5-minute) vs burst decode speed to observe thermal throttling.

**Expected outputs** (representative for a 7–8B instruct model on an A100 80GB; exact numbers vary by model family, calibration set, and task):

| Artifact        | Perplexity Δ vs BF16 | GSM8K Δ     | Batch=1 tok/s    | Batch=8 tok/s |
| --------------- | -------------------- | ----------- | ---------------- | ------------- |
| BF16 baseline   | —                    | —           | 80–100           | 450–600       |
| AWQ INT4        | +0.2 to +0.4         | 0 to −2 pts | 280–380          | 500–650       |
| GPTQ INT4       | +0.2 to +0.5         | 0 to −2 pts | 260–360          | 500–650       |
| GGUF Q4\_K\_M   | +0.3 to +0.5         | 0 to −3 pts | 30–60 (CPU)      | N/A (CPU)     |
| Spec dec + INT4 | same as INT4         | same        | 150–250 (α≈0.70) | ≈ INT4        |

For step 1: if your batch=1 tok/s is below 50% of the theoretical ceiling from the Foundations Box, stop and diagnose before moving to step 2 — something is wrong with the serving stack, not the quantization. For step 3: speculative decoding speedup should be measurable at temperature 0 and shrink noticeably at temperature 0.8 — both outcomes are expected and worth reporting.

**Troubleshooting:**

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Parity check fails (max abs diff > 0.5) | High-sensitivity layer forced to INT4 | Run sensitivity map; promote fragile projections to BF16 |
| INT4 tok/s only 1.2–1.5× BF16, not 3–4× | Dequant kernel not fused or falling back to slow path | Verify GPTQ-Marlin or AWQ-Marlin kernels are active; check GPU compute capability matches the kernel |
| Spec decoding acceptance rate < 0.40 | Drafter domain or temperature mismatch | Switch to a task-matched drafter; reduce γ; verify drafter and target share the same tokenizer |
| Perplexity fine; task metric drops sharply | Long-tail capability degradation (math, low-resource languages) | Add task-specific calibration samples; target QAT on the failing capability |
| Speculative decoding slower than non-speculative | Batch size too high; GPU already compute-saturated | Use speculative decoding only at batch≤4; expected behavior at batch≥16 |
| On-device: sustained speed 40–60% below burst | Thermal throttling — expected on mobile | Report both burst and 5-min sustained numbers; plan UX around the sustained ceiling |

## Interview Q&A

**Q1. Why is LLM decoding memory-bound, and what follows from that?**
**A.** Each decode step generates one token, requiring a full pass that reads all model weights (plus the KV cache) from HBM while performing only ~2N FLOPs for that single token — arithmetic intensity of order 1 FLOP/byte, far below the hundreds of FLOPs/byte modern GPUs need to be compute-limited. So decode speed ≈ bandwidth/bytes-read, and the levers follow directly: **quantize** (fewer bytes per weight → proportionally faster), **batch** (amortize each weight read over many sequences → throughput), **shrink KV** (GQA/MLA/KV-quant — at long context the KV cache, not weights, dominates the bytes), and **speculative decoding** (turn several sequential bandwidth-bound steps into one parallel compute-bound verification). Prefill is the opposite — compute-bound — which is exactly why serving systems treat the two phases differently (next chapter).

**Q2. Compute the KV cache for an 8B model (32 layers, 8 KV heads, head_dim 128) at 32k context, FP16 — and what would you do about it?**
**A.** Per token: 2×32×8×128×2 bytes = 131 KB. At 32k tokens: ~4.3 GB per sequence — comparable to the 4-bit weights of the model itself, so a batch of 16 such requests needs ~69 GB for KV alone. Levers, in order: FP8/INT4 **KV quantization** (2–4×), **paged allocation** so memory is per-token-used not per-max-length, **prefix sharing** across requests with common prompts, architecture-level fixes if you own the model (more aggressive GQA or MLA), sliding-window attention for layers that tolerate it, and KV offload to CPU/storage for long idle sessions. If this is on-device, the honest answer is: 32k context at 8B doesn't fit — cut context or model size.

**Q3. AWQ vs GPTQ vs QAT — how do you choose?**
**A.** All target low-bit weights but differ in cost and robustness. **GPTQ**: layer-by-layer error-compensated rounding using a small calibration set; fast, good general INT4 quality, can overfit calibration data slightly. **AWQ**: observes that a tiny fraction of weight channels (those aligned with large activations) carry disproportionate importance, protects them via per-channel scaling; tends to be more robust on instruction-following models and needs no backprop. Both are PTQ — hours, no training infra. **QAT** simulates quantization during fine-tuning so the network adapts; choose it when (a) PTQ at the target precision measurably hurts your *task* metric, (b) you're going very low-bit (≤4-bit weights+activations or quantizing sensitive layer types), or (c) you're already fine-tuning anyway, making QAT nearly free to add. Practical recipe: try AWQ/GPTQ first, evaluate on task metrics, escalate to mixed-precision QAT guided by a layer sensitivity map only where PTQ fails.

**Q4. Explain speculative decoding and when it stops helping.**
**A.** A cheap drafter (small model, Medusa/EAGLE heads, or MTP heads) proposes γ tokens autoregressively; the target model scores all γ+1 positions in a single parallel forward pass; tokens are accepted left-to-right with a rejection-sampling rule that provably preserves the target distribution exactly; on first rejection, a corrected token is sampled and drafting resumes. Net effect: several decode steps' worth of tokens for ~one target-model bandwidth cost plus cheap drafting. It stops helping when: (1) **acceptance rate drops** — drafter poorly matched to target/domain or high sampling temperature (more rejections); (2) **high batch sizes** — the GPU is already compute-saturated, so the "free" parallel verification is no longer free and speculation can reduce throughput (it's a latency tool for small-batch/interactive and on-device regimes); (3) the draft overhead rivals the target cost (drafter too big relative to target). EAGLE-3-class feature-level drafting currently gives the best acceptance/cost tradeoff; MTP-based self-speculation is attractive because the drafter ships inside the model.

**Q5. You must run a 2B VLM on both iOS and Android. Walk through your engineering plan.**
**A.** Start from constraints: memory (4-bit weights ≈ 1 GB + KV + vision encoder activations — fits modern flagships), bandwidth-derived tok/s target, and thermal sustainability for multi-image sessions. Then **verify exportability per platform before any fine-tuning**: iOS — MLX (flexible, handles novel/hybrid architectures, GPU via Metal) vs Core ML (ANE speed but rigid ops); Android — MNN or ExecuTorch vs ONNX Runtime, where hybrid attention/SSM blocks frequently hit unsupported ops, so prototype the export with the *exact* architecture first; if blocked, choose a different base model rather than fighting the toolchain. Quantize with the sensitivity-map approach (4-bit MLPs, 8-bit fragile projections; QAT during the fine-tune); keep the vision encoder in 8-bit (vision towers are quantization-sensitive). Optimize prefill (the mobile UX killer): resize/tile images conservatively, cache the system prompt, consider NPU for prefill. Ship with on-device task evals in CI for both artifacts — iOS and Android builds *will* diverge numerically — plus telemetry for tok/s, thermal state, and OOM rates per device tier.
