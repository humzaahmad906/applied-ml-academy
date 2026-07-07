# Lab 3 — KV Cache, Quantization, and Benchmarking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/vlm-guide/notebooks/12_lab_inference_efficiency.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup. The full write-up and stack alternatives are below.

Measure the KV cache speedup directly, quantize the same model three ways for three hardware targets, benchmark tokens/sec and peak memory, then run speculative decoding. Makes the [Inference & Efficiency chapter](03_inference_and_efficiency.md) concrete.

## Setup

```bash
pip install torch transformers accelerate
# Install the quant stack(s) that match your hardware:
pip install bitsandbytes          # 4-bit/8-bit on CUDA
pip install llama-cpp-python      # GGUF on CPU or Mac
pip install mlx-lm                # Apple Silicon native
```

**Models:** `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB fp16). Part D also downloads `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB fp16) as the speculative decoding target.
Labels: **[CUDA]** — NVIDIA GPU; **[Universal]** — any CPU/Mac; **[MPS]** — Apple Silicon only.

```python
import random, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if device in ("cuda", "mps") else torch.float32,
    device_map=device,
).eval()
```

---

## Part A — KV Cache Timing

Without the cache, attention recomputes K and V for all prior tokens on every new step: O(n²) total. With the cache, each step adds exactly one new pair: O(n). The difference is proportional to sequence length.

```python
PROMPT = (
    "Explain the transformer architecture in detail — attention, positional encodings, "
    "feed-forward layers, residual connections, and how they combine in a decoder."
)
ids       = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
MAX_NEW   = 80
WARMUP    = 10

def timed_gen(use_cache: bool) -> tuple[float, int]:
    # warmup to avoid cold-start JIT noise
    with torch.no_grad():
        model.generate(ids, max_new_tokens=WARMUP, use_cache=use_cache,
                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, use_cache=use_cache,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return time.perf_counter() - t0, out.shape[1] - ids.shape[1]

t_cache, n1 = timed_gen(use_cache=True)
t_nocache, n2 = timed_gen(use_cache=False)
print(f"with cache    : {n1/t_cache:.1f} tok/s  ({t_cache:.1f}s)")
print(f"without cache : {n2/t_nocache:.1f} tok/s  ({t_nocache:.1f}s)")
print(f"speedup       : {t_nocache/t_cache:.1f}x")
```

---

## Part B — Quantization

Three independent stacks targeting different hardware. Load one at a time.

### B1. bitsandbytes NF4 4-bit [CUDA]

```python
from transformers import BitsAndBytesConfig

if device == "cuda":
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4: best quality/compression
        bnb_4bit_use_double_quant=True,      # quantize the scale constants too
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg, device_map="auto",
    ).eval()
    print(f"4-bit model loaded | GPU mem: {torch.cuda.memory_allocated()/1e6:.0f} MB")
else:
    print("[CUDA] bitsandbytes skipped — no GPU")
```

NF4 fits 4 bits to a normal distribution, minimizing error for normally-distributed weights. `double_quant` costs ~0.1 bit but recovers 0.1–0.3 perplexity points.

### B2. GGUF via llama-cpp-python [Universal — CPU / Mac]

GGUF bundles weights + quant metadata + tokenizer in a single file. `llama.cpp` uses ARM NEON / AVX2, making it the fastest option on CPU.

```python
from llama_cpp import Llama

# Pre-quantized GGUF files live at Qwen/Qwen2.5-0.5B-Instruct-GGUF on HF
# Q4_K_M = 4-bit K-quant, medium accuracy (~0.25 GB)
llm_gguf = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    verbose=False, n_ctx=512, n_threads=4,
)
out = llm_gguf.create_chat_completion(
    messages=[{"role":"user","content":"What is the KV cache?"}],
    max_tokens=60, temperature=0.0,
)
print(out["choices"][0]["message"]["content"])
```

### B3. MLX 4-bit [Apple Silicon — MPS]

```bash
# One-time conversion (or use a pre-converted mlx-community model directly)
python -m mlx_lm.convert \
    --hf-path Qwen/Qwen2.5-0.5B-Instruct \
    --mlx-path ./qwen-mlx-4bit --quantize --q-bits 4
```

```python
if device == "mps":
    from mlx_lm import load as mlx_load, generate as mlx_gen
    model_mlx, tok_mlx = mlx_load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print(mlx_gen(model_mlx, tok_mlx,
                  prompt="What is the KV cache?", max_tokens=60, verbose=False))
else:
    print("[MPS] MLX skipped — not on Apple Silicon")
```

---

## Part C — Benchmark Table

```python
def peak_mb() -> float:
    if device == "cuda": return torch.cuda.max_memory_allocated() / 1e6
    if device == "mps":  return torch.mps.current_allocated_memory() / 1e6
    return float("nan")

BENCH_P  = "Explain attention in one paragraph:"
BENCH_N  = 50
BENCH_IDS = tokenizer(BENCH_P, return_tensors="pt").input_ids.to(device)

def bench(label, fn, n_runs=3):
    rows = [fn() for _ in range(n_runs)]
    tps  = sorted(r[0] for r in rows)[n_runs//2]
    mem  = sorted(r[1] for r in rows)[n_runs//2]
    mem_s = f"{mem:8.0f}" if mem == mem else "     n/a"
    print(f"{label:<40} | {tps:8.1f} tok/s | {mem_s} MB")
    return tps, mem

def run_hf():
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(BENCH_IDS, max_new_tokens=BENCH_N,
                             use_cache=True, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    t = time.perf_counter() - t0
    return (out.shape[1]-BENCH_IDS.shape[1])/t, peak_mb()

print(f"\n{'Config':<40} | {'tok/s':>8} | {'peak MB':>8}")
print("-" * 65)
bench("HF transformers (baseline)", run_hf)

if device == "cuda":
    def run_bnb():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model_4bit.generate(BENCH_IDS, max_new_tokens=BENCH_N,
                                      use_cache=True, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        t = time.perf_counter() - t0
        return (out.shape[1]-BENCH_IDS.shape[1])/t, peak_mb()
    bench("bitsandbytes NF4 4-bit [CUDA]", run_bnb)

def run_gguf():
    t0 = time.perf_counter()
    out = llm_gguf(BENCH_P, max_tokens=BENCH_N, temperature=0.0)
    t = time.perf_counter() - t0
    n = len(tokenizer(out["choices"][0]["text"]).input_ids)
    return n/t, float("nan")
bench("GGUF Q4_K_M [CPU/Universal]", run_gguf)

if device == "mps":
    def run_mlx():
        t0 = time.perf_counter()
        resp = mlx_gen(model_mlx, tok_mlx, prompt=BENCH_P,
                       max_tokens=BENCH_N, verbose=False)
        t = time.perf_counter() - t0
        return len(tokenizer(resp).input_ids)/t, peak_mb()
    bench("MLX 4-bit [Apple Silicon]", run_mlx)
```

Representative numbers (M3 Pro / 4090 vary): GGUF CPU ~25 tok/s ~350 MB; MLX 4-bit ~80 tok/s ~350 MB vs. HF fp32 ~6 tok/s ~2100 MB on M3 Pro. On a 4090: HF bf16 ~200 tok/s ~1 GB, BnB NF4 ~280 tok/s ~400 MB.

---

## Part D — Speculative / Assisted Decoding

The draft model proposes k tokens cheaply; the target model verifies them in one forward pass. Accepted tokens cost one step; the first rejection restarts from that position. Output distribution is identical to the target model alone.

```python
LARGE_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # target — slower, higher quality
target = AutoModelForCausalLM.from_pretrained(
    LARGE_ID,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map=device,
).eval()
# model (0.5B) is the draft — already loaded above

spec_ids = tokenizer(
    "The residual stream in a transformer is", return_tensors="pt"
).input_ids.to(device)

t0 = time.perf_counter()
with torch.no_grad():
    std_out = target.generate(spec_ids, max_new_tokens=60, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
t_std = time.perf_counter() - t0

t0 = time.perf_counter()
with torch.no_grad():
    spec_out = target.generate(
        spec_ids, assistant_model=model,   # 0.5B draft
        max_new_tokens=60, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
t_spec = time.perf_counter() - t0

n_std  = std_out.shape[1]  - spec_ids.shape[1]
n_spec = spec_out.shape[1] - spec_ids.shape[1]
print(f"standard    : {n_std /t_std :.1f} tok/s")
print(f"speculative : {n_spec/t_spec:.1f} tok/s  (speedup: {t_std/t_spec:.2f}x)")
# Outputs should be identical (greedy, same target distribution)
assert tokenizer.decode(std_out[0],  skip_special_tokens=True) == \
       tokenizer.decode(spec_out[0], skip_special_tokens=True), "outputs differ"
```

Speedup is real when the draft's acceptance rate is ≥ 70% and the bottleneck is memory bandwidth (typical on modern GPUs). On CPU the overhead can dominate — profile before committing.

---

## What you built

- Timed KV cache directly: quadratic-to-linear speedup on the same model and prompt.
- Quantized one model three ways for three hardware targets: bitsandbytes NF4 (CUDA), GGUF Q4_K_M (CPU/Mac), MLX 4-bit (Apple Silicon).
- Benchmarked tokens/sec and peak memory across all running configs and printed a comparison table.
- Ran speculative decoding with 0.5B draft + 1.5B target and verified identical output with higher throughput.

## Build it further

Extend the benchmark to three prompt lengths: 64, 256, and 512 tokens. Plot tokens/sec vs. prompt length for each config (matplotlib). Hypothesis first: which config degrades fastest and why? Then verify. Write a short conclusion (3 sentences) comparing memory-bandwidth-bound vs. compute-bound behavior across the configs.

---

## Stacks & alternatives

**vLLM — production high-throughput serving [CUDA]:**

PagedAttention manages KV cache as non-contiguous pages, eliminating wasted allocation for variable-length batches. Continuous batching pulls in new requests as slots free, giving 2–10x higher throughput than HF `transformers` under concurrent load. OpenAI-compatible endpoint, so any client works without changes. Reach for vLLM when you move from notebook to a service handling multiple users (see [Lab 7](16_lab_capstone.md)).

**TensorRT-LLM [CUDA, advanced]:**

Compiles models to hardware-specific TensorRT engines with fused attention + layer-norm kernels and INT8/INT4 GEMM. Higher peak throughput than vLLM on NVIDIA H100/A100, but requires a compilation step and the engine is device-specific. Worth it only after vLLM is the bottleneck.

**AWQ and GPTQ — alternative PTQ methods:**

Both improve on naive per-tensor quantization by accounting for weight importance:

```python
# AWQ (Activation-aware) — finds 1% of sensitive channels, protects them
from awq import AutoAWQForCausalLM
model_awq = AutoAWQForCausalLM.from_pretrained(MODEL_ID)
model_awq.quantize(tokenizer, quant_config={"w_bit":4,"q_group_size":128,"zero_point":True,"version":"GEMM"})
model_awq.save_quantized("./qwen-awq-4bit")

# GPTQ (Hessian-based per-layer error minimization)
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
model_gptq = AutoGPTQForCausalLM.from_pretrained(MODEL_ID, BaseQuantizeConfig(bits=4, group_size=128))
model_gptq.quantize(calibration_examples)
model_gptq.save_quantized("./qwen-gptq-4bit")
```

Decision rule: **AWQ** for best 4-bit accuracy; **GPTQ** if pre-quantized weights already exist on HF or you need wider kernel support; **bitsandbytes NF4** for zero prep time at ~10–15% throughput cost.
