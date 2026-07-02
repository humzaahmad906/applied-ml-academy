# Module 03 — Training & Post-Training Infrastructure

## Why this module matters

You will probably never pretrain a frontier model, but interviews (and real jobs) constantly require you to (a) do training capacity math, (b) choose a parallelism strategy, and (c) design a post-training pipeline — SFT → preference optimization → RL — which in 2026 is the standard way teams turn open-weight models into products. This module is the densest in the course; budget two weeks.

## 1. The arithmetic of training

Memorize: **training FLOPs ≈ 6 × N × D** (N = parameters, D = training tokens; 2ND for the forward pass, ~4ND for backward). A 7B model on 1T tokens ≈ 4.2×10²² FLOPs. An H100 delivers ~1×10¹⁵ BF16 FLOP/s peak; at a realistic **MFU** (model FLOPs utilization) of 35–45%, that's ~3 days × 400 GPUs. Being able to produce this estimate live is a strong interview signal.

**Memory per parameter (mixed-precision Adam):** weights 2 bytes (BF16) + grads 2 + optimizer states 8 (FP32 master weights + two Adam moments... conventions vary, ~12–16 bytes total per param is the planning number) → a 7B model needs ~100 GB of training state before activations. This is why a 7B model does not fully fine-tune on one 80 GB GPU without sharding or tricks, and why activation **gradient checkpointing** (recompute activations in backward, ~30% compute for big memory savings) is on by default.

## 2. Parallelism — the 4D/5D menu

- **Data parallel (DP):** replicate the model, split the batch, all-reduce gradients. Simple; memory-limited.
- **ZeRO / FSDP:** still data parallel, but shard optimizer states (stage 1), + gradients (stage 2), + parameters (stage 3, equivalent to PyTorch **FSDP/FSDP2**) across the DP group, gathering params just-in-time per layer. This is the workhorse for fine-tuning 7–70B models on 8–64 GPUs. Know the stages.
- **Tensor parallel (TP):** split individual matmuls across GPUs (Megatron-style column/row splits). Requires all-reduce *inside every layer* → only viable over NVLink within a node (typically TP ≤ 8).
- **Pipeline parallel (PP):** split layers into stages across nodes; microbatches keep the pipeline full; the cost is "bubble" idle time. Cheap on bandwidth → used across nodes.
- **Context/sequence parallel (CP):** shard the sequence dimension (ring attention lineage) for long-context training.
- **Expert parallel (EP):** for MoE — distribute experts across GPUs, all-to-all routing of tokens. With MoE everywhere in 2026 (DeepSeek-V3, Qwen3-MoE, Mixtral lineage), EP is now part of the standard menu.

Composition heuristic for interviews: TP within a node, PP/FSDP across nodes, CP only for long context, EP if MoE; DP multiplies whatever is left. Frameworks: **torchtitan** (PyTorch-native reference), **Megatron-LM/Megatron-Core**, **DeepSpeed**.

## 3. Architecture-era notes that affect systems

- **MoE:** N total params but only k experts active per token → train/serve compute scales with *active* params while memory scales with *total*. Router load-balancing matters (DeepSeek-V3 popularized an aux-loss-free balancing scheme); serving implication: huge memory footprint, low per-token FLOPs → wide-EP serving (covered in the serving chapter).
- **MLA (multi-head latent attention, DeepSeek):** compresses KV into a low-rank latent → 5–10× smaller KV cache; GQA is the simpler mainstream version. KV-cache size is a *training-time architecture decision with serving consequences* — say this in interviews.
- **FP8 training** went mainstream after DeepSeek-V3 trained a frontier model in FP8 (Transformer Engine on Hopper/Blackwell): ~1.5–2× throughput; requires per-tensor scaling and careful accumulation.
- **MTP (multi-token prediction):** auxiliary heads predicting t+2... during training; improves quality and provides a free draft model for self-speculative decoding (see the inference chapter).

## 4. The post-training pipeline (the part you will actually run)

**Stage 0 — base model choice:** license, size vs latency target, architecture's export-compatibility with your serving stack (hybrid SSM/linear-attention blocks can break ONNX-style export paths — check *before* training, not after).

**Stage 1 — SFT.** Curated instruction/demonstration data, packed sequences, loss masked to assistant tokens. Failure modes: chat-template mismatch between training and serving (silent, catastrophic), low-diversity data (the fifty-paraphrases problem), too many epochs (memorization, capability regression). Hyperparameters matter less than data; eval on held-out *task* metrics, not loss.

**Stage 2 — Preference optimization.** **DPO** (Rafailov et al. 2023) made preference tuning a simple offline classification-style loss over (chosen, rejected) pairs — no reward model, no rollouts. Variants: **KTO** (binary thumbs up/down instead of pairs — matches real product telemetry), **SimPO/ORPO** (reference-free). DPO is cheap, stable, and the default second stage; its limit is that it can't discover behaviors absent from the preference data.

**Stage 3 — RL.** Two flavors:

- **RLHF (PPO + learned reward model):** the InstructGPT/Claude lineage for open-ended quality; expensive (policy + reference + reward + value models in memory).
- **RLVR — RL from verifiable rewards — with GRPO:** the post-DeepSeek-R1 paradigm and, by 2026, the dominant approach for post-training reasoning models: deterministic verifiers (exact-answer checks in math, unit tests for code) provide the reward, and **GRPO** eliminates the value model by normalizing rewards within a group of sampled completions per prompt. Know GRPO's mechanics: sample G completions per prompt, advantage = (reward − group mean)/group std, PPO-style clipped update. Know its failure modes: zero gradient when all samples in a group fail (exploration collapse on hard prompts), length bias, and **reward hacking** of any imperfect verifier; label noise in verifiable rewards degrades RLVR severely — training against noisy verifiers can be barely better than format-only rewards. Variants you can name-drop with understanding: DAPO, Dr. GRPO (bias corrections), GSPO (sequence-level).
- **Systems angle (the interview gold):** RL training is now an *inference-heavy* workload — rollout generation dominates wall-clock, so modern stacks (**verl**, **OpenRLHF**, **TRL**'s GRPOTrainer) embed a vLLM/SGLang engine for rollouts, often on separate "rollout" GPUs from the trainer — training is becoming disaggregated just like serving.

**Stage 4 — Distillation.** Teacher generates outputs (or token distributions) on your task distribution; student SFTs on them; optionally on-policy distillation (GKD) where the student's own samples are corrected by the teacher. R1-style reasoning-trace distillation showed small models inherit a surprising fraction of teacher reasoning. This is the standard way to hit a latency/cost target: prototype with the big model, distill into the small one.

## 5. PEFT and QAT

- **LoRA** (low-rank adapters on attention/MLP projections, ~0.1–1% trainable params) and **QLoRA** (base weights in 4-bit NF4, adapters in BF16) make 7–70B fine-tuning feasible on one or two consumer GPUs. Rule of thumb: LoRA matches full FT for style/format/narrow-domain adaptation; full FT (or high-rank LoRA on all layers) wins when injecting substantial new knowledge or doing RL.
- **QAT (quantization-aware training):** fake-quantize in the forward pass during fine-tuning so the model learns to be robust to its deployment precision (TorchAO is the PyTorch-native path). Key practitioner insight: **layer sensitivity is wildly non-uniform** — MLP blocks tolerate aggressive quantization; attention in-projections and SSM/linear-attention output projections are typically high-sensitivity and should be kept at higher precision or excluded. Building a per-layer sensitivity map before committing to a QAT recipe is the professional move (full story in the inference chapter).

---

## Foundations Box: Gradient Checkpointing / Recomputation

Standard backpropagation stores every layer's intermediate activations during the forward pass — the backward pass needs the activation of each layer to compute the gradient flowing through it. At batch size 4, sequence length 2048, hidden dim 4096, 32 layers, activation memory is on the order of tens of GB, often larger than parameter memory at that batch size.

`torch.utils.checkpoint.checkpoint(fn, *inputs)` discards all intermediate activations inside `fn` after the forward pass, keeping only the inputs. During the backward pass it reruns the forward through `fn` to regenerate what it needs. Each checkpointed block is computed twice; for a fully-checkpointed model this is roughly 1.33× the total forward FLOPs — the origin of the ~30% overhead figure. With selective checkpointing (wrapping only the heaviest blocks), overhead drops to 10–20%.

**Selective (compute-optimal) checkpointing** is the 2026 default: wrap the memory-heaviest sub-modules only — typically the full attention block, or every N layers — and let cheaper activations pass through. TorchTitan's `checkpoint_policy` and FSDP2's `activation_checkpointing_policy` let you declare a memory budget and automate the granularity choice.

Practical tradeoff on models you'll actually run:

- **7B LoRA SFT, 24 GB GPU:** checkpointing cuts activation memory from ~30–40 GB to ~5–8 GB at seq=2048, making single-GPU training viable. Without it, batch size 1 still OOMs.
- **70B full FT at 8K+ context:** activations alone exceed 80 GB per GPU even after FSDP parameter sharding. Checkpointing is non-negotiable; every reference implementation (Megatron-Core, torchtitan, TRL) enables it by default.

In an interview: "I'd enable selective gradient checkpointing on attention blocks — it's a ~15–20% compute cost for a 3–5× activation memory reduction, and at 70B with long context there's simply no alternative."

---

## Foundations Box: FlashAttention

Standard attention materializes an N×N score matrix in HBM: for N=8192 and BF16, that's 8192² × 2 bytes ≈ 128 MB per head per layer per example, and the backward needs it materialized too. Both memory and HBM bandwidth cost scale O(N²).

**FlashAttention's move** (Dao et al. 2022): never write the N×N matrix to HBM. Tile Q into row-blocks, sweep over K and V blocks held in SRAM, maintain a running log-sum-exp for numerically stable online softmax. The full output accumulates in O(1) SRAM per tile; only the final output (O(Nd)) and the log-sum-exp vectors (O(N)) touch HBM. Same FLOPs, radically fewer HBM reads/writes. Attention is memory-bandwidth-bound, not FLOP-bound — keeping intermediate data in SRAM (~10× faster than HBM on H100) is the mechanism of the speedup.

FA-2 (2023) added sequence-dimension parallelism within the CUDA kernel, roughly 2× over FA-1 on typical shapes. **FA-3** (2024) targets Hopper specifically: asynchronous Wgmma + TMA pipeline, native FP8 accumulation, approaching hardware peak on H100 SXM.

**Why it enables long-context training:** at 128K tokens, the raw N×N scores would be 128K² × 2 bytes per head per layer — tens of GB per layer. With FA, those scores never land in HBM; only O(N) tensors per layer are stored or checkpointed. This is the infrastructure change that made 32K–128K context training tractable.

**Key limitation interviewers probe:** FA accelerates prefill — computing attention over the full input. During autoregressive decode you attend one new token against the full KV cache. That is a KV-cache memory-bandwidth-bound read, not an FLOP-bound computation; FA's tiling provides no benefit. Decode-side optimizations — GQA/MLA to shrink the cache, paged attention for efficient allocation, speculative decoding — are the inference chapter's territory.

---

## Foundations Box: Measuring MFU Yourself

MFU = (FLOPs delivered per second) / (hardware peak FLOPs per second). The denominator is on the spec sheet; the numerator requires instrumentation.

**FLOPs per step:** the 6ND rule gives total training FLOPs; per step, FLOPs ≈ 6 × N × T where T is the global-batch token count (2NT forward + 4NT backward). At 7B parameters and 4096 tokens/step: 6 × 7×10⁹ × 4096 ≈ 1.7×10¹⁴ FLOPs.

**The `synchronize()` trap:** CUDA launches are asynchronous. `time.perf_counter()` around a training step without `torch.cuda.synchronize()` measures Python dispatch latency, not GPU execution time — the measured step will appear 50–100× faster than reality.

```python
import time
import logging
import torch

logger = logging.getLogger(__name__)

# H100 SXM5 BF16 ~989 TFLOP/s — representative as of 2026, verify your SKU spec
H100_BF16_PEAK = 989e12


def log_mfu(
    n_params: int,
    tokens_per_step: int,
    step_time_sec: float,
    peak_flops: float = H100_BF16_PEAK,
) -> float:
    """Compute and log MFU after a timed training step. Returns MFU in [0, 1]."""
    flops = 6 * n_params * tokens_per_step
    mfu = (flops / step_time_sec) / peak_flops
    logger.info("MFU=%.1f%%  achieved=%.2e FLOP/s", mfu * 100, flops / step_time_sec)
    return mfu


# In your training loop — skip step 0 (covers CUDA JIT compilation warmup):
torch.cuda.synchronize()
t0 = time.perf_counter()
loss = model(**batch)
loss.backward()
optimizer.step()
optimizer.zero_grad()
torch.cuda.synchronize()
mfu = log_mfu(7_000_000_000, global_batch_tokens, time.perf_counter() - t0)
```

**Realistic baselines (2026):**

- Dense transformer, single A100/H100 node, FA2 + fused ops: **35–50%**
- MoE: **20–35%** (all-to-all overhead; expert imbalance pushes it lower)
- FSDP across PCIe-connected nodes: **25–40%** (NVLink vs PCIe bandwidth gap is large)

Budget at ~40% MFU in capacity estimates — this is why the Q1 interview answer uses 40%, not 100%, and lands at ~190 GPUs instead of ~75. Log MFU every 10–50 steps to wandb; a sudden drop almost always traces to a data pipeline bottleneck or a misconfigured FSDP all-gather size, not the model. Teams that don't track it routinely run at 15–20% without knowing it.

---

## Training Dockerfile

The CUDA base image choice is load-bearing: NVIDIA's official PyTorch container (`nvcr.io/nvidia/pytorch`) ships with a pre-validated CUDA, cuDNN, and NCCL stack. FlashAttention must be compiled against the torch version it runs with — the NGC image guarantees this alignment and avoids the "torch version used to compile flash-attn does not match" runtime failure.

Multi-stage build separates flash-attn compilation (requires build tools + CUDA headers, slow) from the runtime image:

```dockerfile
# Stage 1 — compile Flash Attention
# Representative tag — check current NGC catalog at nvcr.io before using
FROM nvcr.io/nvidia/pytorch:24.04-py3 AS builder

RUN pip install --no-cache-dir ninja packaging && \
    pip install --no-cache-dir flash-attn --no-build-isolation

# Stage 2 — training runtime (build tools excluded from final layers)
FROM nvcr.io/nvidia/pytorch:24.04-py3

COPY --from=builder /usr/local/lib/python3.10/dist-packages/flash_attn* \
     /usr/local/lib/python3.10/dist-packages/

# Training dependencies — pin loosely; lock to a digest in CI after integration-testing
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    peft \
    trl \
    bitsandbytes \
    accelerate \
    wandb

WORKDIR /workspace
```

`--no-build-isolation` is required for flash-attn: without it, pip creates an isolated build environment, installs a fresh torch inside it, and compiles flash-attn against the wrong version. `bitsandbytes` requires a matching CUDA library — the NGC base handles this; a non-NGC base requires verifying the `libcuda.so` path. For the full inference/serving container, Kubernetes deployment, and CI/CD pipeline, see the deployment chapter — this image covers training only. For GPU pricing tables and TCO comparisons between cloud and on-prem training, see the economics chapter.

---

## Going deeper

- The arithmetic of rooflines, parallelism, and training FLOPs is the load-bearing skill in this chapter — drill it until the estimates are automatic.
- ZeRO/FSDP sharding stages, tensor/pipeline/expert parallelism, and their composition heuristics reward studying reference implementations directly (torchtitan and the major distributed-training frameworks).
- The recent frontier-model technical reports document MLA, aux-loss-free MoE balancing, FP8 training, MTP, and RLVR/GRPO at scale — the concrete recipes behind the concepts here.
- The post-training toolchain (SFT/DPO/GRPO trainers, embedded rollout engines, PEFT and QAT libraries) is where the ideas in this chapter become runnable; the Project below walks through the full pipeline on one GPU.

## Project 03 — A full post-training pipeline on one GPU

Take **Qwen3-0.6B** (or any ~0.5–1B base) and run the full modern pipeline on a single consumer GPU: (1) **SFT** on ~10k instruction examples with TRL + Unsloth; log tokens/sec and compute your achieved MFU against your GPU's peak BF16 FLOPs. (2) **DPO** on ~5k preference pairs; verify the KL-vs-win-rate tradeoff by sweeping β. (3) **GRPO** on GSM8K with an exact-match verifier (TRL's GRPOTrainer with vLLM rollouts); watch for and document one failure mode (zero-advantage groups or length inflation). (4) Evaluate all three checkpoints on a fixed eval set and write up which stage bought what. Stretch: repeat SFT with LoRA r=8 vs r=64 vs full FT and compare both quality and wall-clock.

### Walkthrough

#### Environment (representative packages as of 2026 — check current versions before pinning)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft trl accelerate bitsandbytes wandb
# Unsloth accelerates SFT on consumer GPUs — check repo for the correct extras tag
pip install unsloth
# GRPO with vLLM rollouts — verify trl/vllm version compatibility before installing
pip install vllm
```

**Dataset format.** SFT expects ChatML (preferred — matches most modern model tokenizers) or ShareGPT. The field names matter: TRL's SFTTrainer looks for a `messages` column with `role`/`content` dicts.

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Solve: 15 × 37"},
  {"role": "assistant", "content": "The answer is 555."}
]}
```

Public datasets to pull directly: `HuggingFaceH4/ultrachat_200k` (SFT, ChatML), `HuggingFaceH4/ultrafeedback_binarized` (DPO, `chosen`/`rejected` as message lists), `openai/gsm8k` (GRPO math, `question`/`answer`).

**SFT with TRL + QLoRA:**

```python
import logging
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(10_000))

sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch = 16
    gradient_checkpointing=True,
    bf16=True,
    max_seq_length=2048,
    packing=True,                    # fills context windows — critical for tokens/sec
    logging_steps=10,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)
trainer.train()
logger.info("SFT complete. Checkpoint at %s", sft_config.output_dir)
```

`packing=True` fills each context window with multiple short conversations — without it, a 32-token instruction wastes 98% of a 2048-length context and tokens/sec collapses by roughly the same ratio. `gradient_accumulation_steps=8` with batch 2 gives an effective batch of 16 without OOM.

**DPO stage.** Minimal change from SFT: `DPOTrainer` expects `prompt`, `chosen`, `rejected` fields (or message-list equivalents). The critical hyperparameter is `β`:

```python
from trl import DPOConfig, DPOTrainer

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                        # start here; sweep [0.01, 0.05, 0.1, 0.5]
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    report_to="wandb",
)
dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=None,                  # None + LoRA = implicit reference via disabled adapters
    args=dpo_config,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
)
dpo_trainer.train()
```

Watch `rewards/chosen`, `rewards/rejected`, and `kl` in wandb. The gap between chosen/rejected rewards is the win-rate proxy. If `kl` spikes past ~5–10 nats, raise `β`.

**GRPO/RLVR stage — real config complexity.** GRPOTrainer launches a vLLM engine for rollout generation internally. The version triple (TRL + vLLM + transformers) must be mutually compatible — test the import before submitting a long job:

```python
from trl import GRPOConfig, GRPOTrainer  # requires a TRL version that ships GRPOTrainer


def gsm8k_verifier(completions: list[str], ground_truths: list[str], **kwargs) -> list[float]:
    """Exact-match reward: 1.0 for correct final answer after ####, 0.0 otherwise."""
    rewards = []
    for pred, gt in zip(completions, ground_truths):
        pred_ans = pred.split("####")[-1].strip() if "####" in pred else ""
        rewards.append(1.0 if pred_ans == gt.strip() else 0.0)
    return rewards


grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,               # G completions per prompt (group size for GRPO)
    max_new_tokens=512,
    temperature=0.9,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    use_vllm=True,
    report_to="wandb",
)
```

Watch `train/reward_std` per step. Consistently near zero means every prompt's group has identical outcomes — GRPO produces no gradient (zero-advantage groups). Fix: filter to medium-difficulty prompts; reduce temperature slightly.

**OOM handling on 24 GB.**

| Technique | Memory impact | Setting |
| ----------- | -------------- | ------- |
| 4-bit NF4 QLoRA | ~4× reduction in param memory | `BitsAndBytesConfig(load_in_4bit=True)` |
| Gradient checkpointing | 60–80% activation memory | `gradient_checkpointing=True` |
| Reduce per-device batch + increase accum | Linear in ratio | `per_device_train_batch_size=1, gradient_accumulation_steps=32` |
| Flash Attention 2 | Avoids N×N HBM OOM at long context | `attn_implementation="flash_attention_2"` |
| Sequence packing | No memory save; maximizes tokens/step | `packing=True, max_seq_length=2048` |
| LoRA instead of full FT | Only adapters trainable; base frozen | `peft_config=LoraConfig(...)` |

**Expected log output (healthy SFT run, 4090, QLoRA, packing):**

```text
wandb: Syncing run ... → https://wandb.ai/your-entity/...
Step  10: {'loss': 2.12, 'grad_norm': 1.38, 'learning_rate': 2.00e-4, 'epoch': 0.08}
Step  20: {'loss': 1.89, 'grad_norm': 1.21, 'learning_rate': 1.98e-4, 'epoch': 0.16}
Step  50: {'loss': 1.64, 'grad_norm': 1.07, 'learning_rate': 1.92e-4, 'epoch': 0.40}
tokens/sec: ~2 000–5 000 (varies with packing efficiency and sequence length)
```

Loss dropping steadily from ~2.0 toward ~1.0–1.2 is healthy. Flat loss from step 1 almost always means a chat-template bug — the model is predicting every token, not just assistant turns. Verify with `tokenizer.apply_chat_template([...], tokenize=False)` and confirm the loss mask covers only assistant tokens.

**Troubleshooting.**

| Symptom | Likely cause | Fix |
| --------- | ------------- | ----- |
| CUDA OOM at step 0 | Batch too large or checkpointing off | `per_device_train_batch_size=1` + `gradient_checkpointing=True` + 4-bit |
| CUDA OOM mid-training (step 50+) | Packing producing over-length sequences | Set `max_seq_length` explicitly in SFTConfig |
| `ImportError: cannot import GRPOTrainer` | TRL version predates GRPOTrainer | `pip install --upgrade trl`; check TRL CHANGELOG for the version it landed |
| vLLM launch error inside GRPOTrainer | API mismatch between TRL and vLLM | Pin both to a combination listed in TRL's tested requirements |
| Loss flat from step 1 | Chat-template mismatch — model predicts all tokens | Use `DataCollatorForCompletionOnlyLM` with correct `response_template`; verify token ids |
| DPO loss goes negative immediately | `β` too low or wrong ref\_model | Raise `β`; confirm ref\_model is the pre-DPO SFT checkpoint |
| GRPO `reward_std ≈ 0` every step | All-pass or all-fail groups | Filter to intermediate difficulty; raise `temperature` |

## Interview Q&A

**Q1. Estimate the GPUs needed to train a 7B model on 2T tokens in two weeks.**
**A.** FLOPs = 6ND = 6 × 7×10⁹ × 2×10¹² = 8.4×10²² FLOPs. One H100 ≈ 10¹⁵ BF16 FLOP/s peak; at 40% MFU ≈ 4×10¹⁴ sustained. Per GPU over 14 days: 4×10¹⁴ × 1.21×10⁶ s ≈ 4.8×10²⁰ FLOPs. So ~8.4×10²²/4.8×10²⁰ ≈ **175 GPUs**, call it ~190–200 with restarts/eval overhead — e.g., 24 nodes of 8×H100. Follow-up you should volunteer: parallelism plan = FSDP/ZeRO-3 across nodes is sufficient at 7B (no TP/PP needed), BF16 + gradient checkpointing, global batch ~4M tokens.

**Q2. Explain ZeRO stages 1–3 and when you'd use FSDP vs tensor parallelism.**
**A.** All ZeRO stages are data parallelism with progressively more sharding across the DP group: stage 1 shards optimizer states (the ~8–12 bytes/param of Adam state), stage 2 adds gradient sharding, stage 3 adds parameter sharding with just-in-time all-gather per layer — FSDP is PyTorch's native stage-3 equivalent. Use FSDP when the model fits per-GPU *compute*-wise but not *memory*-wise and you can tolerate its communication (overlappable with compute); it's the default for ≤70B fine-tuning. Use TP when a single layer's working set or latency demands splitting the matmuls themselves — TP communicates activations inside every layer, so it needs NVLink-class bandwidth and stays within a node (TP≤8). At large scale you compose: TP=8 in-node, PP or FSDP across nodes, and the decision driver is interconnect bandwidth at each level of the hierarchy.

**Q3. DPO vs PPO-RLHF vs GRPO-RLVR — when each?**
**A.** **DPO** when you have (or can synthesize) preference pairs and want cheap, stable alignment to style/safety/format — offline, no rollouts, fits on small hardware; limited to behaviors represented in the pairs. **PPO-RLHF** when the objective is fuzzy human preference on open-ended tasks and you can afford a reward model plus on-policy rollouts — highest ceiling for general assistants, highest cost and instability (reward hacking against a learned RM). **GRPO-RLVR** when correctness is *checkable* — math, code with unit tests, extraction with exact-match, tool-use with success conditions: the verifier replaces the reward model, GRPO replaces the value network, and you get the strongest reasoning gains per dollar; constraints are needing a reliable verifier (noisy verifiers poison training), sparse rewards on hard prompts, and length/format hacking. Real pipelines stack them: SFT → DPO → RLVR.

**Q4. Your SFT'd model performs great in your eval harness but badly in the product. Top suspects?**
**A.** Ranked: (1) **chat-template/tokenization mismatch** — training applied a different template, system-prompt convention, or special tokens than the serving stack; verify by comparing the exact token ids of one prompt in both paths. (2) **Distribution shift** — eval set drawn from training distribution, production inputs are longer/noisier/different language; check slice metrics on real logged traffic. (3) **Sampling config mismatch** — eval at temperature 0, production at 0.8 with different stop tokens or max-token truncation. (4) **Quantization gap** — you evaluated the BF16 checkpoint but serve a 4-bit quant; always run final evals on the *deployment artifact*. (5) **Context construction differences** — RAG/preamble in production that the eval harness omits. The meta-answer: make the eval harness call the *production serving path*, not the training code.

**Q5. Why did RLVR/GRPO become the dominant reasoning post-training method, and what breaks it?**
**A.** Three reasons: (1) it removes the two most fragile pieces of PPO-RLHF — the learned reward model (replaced by a deterministic verifier, eliminating that reward-hacking surface) and the value network (GRPO's group-relative advantage is a Monte-Carlo baseline) — making large-scale RL dramatically simpler and cheaper; (2) verifiable domains (math, code, agent tasks with success checks) are exactly where pure SFT plateaus, and reasoning-focused RLVR runs showed it elicits emergent long-chain reasoning; (3) infrastructure matured — post-training frameworks with embedded rollout engines made it accessible. What breaks it: no verifier for the task (open-ended writing), noisy/gameable verifiers (the model learns the verifier's bugs — formatting tricks, test-case overfitting), zero-gradient groups when every sample fails on hard prompts (mitigations: curriculum, difficulty filtering), and overlong-response/length bias requiring penalty terms. Plus a systems cost juniors miss: rollout generation dominates compute, so the bottleneck is inference throughput, not gradient steps.
