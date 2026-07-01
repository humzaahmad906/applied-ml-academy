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

## Going deeper

- The arithmetic of rooflines, parallelism, and training FLOPs is the load-bearing skill in this chapter — drill it until the estimates are automatic.
- ZeRO/FSDP sharding stages, tensor/pipeline/expert parallelism, and their composition heuristics reward studying reference implementations directly (torchtitan and the major distributed-training frameworks).
- The recent frontier-model technical reports document MLA, aux-loss-free MoE balancing, FP8 training, MTP, and RLVR/GRPO at scale — the concrete recipes behind the concepts here.
- The post-training toolchain (SFT/DPO/GRPO trainers, embedded rollout engines, PEFT and QAT libraries) is where the ideas in this chapter become runnable; the Project below walks through the full pipeline on one GPU.

## Project 03 — A full post-training pipeline on one GPU

Take **Qwen3-0.6B** (or any ~0.5–1B base) and run the full modern pipeline on a single consumer GPU: (1) **SFT** on ~10k instruction examples with TRL + Unsloth; log tokens/sec and compute your achieved MFU against your GPU's peak BF16 FLOPs. (2) **DPO** on ~5k preference pairs; verify the KL-vs-win-rate tradeoff by sweeping β. (3) **GRPO** on GSM8K with an exact-match verifier (TRL's GRPOTrainer with vLLM rollouts); watch for and document one failure mode (zero-advantage groups or length inflation). (4) Evaluate all three checkpoints on a fixed eval set and write up which stage bought what. Stretch: repeat SFT with LoRA r=8 vs r=64 vs full FT and compare both quality and wall-clock.

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
