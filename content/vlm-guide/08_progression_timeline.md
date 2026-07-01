# The Progression: A Chained Lineage (2017 → 2026)

This is the spine of the whole guide turned into a story. The point is not a list of papers — it's the **causal chain**: each entry says what was introduced, **what it fixed about the thing before it**, and **what limitation remained that motivated the next thing**. Read top to bottom and you watch the field reason its way from one idea to the next. When you then open a 2026 paper, you'll recognize it as the next link in one of these chains.

**How it's organized.** §1 is the **main spine** — architecture + pretraining + alignment, the backbone everything hangs off, 2017→2026. Then five **parallel tracks**, each its own self-contained chained timeline so the cause-and-effect stays clean: §2 efficiency/inference, §3 reasoning/test-time compute, §4 vision/multimodal, §5 RAG/retrieval, §6 agents. §7 is the meta-pattern for placing any new paper. The tracks overlap in time and reference each other (that's the nature of the field); cross-refs point you around.

**Format of each entry:** **Name (lab, date) — core idea.** What it fixed → what remained → (the arrow `⇒` is "this motivated the next entry").

**A dating note:** everything through ~2024 is settled history. From 2025 on, model names/versions proliferate weekly and many "frontier" specifics are undisclosed; I give the *architecturally and methodologically significant* moves confidently and treat point-releases as trend illustrations, not a list to memorize.

---

## Part 0 — The 30-second prehistory (what the Transformer replaced)

You only need this to understand *why 2017 was a break*. These are the missing concepts that motivate everything after.

- **Distributed word representations — Word2Vec (Mikolov, 2013) & GloVe (2014).** Represent each word as a dense vector learned from co-occurrence, so similar words sit nearby and analogies work as vector arithmetic. This is the ancestor of all embeddings (both the token embeddings in the foundations and the retrieval embeddings in the RAG chapter). *Limit:* one fixed vector per word regardless of context ("bank" of a river vs money).
- **RNNs / LSTMs (1997, popularized ~2014).** Process a sequence one token at a time, carrying a hidden state forward. LSTMs added gates to mitigate vanishing gradients and remember longer. *Limit:* strictly **sequential** (can't parallelize over the sequence) and they **forget** over long distances — the hidden state is an information bottleneck.
- **Seq2Seq (Sutskever et al., 2014).** Encoder RNN compresses the whole input into one fixed vector; decoder RNN generates the output from it. Enabled translation/summarization. *Limit:* the single fixed "thought vector" is a brutal bottleneck — long inputs get squashed.
- **Attention, the original (Bahdanau et al., 2014; Luong 2015).** Instead of one fixed vector, let the decoder, at each output step, **attend** over *all* encoder states with learned weights ("soft alignment"). This is literally the attention mechanism from the foundations, invented to fix the seq2seq bottleneck — but it was *bolted onto RNNs*, so it inherited the sequential, slow nature.
- **ELMo (2018).** Contextual word vectors from a bidirectional LSTM LM — "bank" now gets different vectors in different sentences. Fixed Word2Vec's context-blindness, still RNN-based and slow.

The setup by mid-2017: attention clearly worked, recurrence was the bottleneck. The obvious-in-hindsight question: *what if attention is all you need, and we throw the RNN away entirely?*

---

## Part 1 — The Main Spine: architecture + pretraining + alignment

### 2017 — the break
**1. Attention Is All You Need (Vaswani et al., Google, Jun 2017) — the Transformer.** Remove recurrence and convolution *entirely*; build the model from **self-attention + FFN** stacked with residuals and layer norm (the transformer block from the foundations). Because there's no sequential dependency across positions, the whole sequence trains **in parallel** on GPUs — and self-attention connects any two positions in one step (no long-distance forgetting). This single change unlocked scale. Original form: encoder–decoder for translation, with sinusoidal positions and Multi-Head Attention. ⇒ If this architecture scales so well, what happens if you *pretrain* it on huge text and adapt it to tasks?

### 2018 — the pretraining paradigm splits in two
**2. GPT-1 (Radford et al., OpenAI, Jun 2018) — generative pretraining + fine-tuning.** Take a **decoder-only** Transformer, pretrain it as a plain next-token language model on unlabeled text, then fine-tune on each downstream task. Showed unsupervised pretraining transfers broadly. *Limit:* unidirectional (left-to-right only), and still needs task-specific fine-tuning. ⇒ For *understanding* tasks, wouldn't seeing both directions help?

**3. BERT (Devlin et al., Google, Oct 2018) — bidirectional masked-LM pretraining.** Use an **encoder-only** Transformer; pretrain by **masking** random tokens and predicting them (so every token sees full left+right context) plus next-sentence prediction. Crushed understanding benchmarks; defined the "pretrain then fine-tune" era for classification/QA/embeddings. *Limit:* not generative (the masking objective doesn't produce fluent text) — great at understanding, can't write. ⇒ The field forks: BERT-style (understanding) vs GPT-style (generation). Generation wins the scaling race; here's why.

*(Also 2018: ULMFiT formalized transfer learning for NLP; RoBERTa (2019) showed BERT was undertrained and dropped NSP.)*

### 2019 — scale and unification
**4. GPT-2 (OpenAI, Feb 2019) — scale + zero-shot multitask.** Same decoder-only recipe, 10× bigger (1.5B), more data. Discovery: at scale the model does tasks **zero-shot** from a prompt alone, with no fine-tuning ("language models are unsupervised multitask learners"). *Limit:* zero-shot was inconsistent; clearly more scale was needed to make it reliable. ⇒ Push scale much harder.

**5. T5 (Google, Oct 2019) — everything is text-to-text.** Cast *every* NLP task (translation, classification, QA, summarization) as "text in → text out" with a unified **encoder–decoder** Transformer, pretrained on the cleaned C4 web corpus with a span-corruption objective. Unified the interface and ran the era's most systematic ablations. The encoder–decoder lineage (later BART, and many early VLMs) lives here. ⇒ Meanwhile, how far does *pure decoder-only scale* go?

### 2020 — emergence of in-context learning, and the first scaling map
**6. GPT-3 (OpenAI, May 2020) — in-context / few-shot learning.** 175B params. The breakthrough wasn't just scale — it was that you could give the model a few examples *in the prompt* and it would do the task with **no weight updates** (the in-context-learning property from the LLM chapter). "Prompting" became a discipline. This is the moment LLMs became general-purpose. *Limit:* it followed *patterns*, not *intentions* — ask a question and it might continue with more questions; not yet an assistant. Also: was 175B even the right size? ⇒ Two open questions: (a) how to make it *follow instructions/help*, (b) what's the *compute-optimal* size.

**7. Scaling Laws (Kaplan et al., OpenAI, Jan 2020) — loss is a predictable power law.** Showed test loss falls smoothly and predictably with model size, data, and compute, so you can *forecast* a big run from small ones. Justified the scaling bets. *Limit:* their recommended size/data tradeoff turned out to be wrong (too-big, too-little-data) — corrected in 2022 by Chinchilla. ⇒ (see entry 11)

### 2021 — efficiency knobs and the instruction-following seed
**8. RoPE / RoFormer (Su et al., Apr 2021) — rotary position embeddings.** Encode position by *rotating* Q/K so attention scores depend on *relative* offset, with no learned position params and better length behavior (the RoPE material in the foundations). Became the near-universal positional scheme. ⇒ (feeds every later model)

**9. FLAN / T0 (Google, Stanford, late 2021) — instruction tuning.** Fine-tune on many tasks *phrased as instructions*; the model then follows *unseen* instructions zero-shot far better. The seed of "instruction-following," the precursor to SFT (covered in the LLM chapter). *Limit:* instruction-tuning teaches format, not nuanced human preference. ⇒ Add a preference signal (entry 13).

*(Also 2021, efficiency track: Switch Transformer simplified MoE to top-1 routing at trillion-param scale — see §2; LoRA introduced cheap adapters — see §2.)*

### 2022 — the corrective, the reasoning spark, and alignment
**11. Chinchilla (Hoffmann et al., DeepMind, Mar 2022) — compute-optimal scaling (~20 tokens/param).** Re-ran the scaling analysis and found nearly all prior models (GPT-3, Gopher) were **too big and trained on too little data**. A *smaller* model on *more* tokens beats a bigger undertrained one at equal compute. Reframed the whole field toward train-longer-smaller. *Limit:* it optimizes *training* compute, ignoring that you serve a model millions of times — later models deliberately "overtrain" small models for cheap inference (see the scaling-laws discussion in the LLM chapter). ⇒ (shapes Llama, entry 15)

**12. Chain-of-Thought prompting (Wei et al., Google, Jan 2022) + Self-Consistency (Mar 2022) + Emergent Abilities (Jun 2022).** Just prompting the model to "think step by step" massively improves reasoning — and this ability *emerges* sharply only at scale. Self-consistency (sample many chains, take the majority) improves it further. This is the conceptual seed of the entire reasoning-model era (see the reasoning-model material in the LLM chapter, and §3 below). ⇒ Can we *train* the model to do this natively rather than prompt it? (§3)

**13. InstructGPT (Ouyang et al., OpenAI, Mar 2022) — RLHF.** The recipe that made models *assistants*: SFT on demonstrations → train a **reward model** on human preference pairs → **RL (PPO)** to maximize reward with a KL leash (the RLHF pipeline in the LLM chapter). A 1.3B InstructGPT was preferred over 175B GPT-3 — *alignment*, not size, drove perceived quality. *Limit:* RLHF is a complex, unstable four-model pipeline. ⇒ Simplify it (DPO, entry 18).

**14. Constitutional AI / RLAIF (Anthropic, Dec 2022).** Replace much of the *human* feedback with **AI feedback** guided by a written set of principles ("constitution"), making alignment more scalable and steerable. ⇒ (feeds the modern post-training stack)

**ChatGPT (Nov 30, 2022)** — InstructGPT-style alignment in a chat wrapper. Not a paper, but *the* inflection point: it made LLMs mainstream and kicked off the arms race that the rest of this timeline races through.

### 2023 — the open-model wave and the alignment shortcut
**15. LLaMA (Meta, Feb 2023) — the efficient open recipe.** A clean decoder-only stack — **RMSNorm + SwiGLU + RoPE**, no bias terms — trained on far more tokens than Chinchilla-optimal to be cheap to run. LLaMA-13B matched GPT-3-175B. It set the *de-facto architecture template* every open model copied, and (via leak then Llama 2's open license) ignited the open ecosystem. ⇒ Everyone now needs to fine-tune these cheaply and align them simply.

**16. GPT-4 (OpenAI, Mar 2023) — the multimodal frontier.** Large (undisclosed, widely believed MoE), accepts images, far stronger reasoning. Defined the closed-frontier bar for ~2 years. ⇒ (the target everyone chases)

**17. Llama 2 (Meta, Jul 2023).** Open weights with a permissive license, a documented RLHF pipeline, and **GQA** on the larger sizes (the attention variants covered in the LLM chapter and in §2). Made aligned open models real. ⇒

**18. DPO (Rafailov et al., May 2023) — preference alignment without RL.** Proved you can reparameterize the RLHF objective into a **simple classification loss** on (prompt, chosen, rejected) triples — no reward model, no PPO, no sampling loop (the preference-optimization material in the LLM chapter). Stable, simple; became the default preference method. *Limit:* preferences are *subjective* — for math/code there's a *correct answer* you could check instead. ⇒ Use verifiable rewards (§3, the R1 line).

**19. Mistral 7B (Sep 2023) & Mixtral 8×7B (Dec 2023).** Mistral: a small model punching far above its weight (sliding-window attention + GQA). Mixtral: the first widely-used **open sparse MoE** — 8 experts, top-2, ~47B total/~13B active (the MoE material in the LLM chapter and in §2). ⇒ MoE goes mainstream-open.

**20. Mamba (Gu & Dao, Dec 2023) — selective state-space models.** A serious **sub-quadratic, no-KV-cache** alternative to attention, with input-dependent (selective) state to recover content-routing (the sub-quadratic-mixer material in the LLM chapter and in §2). *Limit:* weaker than attention at exact long-range *retrieval*. ⇒ Don't replace attention — *hybridize* (2025, entry 27).

### 2024 — efficiency frontier, long context, and the reasoning turn
**21. Llama 3 / 3.1 (Meta, Apr–Jul 2024).** Pushed the open recipe hard: Llama-3-8B on **15T tokens** (~1800 tokens/param — far past Chinchilla, the inference-aware overtraining bet), then **405B** matching closed frontier. GQA across the line, 128k context. ⇒ Architecture is converging; the next gains come from *efficiency* and *post-training*.

**22. DeepSeek-V2 (May 2024) → DeepSeek-V3 (Dec 2024) — the efficient-flagship template.** V2 introduced **MLA** (multi-head latent attention — compress the KV cache to a low-rank latent, as covered in the LLM chapter) and **DeepSeekMoE** (fine-grained + shared experts). V3 (671B total/**37B active**) added an **auxiliary-loss-free** load-balancing scheme, **FP8** training, and **multi-token prediction** — frontier quality at a fraction of the training/inference cost, fully open. This is *the* reference design for modern efficient large models. ⇒ Now apply post-training to make it *reason*.

**23. o1 (OpenAI, Sep 2024) — inference-time compute as a first-class axis.** A model trained (via RL) to produce a long internal chain-of-thought before answering, and to *spend more thinking on harder problems* (the test-time-compute material in the LLM chapter and in §3). Established a **second scaling axis** orthogonal to model size: scale *thinking at inference*. Reasoning models become a category. *Limit:* the method was secret. ⇒ DeepSeek reveals (and open-sources) the recipe.

### 2025 — the reasoning explosion, hybrids go mainstream, multipolar frontier
**24. DeepSeek-R1 / R1-Zero (Jan 22, 2025) — reasoning from pure RL with verifiable rewards.** Applied **GRPO** (critic-free, group-relative RL, from DeepSeekMath Feb 2024) with **RLVR** (rule-based, *verifiable* correct/incorrect rewards — no reward model) on top of V3. The shock: **R1-Zero, with no SFT at all**, *spontaneously* learned long CoT, self-verification, backtracking, "aha moments." R1 added a small cold-start SFT for readability. Then distilled the traces into small Qwen/Llama models. This reframed post-training around verifiable rewards and is the single most-copied 2025 method (the RLVR/GRPO material in the LLM chapter and in §3). ⇒ Everyone ships a reasoning tier (Kimi 1.5, QwQ, o3, Gemini Deep Think, Claude extended thinking, Phi-4-reasoning).

**25. Qwen3 (Alibaba, Apr 2025) — unified hybrid-thinking, fully open.** One model line that switches between **thinking and non-thinking** modes, trained on ~36T tokens, shipped as both **dense and MoE** (e.g. 235B-A22B), competitive with R1/o1. Made strong reasoning *open and self-hostable* across sizes. ⇒

**26. Llama 4 (Meta, 2025) — natively multimodal MoE at huge context.** MoE (Scout/Maverick), **early-fusion** multimodality (the fusion spectrum in the VLM chapter), and a very large context window (multi-million tokens). Pushed native-multimodal + MoE + long-context into the open mainstream. ⇒

**27. The hybrid-attention turn — Qwen3-Next (2025) → Qwen3.5 / Kimi Linear (late 2025–2026).** The Mamba/linear-attention thread (entry 20) and full attention *combine*: interleave mostly-cheap linear/SSM blocks with periodic full-attention blocks (commonly **3:1**). Qwen3-Next (Gated DeltaNet + gated attention) proved it near-flagship; **Qwen3.5** promoted the hybrid into the *main* flagship line (signal that the strategy won), and **Kimi Linear** refined it (channel-wise-gated Kimi Delta Attention + gated MLA). NVIDIA's **Nemotron 3** (Dec 2025) shipped a hybrid **Mamba-Transformer MoE**. This is the live architectural frontier (the hybrid-mixer material in the LLM chapter and in §2). ⇒

**28. Kimi K2 (Moonshot, Jul 2025) + K2 Thinking (Nov 2025).** A very large **open** MoE (≈1T total) built on the V3-style template (MLA, *more* experts), with a strong "thinking" variant and a focus on agentic/tool use. Showed open models matching closed frontier on coding/agents. ⇒

**The closed frontier in 2025 → mid-2026:** GPT-5 (then 5.1/5.4/5.5), **Gemini 3.0** (Nov, top of benchmarks), **Claude 4 → 4.5 → 4.6/4.7/4.8** (Sonnet 4.6 brought 1M-token context; Opus 4.8 pushed browser/computer-use agents into production territory) and the **Fable 5** class (Jun 2026), all converging on the same themes: reasoning by default, native multimodality, and **agentic** capability (computer use, long-horizon coding).

### 2026 — multipolar, efficient, agentic
By 2026 the field is **multipolar** (Chinese open-weight labs — DeepSeek, Qwen, Kimi, GLM, MiniMax — within ~1–2 points of the closed frontier on coding) and **hardware-diverse** (DeepSeek **V4-Pro**, Apr 2026: **1.6T total / 49B active** params, trained on non-Nvidia Huawei Ascend, breaking the GPU monopoly — introducing **mHC** manifold-constrained hyper-connections for training stability, **Engram** conditional memory separating static knowledge lookup from dynamic reasoning, and a hybrid **CSA+HCA** attention cutting inference FLOPs up to ~73%). Architecturally the consolidation is clear: **MoE + hybrid-linear-attention + reasoning-by-default + native-multimodal** is the standard recipe; **self-verifiable reasoning** (DeepSeekMath-V2, Nov 2025, generating *and checking* its own proofs) extends RLVR to domains without easy verifiers; and **agentic/long-horizon coding** is the dominant application target (specialized agent models like the GPT-5.x Codex-Max line). Point-releases (GPT-5.x, Claude 4.x, Gemini 3.x, Qwen3.x) now ship every few days — the *trends* above matter; the version numbers don't.

**The spine in one breath:** *remove recurrence (2017) → pretrain at scale (GPT/BERT, 2018–20) → discover in-context learning and scaling laws (2020) → correct the scaling recipe (Chinchilla 2022) → align into assistants (RLHF 2022, simplified by DPO 2023) → open the recipe (LLaMA 2023) → make it efficient (MoE, MLA, FP8: DeepSeek 2024) → make it reason via verifiable-reward RL (o1 2024, R1 2025) → make attention sub-quadratic via hybrids and turn it agentic (2025–26).*

---

## Part 2 — Efficiency & Inference Lineage

The whole track is the war on attention's `O(n²)` and the KV-cache memory wall (the attention material in the foundations, and the inference chapter).

**KV-cache / attention shape:**
**MHA (2017)** full but huge cache ⇒ **MQA (Shazeer 2019)** one shared KV head, tiny cache but quality drop ⇒ **GQA (Ainslie 2023)** grouped KV heads, the tunable sweet spot (Llama 2/3, most models) ⇒ **MLA (DeepSeek-V2 2024)** compress KV to a low-rank latent — MHA-level quality at a fraction of the cache, with a decoupled-RoPE trick. (LLM chapter)

**Sparse capacity (MoE):**
**Outrageously Large NN (Shazeer 2017)** MoE on LSTMs, decouple params from compute ⇒ **GShard (2020)** MoE for Transformers + sharding ⇒ **Switch Transformer (2021)** top-1 routing, trillion params, simpler ⇒ **Mixtral (2023)** first big *open* MoE ⇒ **DeepSeekMoE / V3 (2024)** fine-grained + shared experts, *auxiliary-loss-free* balancing — the modern standard. (LLM chapter)

**Sub-quadratic sequence mixing:**
**Linear/efficient attention & RWKV (2020–22)** kernelize away `n²`, often quality gaps ⇒ **Mamba (2023)** selective SSM, strong but weak at exact retrieval ⇒ **Hybrids (Qwen3-Next/Qwen3.5, Kimi Linear, Nemotron 3, MiniMax-M1 lightning attention, 2025–26)** mix cheap linear/SSM blocks with periodic full-attention to get both. (LLM chapter)

**Exact-attention IO efficiency:**
naive attention materializes the `n×n` matrix (memory-bound) ⇒ **FlashAttention (Dao 2022)** tile + online-softmax, never materialize it, `O(n)` memory, *exact* ⇒ **FA-2 (2023)** better parallelism ⇒ **FA-3 (2024)** Hopper warp-specialization + FP8 ⇒ FlashDecoding/FlashInfer for decode + sparsity. (inference chapter)

**Serving:**
contiguous per-request KV → huge fragmentation ⇒ **PagedAttention / vLLM (Kwon 2023)** paged, shareable KV like OS virtual memory, enabling **continuous batching** and **prefix caching** ⇒ FP8 KV-quant, chunked prefill, prefill/decode disaggregation. (inference chapter)

**Compression:**
**LoRA (2021)** low-rank adapters, train <1% of params ⇒ **QLoRA (2023)** LoRA on a 4-bit-quantized base, big models on one GPU ⇒ PTQ **GPTQ (2022)** / **AWQ (2023)** 4-bit weights, activation-aware ⇒ **SmoothQuant** (W8A8 via outlier migration), **GGUF** (llama.cpp consumer), **QAT** + layer-sensitivity maps for on-device int4. (inference chapter)

**Decode acceleration:**
sequential one-token decode wastes idle GPU compute ⇒ **Speculative decoding (Leviathan 2023)** draft-and-verify in parallel, *identical* output ⇒ **Medusa (2024)** extra heads, no draft model ⇒ **EAGLE / EAGLE-3 (2024–25)** feature-level drafting, current SOTA. (inference chapter)

---

## Part 3 — Reasoning & Test-Time-Compute Lineage

How "think before answering" went from a prompt trick to a trained capability and a scaling axis (the reasoning-model material in the LLM chapter).

**Scratchpads (2021)** show your work in intermediate tokens ⇒ **Chain-of-Thought prompting (Wei 2022)** "think step by step" unlocks latent reasoning at scale ⇒ **Self-Consistency (2022)** sample many chains, majority-vote ⇒ **STaR (Zelikman 2022)** bootstrap: keep CoTs that reach correct answers, fine-tune on them, repeat — *train* reasoning from its own correct traces ⇒ **Let's Verify Step by Step / PRMs (OpenAI 2023)** reward *each step*, not just the final answer (process vs outcome supervision) ⇒ **Tree-of-Thoughts / LATS (2023)** search over reasoning branches with backtracking ⇒ **Quiet-STaR (2024)** learn to think between *every* token ⇒ **o1 (OpenAI 2024)** RL-trained long CoT + **test-time compute scaling** as a first-class, secret method ⇒ **DeepSeek-R1 / GRPO / RLVR (Jan 2025)** open it: verifiable rewards + critic-free group-relative RL; reasoning *emerges* from pure RL (R1-Zero), then distill into small models ⇒ **Dr. GRPO / DAPO (2025)** fix length/difficulty biases in GRPO's normalization; **process reward models** and turn-level credit assignment mature ⇒ **DeepSeekMath-V2 (late 2025)** *self-verifiable* reasoning (the model generates *and* checks its own proofs), extending verifiable-reward RL to domains lacking cheap external verifiers. Open debate throughout: does RLVR *create* new reasoning or *elicit* what pretraining already had, and when should the model *stop* thinking (overthinking hurts).

---

## Part 4 — Vision & Multimodal Lineage

How pixels learned to talk to language models (the VLM chapter). Two feeder threads (vision backbones, image generation) converge into VLMs.

**Vision backbones:** **AlexNet (2012)** CNNs win vision ⇒ **ResNet (2015)** residual connections train very deep nets (and, note, *residuals are what the Transformer later borrows*) ⇒ **ViT (Dosovitskiy 2020)** drop convolutions: cut the image into patches and run a *Transformer* — unifies the architecture with NLP (the vision-encoder material in the VLM chapter).

**Language-aligned encoders:** **CLIP (Radford 2021)** + **ALIGN (2021)** contrastively align image and text encoders on web-scale pairs → vision features *already in language space*, zero-shot classification ⇒ **SigLIP (2023) / SigLIP 2 (2025)** sigmoid loss, better/multilingual, the default VLM encoder ⇒ **DINO / DINOv2 (2021–23)** self-supervised, strong *spatial/geometric* features where CLIP is weak. (VLM chapter)

**Vision → LLM fusion (the main VLM line):** **Flamingo (DeepMind 2022)** inject frozen-LLM **cross-attention** + a Perceiver resampler, few-shot multimodal (the deep-fusion end of the VLM fusion spectrum) ⇒ **BLIP / BLIP-2 (2022–23)** the **Q-Former** compresses vision to a few learned query tokens ⇒ **LLaVA (2023)** the dominant simplification: a tiny **MLP projector** maps patches to visual tokens, *concatenate* with text into one decoder, two-stage train (align → instruction-tune) — "visual instruction tuning" (the projector and training-recipe material in the VLM chapter) ⇒ **Qwen2-VL / InternVL (2024)** native **dynamic resolution** + **M-RoPE** + token compression for documents/charts ⇒ **early/native fusion: Chameleon (Meta 2024), Emu3 (2024)** drop the bolted-on encoder entirely — tokenize images (often via VQ) into the *same vocabulary* and train one model on interleaved streams, enabling **image generation** too ⇒ **Llama 4 / Gemini / GPT-4o-style (2025–26)** natively-multimodal frontier (vision stacks largely undisclosed). The arc: *late fusion (CLIP) → cross-attention (Flamingo) → prefix-concat (LLaVA) → early-native (Chameleon)* — barely-interact to one-unified-model. (VLM chapter)

**Image generation (feeds unified models):** **DDPM (Ho 2020)** diffusion: generate by iterative denoising ⇒ **Latent Diffusion / Stable Diffusion (2022)**, DALL·E 2 — high-quality text-to-image ⇒ this capability gets folded into discrete-token unified models (Chameleon/Emu3) and 2025–26 "any-to-any" systems (e.g. Gemini-based image models).

---

## Part 5 — RAG & Retrieval Lineage

Giving frozen models fresh/private knowledge and grounding (the RAG chapter).

**DrQA (2017)** open-domain QA = retrieve Wikipedia + read ⇒ **DPR (Karpukhin 2020)** learn **dense** bi-encoder retrieval (beats BM25 on semantics) (RAG chapter) ⇒ **REALM (2020)** retrieval-*augmented pretraining* (learn to retrieve end-to-end) ⇒ **RAG (Lewis 2020)** the name and the pattern: retrieve passages, condition generation on them ⇒ **FiD / Fusion-in-Decoder (2021)** encode many passages, fuse in the decoder for multi-doc answers ⇒ **ColBERT (2020) / v2 (2021)** **late interaction** (per-token vectors, max-sim) — between bi- and cross-encoder ⇒ **HyDE (2022)** embed a *hypothetical answer* instead of the question (closer to real answer passages) (the query-transformation material in the RAG chapter) ⇒ **Self-RAG (2023) / CRAG (2024)** the model *decides when to retrieve*, *critiques* passages, and *self-corrects* (the advanced-RAG material) ⇒ **RAPTOR (2024)** recursively cluster+summarize chunks into a tree for multi-level retrieval ⇒ **GraphRAG (Microsoft 2024)** extract an entity/relation **knowledge graph** + community summaries to answer *corpus-wide/global* questions vanilla chunk-retrieval can't (the advanced-RAG material) ⇒ **HippoRAG / HippoRAG 2 (2024–25)** graph + personalized PageRank for cheap multi-hop "memory" ⇒ **Agentic RAG (Search-o1, Search-R1, Graph-R1, 2025)** make retrieval a *reasoned, iterative tool decision*, often **RL-trained** end-to-end (the agentic-RAG material) — converging with the agents track and productized as **Deep Research**.

---

## Part 6 — Agents Lineage

From "answer a question" to "drive a loop and act" (the agents chapter).

**WebGPT (2021)** an LLM that browses to answer ⇒ **ReAct (Yao Oct 2022)** interleave **Thought → Act → Observation**, grounding reasoning in real tool results — the substrate of nearly every agent (the ReAct material in the agents chapter) ⇒ **Toolformer (2023)** the model *self-teaches* which APIs to call and when ⇒ **Reflexion (Mar 2023)** verbal self-critique of a failed attempt, retry with the lesson in context ("verbal RL," no weight updates) (the planning material) ⇒ **OpenAI function calling (Jun 2023)** standardized **structured tool calls** (schema → JSON call → observation) (the tool-use material) ⇒ **AutoGPT / BabyAGI (2023)** autonomous goal-pursuit hype — exposed how brittle long-horizon loops are ⇒ **Generative Agents (Stanford 2023)** memory + **reflection** (consolidate experiences into higher-level insights) (the memory material); **Voyager (2023)** a growing **skill library** (lifelong learning) ⇒ **MetaGPT / ChatDev / AutoGen (2023)** **multi-agent** role teams (planner/coder/critic) (the multi-agent material) ⇒ **Tree-of-Thoughts / LATS (2023)** search-based planning ⇒ **Computer use (Anthropic Oct 2024)** agents that operate a GUI like a human ⇒ **MCP (Anthropic Nov 2024)** an open standard to expose any tool/data uniformly — the "USB-C for tools" (the tool-use material) ⇒ **Coding agents (Devin 2024, OpenAI Codex agent May 2025, Claude Code 2025)** long-horizon software engineering, evaluated on **SWE-bench Verified** ⇒ **RL-trained agents + context engineering (2025–26)** turn-level credit assignment for multi-turn tasks, self-evolving memory (ReasoningBank, Mem0, A-MEM), and **context engineering** (compression/offloading/isolation) as the dominant practical lever (the context-engineering material). The binding constraint throughout, still unsolved: **long-horizon reliability** — per-step success compounds badly over many steps (the reliability material).

---

## Part 7 — The meta-pattern: placing any 2026+ paper on these chains

When a new paper lands, you no longer ask "what is this" — you ask "which chain, and what's the next link":

1. **Which track?** Architecture/training spine (§1), efficiency (§2), reasoning (§3), multimodal (§4), RAG (§5), or agents (§6)? Usually one, occasionally a merge (agentic RAG = §5×§6).
2. **What's the predecessor it's beating, and on what axis?** Every entry above improved a *specific* prior link on a *specific* axis (quality / `n²` cost / KV memory / reasoning / context / reliability / data). Name both.
3. **Which recurring move is it?** Almost everything is one of: *compress something* (KV→MLA, weights→quant, vision→Q-Former), *make a quadratic thing linear* (linear attn/SSM/hybrids), *decouple capacity from compute* (MoE), *change the training signal* (SFT→DPO→RLVR→self-verify), *spend compute at inference* (CoT→search→o1), *add a control loop* (RAG, ReAct, agents), or *unify modalities/interfaces* (T5 text-to-text, Chameleon tokens-for-everything).
4. **What did it trade away?** Find the cost even when hidden — it's the predecessor's strength (e.g. linear attention trades exact retrieval; MoE trades VRAM; reasoning trades latency/cost; agents trade reliability).
5. **Will it survive at a different scale / hardware / base model?** The senior judgment. Most links don't generalize; knowing which do is the skill this whole timeline is meant to build.

If you can run those five on a fresh paper in a couple of minutes, you have what the original request asked for: the ability to read almost anything in this field and understand it as the next move in a game you already know.

---

## Concepts now covered here (that the earlier chapters assumed)

For completeness, the prehistory and tracks above fold in the foundational pieces the rest of the guide built on top of without deriving: **distributed word embeddings (Word2Vec/GloVe), RNNs/LSTMs, seq2seq, the original (Bahdanau) attention, ELMo** (Part 0); the **MoE history** (Shazeer→GShard→Switch, §2); **scaling-law and emergence framing** (entries 7, 11, 12); **instruction tuning and Constitutional AI/RLAIF** (entries 9, 14); the **CNN→ResNet→ViT** vision backbone path and **diffusion/image-generation** thread (§4); and the **process-reward / STaR reasoning lineage** and **tool-use/MCP/computer-use** agent lineage (§3, §6). Combined with the rest of the guide, there should be no load-bearing concept in a modern paper that isn't introduced somewhere in this guide.
