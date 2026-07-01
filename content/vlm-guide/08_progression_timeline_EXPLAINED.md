# The Progression, Explained for a Junior ML Engineer

This is a companion to the progression timeline. The timeline is dense and assumes you already half-know the jargon. This chapter slows down and explains **every term, every mechanism, and every "why"** so that you can read the timeline and actually understand what each move *does* mechanically — not just recognize the names.

How to use this: read a section here, then read the matching section in the timeline. The timeline is the map; this is the field guide that tells you what the symbols mean.

A note on the mental model the whole guide is built on: **the field is a chain of fixes.** Nobody invented the modern LLM in one shot. Each idea solved a concrete pain in the previous idea, and in solving it created a *new* pain that the next idea attacked. If you learn the chain of pains, the papers stop being a memorization list and become a story you can predict.

---

## Table of contents

- [Vocabulary you need first](#vocabulary-you-need-first)
- [Part 0 — Prehistory: the world before Transformers](#part-0--prehistory-the-world-before-transformers)
- [Part 1 — The Main Spine](#part-1--the-main-spine)
- [Part 2 — Efficiency & Inference](#part-2--efficiency--inference)
- [Part 3 — Reasoning & Test-Time Compute](#part-3--reasoning--test-time-compute)
- [Part 4 — Vision & Multimodal](#part-4--vision--multimodal)
- [Part 5 — RAG & Retrieval](#part-5--rag--retrieval)
- [Part 6 — Agents](#part-6--agents)
- [Part 7 — The meta-pattern](#part-7--the-meta-pattern-how-to-read-any-new-paper)
- [Cheat sheet](#cheat-sheet)

---

## Vocabulary you need first

Before the timeline makes sense, you need these words. Skim now, refer back later.

- **Token.** A chunk of text the model actually sees — usually a sub-word (e.g. "playing" → `play` + `ing`). Models predict tokens, not characters or whole words. A tokenizer maps text ↔ token IDs.
- **Embedding.** A token ID turned into a vector of numbers (e.g. 4096 floats). This vector is the model's internal "meaning" of that token. Similar meanings → nearby vectors.
- **Parameter (weight).** A single learnable number inside the model. "7B params" = 7 billion of these. Training = adjusting them to lower the loss.
- **Loss.** A number measuring how wrong the model is. Training minimizes it. For language models the standard loss is **next-token prediction** (cross-entropy): "given these tokens, how surprised were you by the true next token?"
- **Pretraining.** The huge, expensive first phase: read trillions of tokens of raw web text, just predicting the next token. Produces a model that "knows" language and facts but isn't yet helpful or safe.
- **Fine-tuning.** A cheaper second phase: take the pretrained model and nudge it on a smaller, targeted dataset (e.g. instruction-following examples).
- **Inference.** Actually *using* the trained model to generate output. Distinct from training. "Inference cost" = what you pay every time a user sends a request.
- **FFN (feed-forward network).** The simple "thinking" sub-layer inside a Transformer block — two linear layers with a nonlinearity between them. Each token is processed independently here.
- **Attention.** The mechanism that lets one token *look at* other tokens and pull in their information. The core of the Transformer. Explained in detail below.
- **Logits / softmax.** Logits = raw scores the model outputs for each possible next token. Softmax squashes them into probabilities that sum to 1.
- **Autoregressive.** Generate one token, append it, feed the whole thing back in, generate the next. Left-to-right, one at a time. This is how GPT-style models produce text.

Keep these in your head. Almost every entry in the timeline is "we made *one* of these things cheaper, smarter, or different."

---

## Part 0 — Prehistory: the world before Transformers

You only need this to feel *why 2017 was a revolution*. The enemy this whole era was fighting is **recurrence**.

### Word2Vec / GloVe (2013–14) — words become vectors

**The idea.** Represent each word as a dense vector (say 300 numbers) learned from how words co-occur in text. Words used in similar contexts end up with similar vectors. Famous party trick: `vector("king") - vector("man") + vector("woman") ≈ vector("queen")` — analogies as arithmetic.

**Why it mattered.** This is the ancestor of *all* embeddings. The insight that "meaning can live in a vector space" underpins everything after.

**The limitation that motivated the next thing.** One fixed vector per word, forever. The word "bank" gets *one* vector whether you mean a riverbank or a money bank. The model is **context-blind**. ⇒ We need word representations that change depending on the sentence.

### RNNs / LSTMs (1997, popular ~2014) — reading a sequence one step at a time

**The idea.** A Recurrent Neural Network reads a sequence left to right. At each step it combines the current word with a **hidden state** (a memory vector) carried forward from the previous step. LSTMs ("Long Short-Term Memory") added **gates** — little learned valves that decide what to keep, forget, and output — which let them remember things over longer stretches without the gradient vanishing to zero during training.

**Why it mattered.** First models that genuinely handled variable-length sequences and order.

**The two killer limitations:**
1. **Strictly sequential.** Step 5 cannot be computed until step 4 is done. You *cannot* parallelize across the sequence. GPUs love parallelism; RNNs starve them.
2. **Forgetting / bottleneck.** Everything the model knows about the past is crammed into one fixed-size hidden vector. Long-range information gets diluted and lost.

⇒ Both of these are exactly what the Transformer kills.

### Seq2Seq (2014) — encoder squashes, decoder unpacks

**The idea.** Two RNNs. The **encoder** reads the whole input sentence and compresses it into a single fixed "thought vector." The **decoder** reads that vector and generates the output (e.g. the translation). Enabled machine translation and summarization in one neural framework.

**The limitation.** Cramming a 50-word sentence into one fixed vector is a **brutal bottleneck**. Long inputs get squashed beyond recognition. ⇒ What if the decoder could look back at the whole input instead of one summary vector?

### Attention, the original (Bahdanau 2014; Luong 2015) — let the decoder look back

**The idea — this is the seed of everything.** Instead of one fixed thought vector, let the decoder, *at each output step*, look at **all** the encoder's hidden states and compute a weighted blend of them. The weights ("how much should I focus on input word *j* while producing output word *i*") are **learned** and computed on the fly. This is called **soft alignment**.

This is *literally* the attention mechanism used in modern Transformers — invented here to fix the seq2seq bottleneck.

**The catch.** It was bolted *onto* RNNs. So it still inherited the sequential, slow nature of recurrence. Attention worked; recurrence was dead weight dragging it down.

### ELMo (2018) — context-aware word vectors

**The idea.** Run a bidirectional LSTM language model and use its internal states as word vectors. Now "bank" gets a *different* vector in "river bank" vs "savings bank." Fixed Word2Vec's context-blindness.

**The catch.** Still RNN-based, still slow.

### The setup by mid-2017

Attention clearly worked. Recurrence was clearly the bottleneck. So the obvious-in-hindsight question:

> **What if attention is all you need — and we throw the RNN away entirely?**

That question is the title of the 2017 paper. The rest of the field flows from the answer.

---

## Part 1 — The Main Spine

Architecture + pretraining + alignment. This is the backbone. If you only memorize one chain, memorize this.

### 2017 — Transformer (Vaswani et al., Google): "Attention Is All You Need"

**What it did.** Removed recurrence and convolution *entirely*. A Transformer block is just:
- **Self-attention** — every token looks at every other token and pulls in relevant info.
- **FFN** — each token gets processed independently through a small network.
- Wrapped with **residual connections** (add the input back to the output, so gradients flow and you can stack deep) and **layer normalization** (keeps activations numerically stable).

**Self-attention, mechanically (the one piece worth understanding deeply).** For each token, the model computes three vectors via learned weight matrices:
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer?"
- **Value (V)** — "what information do I carry?"

To update token *i*: take its Query, compare (dot-product) against *every* token's Key → a score for each. Softmax the scores into weights. Use those weights to blend everyone's Values. That blend becomes token *i*'s new representation.

**Multi-Head Attention** = do this several times in parallel with different learned Q/K/V projections ("heads"), so different heads can specialize (one tracks syntax, another tracks subject-verb agreement, etc.), then concatenate.

**Why this was the break:**
1. **Parallel training.** There's no "wait for the previous step" — every position's attention is computed at once. GPUs go brrr. This is *the* reason Transformers could scale to billions of params.
2. **No long-distance forgetting.** Any two tokens are connected in a *single* step, no matter how far apart. The bottleneck is gone.

**The cost it introduced (remember this — it drives all of Part 2).** Comparing every token to every token is `O(n²)` in sequence length `n`. Double the context, quadruple the compute and memory. This quadratic cost becomes the central enemy of the efficiency track.

**Original form.** Encoder–decoder, built for translation, with **sinusoidal position encodings** (since attention itself is order-blind, you must *add* position information) and Multi-Head Attention.

⇒ If this scales so well, what happens if we **pretrain** it on huge text and adapt to tasks?

### 2018 — the pretraining paradigm splits in two

**GPT-1 (OpenAI) — generative pretraining + fine-tuning.** Take a **decoder-only** Transformer (only the generation half). Pretrain it as a plain next-token predictor on unlabeled text. Then fine-tune on each downstream task. Showed that unsupervised pretraining *transfers* — the model learns general language skill, then specializes cheaply.
- *Decoder-only* means each token can only attend to tokens *before* it (causal masking) — necessary because at generation time you don't know the future.
- **Limit:** unidirectional (left-to-right only), and still needs separate fine-tuning per task. ⇒ For *understanding* tasks (classification, QA), wouldn't seeing the whole sentence both directions help?

**BERT (Google) — bidirectional masked-language-model pretraining.** Use an **encoder-only** Transformer. Pretrain by **masking** ~15% of tokens at random and asking the model to predict them. Because it's not generating left-to-right, every token can see *full left and right context*. Also trained "next-sentence prediction." Crushed understanding benchmarks (sentiment, QA, entailment).
- **Limit:** the masking objective doesn't teach fluent generation. BERT is brilliant at *understanding* a sentence, useless at *writing* one.

**The fork.** BERT-style (encoder, understanding) vs GPT-style (decoder, generation). **Generation won the scaling race** — because a model that can generate can also understand (you can phrase any task as "generate the answer"), but not vice versa. The whole future is decoder-only.

*(Side notes from 2018–19: ULMFiT formalized transfer learning for NLP; RoBERTa showed BERT was undertrained and dropped the next-sentence task.)*

### 2019 — scale and unification

**GPT-2 (OpenAI) — scale + zero-shot.** Same decoder-only recipe, 10× bigger (1.5B params), more data. The discovery: at scale, the model can do tasks **zero-shot** — just describe the task in the prompt, no examples, no fine-tuning. ("Language models are unsupervised multitask learners.")
- **Limit:** zero-shot was flaky and inconsistent. Clearly more scale needed. ⇒ Push harder.

**T5 (Google) — "everything is text-to-text."** Reframe *every* NLP task as "text in → text out." Translation, classification, summarization, QA — all the same interface, just with a task prefix ("translate English to German: ..."). Used a unified **encoder–decoder** Transformer pretrained on the cleaned C4 web corpus with a "span corruption" objective (mask out spans, predict them). Ran the most systematic ablation study of the era.
- This encoder–decoder lineage lives on in BART and many *early* vision-language models. ⇒ Meanwhile: how far does *pure decoder-only scale* go?

### 2020 — in-context learning and the first scaling map

**GPT-3 (OpenAI) — in-context / few-shot learning.** 175B params. The breakthrough wasn't only the size — it was that you could put a few examples *in the prompt* and the model would do the task **with no weight updates at all**:

```text
Translate English to French:
sea otter => loutre de mer
cheese => fromage
plush giraffe => ___        <- model completes this
```

This is **in-context learning**. "Prompting" became a discipline. This is the moment LLMs became general-purpose tools instead of task-specific models.
- **Limit:** GPT-3 followed *patterns*, not *intentions*. Ask it a question and it might just continue with more questions, because that's a plausible text continuation. It wasn't an assistant yet. Also: was 175B even the right size?
- ⇒ Two open questions: (a) how do we make it *follow instructions and help*, (b) what's the *compute-optimal* size?

**Scaling Laws (Kaplan et al., OpenAI) — loss is a predictable power law.** Showed that test loss falls smoothly and predictably as you increase model size, data, and compute — following a clean power law. Meaning: you can run a few small experiments and **forecast** the loss of a giant run before spending millions on it. This *justified* the huge scaling bets.
- **Limit:** their recommended size-vs-data tradeoff was wrong — it pushed toward models that were *too big* and trained on *too little data*. Corrected in 2022 by Chinchilla.

### 2021 — efficiency knobs and the instruction-following seed

**RoPE / RoFormer — rotary position embeddings.** A better way to tell the model about token positions. Instead of *adding* a position vector, RoPE **rotates** the Query and Key vectors by an angle proportional to their position. The math works out so that the attention score between two tokens depends only on their *relative* distance — which is what actually matters for language. No learned position parameters, and it extrapolates to longer sequences better. Became near-universal (Llama, most modern models).

**FLAN / T0 — instruction tuning.** Fine-tune the model on many tasks, each *phrased as a natural-language instruction* ("Summarize this article:", "Is this review positive?"). Result: the model then follows *unseen* instructions zero-shot far better. This is the seed of "instruction following" — the direct precursor to **SFT** (supervised fine-tuning) in the modern alignment stack.
- **Limit:** instruction tuning teaches the model the *format* of helpful answers, but not nuanced *human preference* (which of two good answers is better, what's harmful, what tone). ⇒ Add a preference signal.

*(Also 2021, efficiency track: Switch Transformer simplified Mixture-of-Experts to top-1 routing at trillion-param scale; LoRA introduced cheap fine-tuning adapters. Both covered in Part 2.)*

### 2022 — the corrective, the reasoning spark, and alignment

**Chinchilla (DeepMind) — compute-optimal scaling (~20 tokens/param).** Re-ran the scaling analysis carefully and found nearly all prior big models (GPT-3, Gopher) were **too big and trained on too little data**. The rule of thumb: for a fixed training budget, you want roughly **20 training tokens per parameter**. A *smaller* model trained on *more* data beats a *bigger* undertrained one at equal compute. This reframed the entire field toward "train smaller models longer."
- **Limit:** Chinchilla optimizes *training* compute only. But you serve a deployed model *millions* of times — its **inference** cost matters more than its training cost. So later models deliberately **overtrain** small models (way past 20 tokens/param) to get a small, cheap-to-run model that's still very capable. (Llama did exactly this.)

**Chain-of-Thought (Wei et al.) + Self-Consistency + Emergent Abilities.** Discovery: if you prompt the model to **"think step by step"** and show its reasoning before the final answer, accuracy on math/logic problems jumps massively. And this ability **emerges sharply only at scale** — small models don't benefit, big ones do. **Self-Consistency** improves it further: sample many independent reasoning chains, take the majority-vote answer.
- This is the conceptual seed of the *entire* reasoning-model era. ⇒ Can we *train* the model to reason natively, instead of having to prompt it every time?

**InstructGPT (OpenAI) — RLHF (Reinforcement Learning from Human Feedback).** *The* recipe that turned language models into **assistants**. Three stages:
1. **SFT** — fine-tune on human-written demonstrations of good answers.
2. **Reward model** — show humans pairs of model answers, have them pick the better one; train a separate model to predict these human preferences (output a scalar "goodness" score).
3. **RL (PPO)** — use reinforcement learning to update the language model to maximize the reward model's score, with a **KL penalty** ("KL leash") that stops it drifting too far from the SFT model and producing gibberish that games the reward.

The stunning result: a **1.3B** InstructGPT was *preferred by humans* over the **175B** GPT-3. **Alignment, not raw size, drove perceived quality.**
- **Limit:** RLHF is a complex, unstable pipeline juggling four models (policy, reference, reward, value). Hard to get right. ⇒ Can we simplify it?

**Constitutional AI / RLAIF (Anthropic).** Replace much of the expensive *human* feedback with **AI feedback** guided by a written set of principles (a "constitution"). The model critiques and revises its own answers against the principles. Makes alignment more scalable (less human labor) and more steerable (you edit the principles).

**ChatGPT (Nov 30, 2022).** Not a paper — InstructGPT-style alignment wrapped in a chat interface. But *the* inflection point: it made LLMs mainstream overnight and kicked off the arms race the rest of the timeline races through.

### 2023 — the open-model wave and the alignment shortcut

**LLaMA (Meta) — the efficient open recipe.** A clean decoder-only stack that became the **template every open model copied**:
- **RMSNorm** — a simpler, cheaper normalization than LayerNorm.
- **SwiGLU** — a better FFN activation (a gated variant) that improves quality.
- **RoPE** — the rotary positions from 2021.
- **No bias terms** — drop them, they don't help and cost params.
- Trained on *far more tokens than Chinchilla-optimal* — the inference-aware overtraining bet.

LLaMA-13B matched GPT-3-175B. When the weights leaked (and then Llama 2 shipped with an open license), it ignited the entire open-source ecosystem.

**GPT-4 (OpenAI) — the multimodal frontier.** Large (size undisclosed, widely believed to be a Mixture-of-Experts), accepts **images** as input, far stronger reasoning. Set the closed-frontier bar for ~2 years — the target everyone else chased.

**Llama 2 (Meta).** Open weights with a *permissive commercial license*, a documented RLHF pipeline, and **GQA** (grouped-query attention — see Part 2) on the larger sizes. Made *aligned* open models actually usable in products.

**DPO (Direct Preference Optimization) — preference alignment without RL.** Proved you can skip the whole unstable RLHF machine. The math: you can algebraically reparameterize the RLHF objective into a **simple classification loss** directly on (prompt, chosen-answer, rejected-answer) triples. No separate reward model, no PPO, no sampling loop. Just gradient descent on a clean loss. Stable, simple — became the default preference-tuning method.
- **Limit:** preferences are *subjective*. For math and code, there's an *objectively correct answer* you could just check, instead of asking humans which response "looks" better. ⇒ Use verifiable rewards (the R1 line).

**Mistral 7B / Mixtral 8×7B.** Mistral 7B: a small model punching far above its weight (used sliding-window attention + GQA for efficiency). Mixtral 8×7B: the first widely-used **open sparse Mixture-of-Experts** — 8 expert FFNs, a router picks the top-2 per token, ~47B total parameters but only ~13B *active* per token. You get big-model quality at small-model inference cost.

**Mamba (selective state-space models).** A serious **sub-quadratic** alternative to attention — meaning it scales *linearly* with sequence length, not quadratically, and needs **no KV cache**. It processes sequences with a state-space model whose dynamics are *input-dependent* ("selective"), letting it route information by content the way attention does.
- **Limit:** weaker than attention at *exact long-range retrieval* (e.g. "what was the 3rd word 10,000 tokens ago" — attention nails this, SSMs blur it). ⇒ Don't *replace* attention — **hybridize** (this happens in 2025).

### 2024 — efficiency frontier, long context, the reasoning turn

**Llama 3 / 3.1 (Meta).** Pushed the open recipe hard: Llama-3-8B trained on **15 trillion tokens** (~1800 tokens/param — *far* past Chinchilla's 20, the inference-aware overtraining bet taken to the extreme), then a **405B** model matching the closed frontier. GQA across the whole line, 128k-token context. Lesson: architecture is converging; the next gains come from *efficiency* and *post-training*.

**DeepSeek-V2 → V3 — the efficient-flagship template.** This is *the* reference design for modern efficient large models. Two big ideas plus a pile of systems tricks:
- **MLA (Multi-head Latent Attention)** — compress the KV cache down to a small low-rank *latent* vector instead of storing full keys and values for every token. Gets near-MHA quality at a fraction of the memory (with a clever "decoupled RoPE" trick to keep position info working). See Part 2.
- **DeepSeekMoE** — fine-grained experts (many small experts) plus a few *shared* experts that always run.
- V3 (671B total / **37B active**) added: **auxiliary-loss-free load balancing** (keep experts evenly used without a hacky extra loss term), **FP8 training** (8-bit floating point — half the memory of bf16), and **multi-token prediction** (predict several future tokens at once for a training signal boost). Frontier quality at a fraction of training and inference cost — and fully open.

⇒ Now apply post-training to make it *reason*.

**o1 (OpenAI) — inference-time compute as a first-class axis.** A model trained (via RL) to produce a long *internal* chain-of-thought before answering, and crucially to **spend more thinking time on harder problems**. This established a **second scaling axis**, orthogonal to model size: instead of (or in addition to) a bigger model, **scale the amount of thinking at inference time.** Reasoning models became their own category.
- **Limit:** OpenAI kept the method secret. ⇒ DeepSeek reveals and open-sources the recipe.

### 2025 — the reasoning explosion, hybrids go mainstream, multipolar frontier

**DeepSeek-R1 / R1-Zero — reasoning from pure RL with verifiable rewards.** The most-copied method of 2025. The pieces:
- **GRPO (Group Relative Policy Optimization)** — a simpler RL algorithm than PPO. It drops the separate "value/critic" model; instead it samples a *group* of answers for each prompt and judges each one *relative to the group average*. Cheaper and more stable.
- **RLVR (RL from Verifiable Rewards)** — instead of a learned reward model guessing what humans like, use **rule-based, objectively verifiable rewards**: did the math answer match? did the code pass the tests? Reward = correct/incorrect. No reward model needed.
- **The shock:** **R1-Zero, with no SFT at all** (pure RL on the base model), *spontaneously* learned long chain-of-thought, self-verification, backtracking, and "aha moments" — reasoning behavior *emerged* from the reward signal alone. R1 then added a small "cold-start" SFT for readability. Finally they **distilled** the reasoning traces into small Qwen/Llama models, transferring the skill cheaply.

This reframed all post-training around verifiable rewards. ⇒ Everyone ships a reasoning tier (Kimi 1.5, QwQ, o3, Gemini Deep Think, Claude extended thinking, Phi-4-reasoning).

**Qwen3 (Alibaba) — unified hybrid-thinking, fully open.** One model line that switches between **thinking** and **non-thinking** modes (spend reasoning tokens when needed, answer instantly when not). Trained on ~36T tokens, shipped as both **dense** and **MoE** (e.g. 235B total / 22B active). Competitive with R1/o1 — and self-hostable across sizes.

**Llama 4 (Meta) — natively multimodal MoE at huge context.** MoE (variants named Scout/Maverick), **early-fusion** multimodality (vision and text fused from the start, not bolted on — see Part 4), and a *very* large context window (multi-million tokens). Pushed native-multimodal + MoE + long-context into the open mainstream.

**The hybrid-attention turn (Qwen3-Next → Qwen3.5 / Kimi Linear).** The Mamba/linear-attention thread finally *combines* with full attention instead of competing. Recipe: interleave mostly-cheap linear/SSM blocks with *periodic* full-attention blocks (commonly **3:1** — three cheap blocks per one full block). You get linear-ish cost *and* attention's exact retrieval.
- Qwen3-Next (Gated DeltaNet + gated attention) proved it near-flagship; **Qwen3.5** promoted the hybrid into the *main flagship* line (the signal that the strategy won); **Kimi Linear** refined it further. NVIDIA's **Nemotron 3** shipped a hybrid Mamba-Transformer MoE. This is the *live* architectural frontier.

**Kimi K2 (Moonshot) + K2 Thinking.** A very large **open** MoE (~1T total params) built on the V3-style template (MLA, more experts), with a strong "thinking" variant and a focus on **agentic / tool use**. Showed open models matching the closed frontier on coding and agent tasks.

**The closed frontier in 2025:** GPT-5 (and 5.1 thinking/instant modes), **Gemini 3.0** (top of benchmarks), **Claude 4 → 4.5** (Opus/Sonnet/Haiku; aimed at coding and long-running agents). All converging on the same themes: **reasoning by default, native multimodality, agentic capability** (computer use, long-horizon coding).

### 2026 — multipolar, efficient, agentic

The field is now **multipolar**: Chinese open-weight labs (DeepSeek, Qwen, Kimi, GLM, MiniMax) sit within ~1–2 benchmark points of the closed frontier on coding. It's also **hardware-diverse**: DeepSeek **V4** (~trillion params) was trained on **non-Nvidia Huawei Ascend** chips, breaking the GPU monopoly.

The architectural consolidation is clear — the standard recipe is:

> **MoE + hybrid-linear-attention + reasoning-by-default + native-multimodal.**

Plus: **self-verifiable reasoning** (DeepSeekMath-V2 generates *and checks* its own proofs, extending RLVR to domains that lack an easy automatic verifier), and **agentic / long-horizon coding** as the dominant application target (specialized agent models like the GPT-5.x Codex-Max line). Point-releases ship every few days now — **the trends matter, the version numbers don't.**

### The spine in one breath

remove recurrence (2017) → pretrain at scale (GPT/BERT, 2018–20) → discover in-context learning and scaling laws (2020) → correct the scaling recipe (Chinchilla 2022) → align into assistants (RLHF 2022, simplified by DPO 2023) → open the recipe (LLaMA 2023) → make it efficient (MoE, MLA, FP8: DeepSeek 2024) → make it reason via verifiable-reward RL (o1 2024, R1 2025) → make attention sub-quadratic via hybrids and turn it agentic (2025–26).

---

## Part 2 — Efficiency & Inference

The entire track is one long war on two enemies introduced by the Transformer:
1. **Attention's `O(n²)` cost** — quadratic in sequence length.
2. **The KV-cache memory wall** — explained below.

**First, what is the KV cache?** When generating text autoregressively, at each new token you'd recompute attention over all previous tokens. That's wasteful — the Keys and Values of past tokens don't change. So you **cache** them. The KV cache stores K and V for every token in the context. The problem: it grows linearly with context length *and* batch size, and it lives in precious GPU memory. For long contexts it becomes the dominant memory cost and the main thing limiting how many users you can serve at once. Much of this track is "shrink the KV cache."

### KV-cache / attention shape

- **MHA (Multi-Head Attention, 2017)** — the original. Full quality, but every head stores its own K and V → huge cache.
- **MQA (Multi-Query Attention, 2019)** — all heads *share a single* K and V head. Cache shrinks dramatically. *Cost:* a quality drop.
- **GQA (Grouped-Query Attention, 2023)** — the compromise that won. Group the heads; each *group* shares one K/V. A tunable knob between MHA (best quality) and MQA (smallest cache). Used by Llama 2/3 and most models.
- **MLA (Multi-head Latent Attention, DeepSeek-V2 2024)** — compress K and V into a small **low-rank latent** vector, decompress on the fly. MHA-level quality at a fraction of the cache, using a "decoupled RoPE" trick to keep positional info intact.

### Sparse capacity (Mixture-of-Experts)

The core MoE idea: **decouple the number of parameters from the compute per token.** Instead of one big FFN that every token passes through, have *many* expert FFNs and a **router** that sends each token to only a few. The model "has" huge knowledge (many params) but each token only "uses" a small slice (cheap compute).

- **Outrageously Large Neural Networks (Shazeer 2017)** — MoE on LSTMs; introduced the decouple-params-from-compute idea.
- **GShard (2020)** — MoE for Transformers, plus the sharding to train it across many devices.
- **Switch Transformer (2021)** — simplified routing to **top-1** (send each token to just *one* expert), reached trillion params, easier to train.
- **Mixtral (2023)** — the first big *open* MoE; made the technique mainstream.
- **DeepSeekMoE / V3 (2024)** — fine-grained experts + shared experts + **auxiliary-loss-free** load balancing (keep experts evenly used without a hacky penalty term). The modern standard.

### Sub-quadratic sequence mixing

Goal: replace `O(n²)` attention with something cheaper.
- **Linear/efficient attention & RWKV (2020–22)** — mathematically reorganize attention so it's `O(n)`. Often a quality gap vs full attention.
- **Mamba (2023)** — selective state-space model; strong, but weak at *exact* retrieval.
- **Hybrids (Qwen3-Next/Qwen3.5, Kimi Linear, Nemotron 3, MiniMax-M1 "lightning attention", 2025–26)** — interleave cheap linear/SSM blocks with periodic full-attention blocks. Get both cheap cost and exact retrieval. The current frontier.

### Exact-attention IO efficiency

This thread keeps attention *mathematically exact* but makes it run faster on real hardware by being smart about memory movement.
- The problem: naive attention **materializes** the full `n×n` score matrix in slow GPU memory (HBM). It's *memory-bandwidth-bound* — the GPU spends its time shuffling data, not computing.
- **FlashAttention (Dao 2022)** — never materialize the full matrix. Process it in **tiles** that fit in fast on-chip SRAM, using an **online softmax** (compute softmax incrementally) so you get the *exact* same answer with `O(n)` memory and far less data movement. Huge real-world speedup, zero accuracy loss.
- **FlashAttention-2 (2023)** — better parallelism across GPU warps.
- **FlashAttention-3 (2024)** — exploits Hopper-GPU features (warp specialization + FP8).
- **FlashDecoding / FlashInfer** — variants tuned for the decode phase and sparsity.

### Serving (systems layer)

How you run inference for many users efficiently.
- The problem: storing each request's KV cache in one contiguous memory block causes huge **fragmentation** (wasted gaps).
- **PagedAttention / vLLM (Kwon 2023)** — store the KV cache in fixed-size **pages**, like an operating system's virtual memory. Pages can be shared and allocated on demand. This enables:
  - **Continuous batching** — swap requests in and out of a batch as they finish, keeping the GPU full.
  - **Prefix caching** — if many requests share a prompt prefix (e.g. a system prompt), compute it once and share.
- Plus: **FP8 KV quantization**, **chunked prefill**, **prefill/decode disaggregation** (run the two phases on different hardware tuned for each).

### Compression (fit big models on small hardware)

- **LoRA (2021)** — Low-Rank Adaptation. Freeze the giant pretrained weights; train only tiny *low-rank adapter* matrices added alongside them. You train <1% of the params and get most of the benefit of full fine-tuning, cheaply.
- **QLoRA (2023)** — do LoRA on top of a base model **quantized to 4-bit**. Lets you fine-tune very large models on a *single* GPU.
- **GPTQ (2022) / AWQ (2023)** — **post-training quantization** (PTQ): squeeze trained weights down to 4-bit *after* training. AWQ is "activation-aware" (protects the weights that matter most).
- **SmoothQuant** (8-bit weights *and* activations via outlier migration), **GGUF** (the llama.cpp format for running models on consumer CPUs/laptops), **QAT** (quantization-*aware* training) + layer-sensitivity maps for on-device int4.

*(Quantization = representing weights with fewer bits — fp16 → int8 → int4 — to save memory and speed up inference, trading a little accuracy.)*

### Decode acceleration

- The problem: generating one token at a time *underuses* the GPU — it's built for parallel work but decode is sequential.
- **Speculative decoding (Leviathan 2023)** — a small fast "draft" model guesses the next several tokens; the big model **verifies them all in parallel** in one pass. Accepted guesses are free; rejected ones fall back. Output is **provably identical** to the big model alone — just faster.
- **Medusa (2024)** — add extra prediction "heads" to the model itself so it drafts its own future tokens; no separate draft model.
- **EAGLE / EAGLE-3 (2024–25)** — draft at the *feature* level (more accurate guesses → more accepted). Current state of the art.

---

## Part 3 — Reasoning & Test-Time Compute

How "think before answering" went from a prompt trick → a trained capability → a whole scaling axis.

The chain:
- **Scratchpads (2021)** — let the model write intermediate working in tokens before the answer.
- **Chain-of-Thought prompting (Wei 2022)** — "think step by step" unlocks latent reasoning; emerges at scale.
- **Self-Consistency (2022)** — sample many reasoning chains, take the majority answer. Trades compute for accuracy.
- **STaR (Zelikman 2022)** — *bootstrap* reasoning: generate chains, **keep only the ones that reached the correct answer**, fine-tune on those, repeat. The model learns to reason from its *own* correct traces.
- **Let's Verify Step by Step / PRMs (OpenAI 2023)** — reward *each step* of the reasoning (a **Process Reward Model**), not just the final answer. Catches reasoning that gets the right answer for the wrong reasons. (Process supervision vs outcome supervision.)
- **Tree-of-Thoughts / LATS (2023)** — don't commit to one chain; *search* over a tree of reasoning branches with backtracking.
- **Quiet-STaR (2024)** — learn to generate a silent "thought" between *every* token.
- **o1 (OpenAI 2024)** — RL-trained long CoT + **test-time compute scaling** as a first-class method (but secret).
- **DeepSeek-R1 / GRPO / RLVR (2025)** — open the recipe: **verifiable rewards** + **critic-free group-relative RL**; reasoning *emerges* from pure RL (R1-Zero), then distill into small models.
- **Dr. GRPO / DAPO (2025)** — fix subtle biases in GRPO's normalization (it was unintentionally favoring certain lengths/difficulties); process reward models and turn-level credit assignment mature.
- **DeepSeekMath-V2 (late 2025)** — *self-verifiable* reasoning: the model generates **and checks** its own proofs, extending verifiable-reward RL to domains that have no cheap external verifier.

**The open debate to be aware of:** does RLVR *create* genuinely new reasoning ability, or merely *elicit* what pretraining already buried in the weights? And: when should the model *stop* thinking? (Overthinking measurably hurts on easy problems.)

---

## Part 4 — Vision & Multimodal

How pixels learned to talk to language models. Two feeder threads (vision backbones, image generation) converge into VLMs (Vision-Language Models).

### Vision backbones (how to turn an image into features)

- **AlexNet (2012)** — CNNs (convolutional neural nets) decisively win image classification; the deep-learning vision era begins.
- **ResNet (2015)** — **residual connections** let you train *very* deep networks (the gradient has a shortcut path). Side note worth remembering: residuals are exactly what the Transformer later borrows.
- **ViT (Vision Transformer, 2020)** — drop convolutions entirely: cut the image into fixed patches (e.g. 16×16 pixels), treat each patch as a "token," and run a plain **Transformer**. This *unifies* vision with NLP — same architecture for both.

### Language-aligned encoders (put image features in the same space as words)

- **CLIP (2021) + ALIGN (2021)** — train an image encoder and a text encoder **contrastively** on web-scale image-caption pairs: pull matching image/text vectors together, push mismatched ones apart. Result: image features that *already live in language space* → zero-shot classification (classify an image by comparing it to text label embeddings).
- **SigLIP (2023) / SigLIP 2 (2025)** — swap CLIP's softmax contrastive loss for a **sigmoid** loss (simpler, scales better, multilingual). The default VLM encoder today.
- **DINO / DINOv2 (2021–23)** — **self-supervised** (no captions needed) encoders with strong *spatial/geometric* features — good where CLIP is weak (precise localization, structure).

### Vision → LLM fusion (the main VLM line)

This is the heart of the track — *how* you connect a vision encoder to a language model. The arc goes from "barely interact" to "one unified model":

- **Flamingo (DeepMind 2022)** — keep the LLM **frozen**, inject vision via **cross-attention** layers + a "Perceiver resampler" (compresses many image features into a fixed few). Few-shot multimodal. This is **deep fusion** (vision injected throughout the LLM's layers).
- **BLIP / BLIP-2 (2022–23)** — the **Q-Former**: a small module that compresses the image into a handful of learned "query" tokens the LLM can ingest.
- **LLaVA (2023)** — the dominant *simplification* and the one to know. Just use a tiny **MLP projector** to map vision patches into "visual tokens," then **concatenate** them with the text tokens and feed the whole thing into one decoder LLM. Two-stage training: (1) align the projector, (2) instruction-tune. This is "**visual instruction tuning**." Simple, effective, widely copied.
- **Qwen2-VL / InternVL (2024)** — add **native dynamic resolution** (handle images at their real aspect ratio/size instead of forcing a fixed square), **M-RoPE** (multimodal rotary positions), and token compression — crucial for documents and charts.
- **Early / native fusion: Chameleon (Meta 2024), Emu3 (2024)** — drop the separate vision encoder entirely. **Tokenize images** (often via VQ — vector quantization, turning image patches into discrete codebook tokens) into the *same vocabulary* as text, and train one model on interleaved image+text streams. This also enables **image generation** (the model can output image tokens too).
- **Llama 4 / Gemini / GPT-4o-style (2025–26)** — natively-multimodal frontier models (their exact vision stacks are mostly undisclosed).

**The arc to remember:** *late fusion (CLIP) → cross-attention (Flamingo) → prefix-concat (LLaVA) → early-native (Chameleon)* — going from two barely-connected models to one unified model.

### Image generation (feeds the unified models)

- **DDPM (Ho 2020)** — diffusion: generate an image by starting from noise and **iteratively denoising** it.
- **Latent Diffusion / Stable Diffusion (2022), DALL·E 2** — do the diffusion in a compressed *latent* space → fast, high-quality text-to-image.
- This capability then gets **folded into** discrete-token unified models (Chameleon/Emu3) and the 2025–26 "any-to-any" systems (e.g. Gemini-based image models).

---

## Part 5 — RAG & Retrieval

The problem: a pretrained model's knowledge is **frozen** at training time and it can't see your *private* data. **RAG (Retrieval-Augmented Generation)** fixes this: fetch relevant documents at query time and feed them into the prompt, so the model answers *grounded* in fresh/private facts (and can cite them, reducing hallucination).

The chain:
- **DrQA (2017)** — open-domain QA = *retrieve* Wikipedia passages + *read* them for the answer.
- **DPR (Dense Passage Retrieval, 2020)** — learn a **dense bi-encoder**: embed the question and each passage into vectors, retrieve by nearest-neighbor. Beats keyword search (BM25) on *semantic* match.
- **REALM (2020)** — retrieval-*augmented pretraining*: learn the retriever end-to-end with the model.
- **RAG (Lewis 2020)** — coined the name and the pattern: retrieve passages, condition generation on them.
- **FiD (Fusion-in-Decoder, 2021)** — encode many passages separately, **fuse them in the decoder** to answer from multiple documents.
- **ColBERT (2020) / v2 (2021)** — **late interaction**: keep a vector *per token* and match with "MaxSim." A middle ground between cheap bi-encoders and accurate-but-slow cross-encoders.
- **HyDE (2022)** — embed a *hypothetical answer* (let the LLM imagine one) instead of the question, because a fake answer sits closer to the real answer passages in vector space.
- **Self-RAG (2023) / CRAG (2024)** — the model *decides when* to retrieve, *critiques* the retrieved passages, and *self-corrects* if they're bad.
- **RAPTOR (2024)** — recursively cluster + summarize chunks into a **tree**, so you can retrieve at multiple levels of abstraction.
- **GraphRAG (Microsoft 2024)** — extract an entity/relation **knowledge graph** + community summaries, enabling *corpus-wide / global* questions that plain chunk-retrieval can't answer ("what are the main themes across all these docs?").
- **HippoRAG / HippoRAG 2 (2024–25)** — graph + personalized PageRank for cheap multi-hop "memory."
- **Agentic RAG (Search-o1, Search-R1, Graph-R1, 2025)** — make retrieval a *reasoned, iterative tool decision* (search, read, search again), often **RL-trained** end-to-end. This converges with the agents track and is productized as **Deep Research**.

---

## Part 6 — Agents

The shift from "answer a question" to "drive a loop and *act* in the world."

The chain:
- **WebGPT (2021)** — an LLM that browses the web to answer questions.
- **ReAct (2022)** — interleave **Thought → Act → Observation**: the model reasons, takes an action (calls a tool), observes the result, repeats. This grounds reasoning in *real* tool results and is the substrate of nearly every agent since.
- **Toolformer (2023)** — the model *self-teaches* which APIs to call and when, by inserting API calls into training text and keeping the ones that help.
- **Reflexion (2023)** — after a failed attempt, the model writes a **verbal self-critique** ("I failed because..."), keeps it in context, and retries. "Verbal RL" — improvement with no weight updates.
- **OpenAI function calling (2023)** — standardized **structured tool calls**: define a tool with a JSON schema, the model emits a JSON call, you run it and return the observation.
- **AutoGPT / BabyAGI (2023)** — autonomous goal-pursuit hype. Mostly *exposed* how brittle long-horizon loops are (they wander, loop, and fail).
- **Generative Agents (Stanford 2023)** — memory + **reflection** (consolidate experiences into higher-level insights). **Voyager (2023)** — a growing **skill library** (lifelong learning in Minecraft).
- **MetaGPT / ChatDev / AutoGen (2023)** — **multi-agent** role teams (planner / coder / critic collaborate).
- **Tree-of-Thoughts / LATS (2023)** — search-based planning (shared with the reasoning track).
- **Computer use (Anthropic 2024)** — agents that operate a **GUI** (mouse/keyboard/screenshots) like a human.
- **MCP (Model Context Protocol, Anthropic 2024)** — an open standard to expose any tool or data source uniformly. The "USB-C for tools" — write a tool once, any model can use it.
- **Coding agents (Devin 2024, OpenAI Codex agent 2025, Claude Code 2025)** — long-horizon software engineering, benchmarked on **SWE-bench Verified** (real GitHub issues).
- **RL-trained agents + context engineering (2025–26)** — turn-level credit assignment for multi-turn tasks; self-evolving memory (ReasoningBank, Mem0, A-MEM); and **context engineering** (compressing/offloading/isolating what's in the context window) as the dominant practical lever.

**The binding constraint, still unsolved:** **long-horizon reliability.** Per-step success rates *compound*: if each step is 95% reliable, a 20-step task succeeds only ~36% of the time (0.95²⁰). This is *the* wall agents keep hitting.

---

## Part 7 — The meta-pattern: how to read any new paper

This is the actual *skill* the whole guide builds toward. When a 2026+ paper lands, don't ask "what is this." Ask "which chain, and what's the next link." Run these five questions:

1. **Which track?** Architecture/training spine (§1), efficiency (§2), reasoning (§3), multimodal (§4), RAG (§5), or agents (§6)? Usually one, occasionally a merge (agentic RAG = §5×§6).

2. **What predecessor is it beating, and on what axis?** Every entry above improved a *specific* prior link on a *specific* axis: quality / `n²` cost / KV memory / reasoning / context length / reliability / data efficiency. Name both the predecessor and the axis.

3. **Which recurring move is it?** Almost everything is one of a handful of moves:
   - *compress something* (KV → MLA, weights → quantization, vision → Q-Former)
   - *make a quadratic thing linear* (linear attention / SSM / hybrids)
   - *decouple capacity from compute* (MoE)
   - *change the training signal* (SFT → DPO → RLVR → self-verify)
   - *spend compute at inference* (CoT → search → o1)
   - *add a control loop* (RAG, ReAct, agents)
   - *unify modalities/interfaces* (T5 text-to-text, Chameleon tokens-for-everything)

4. **What did it trade away?** There's always a hidden cost, and it's usually the *predecessor's strength*. Linear attention trades exact retrieval. MoE trades VRAM. Reasoning trades latency/cost. Agents trade reliability. Find the cost even when the paper hides it.

5. **Will it survive at a different scale / hardware / base model?** This is the senior judgment. Most links *don't* generalize — a trick that helps a 1B model may vanish at 100B, or depend on one specific GPU. Knowing which ones survive is the real skill.

If you can run those five on a fresh paper in a couple of minutes, you can read almost anything in the field and understand it as the next move in a game you already know.

---

## Cheat sheet

**The one-line spine:** recurrence → attention (2017) → pretrain (2018) → scale + in-context learning (2020) → Chinchilla correction (2022) → RLHF/DPO alignment (2022–23) → open recipe LLaMA (2023) → efficient MoE/MLA/FP8 (2024) → reasoning via RLVR (2024–25) → hybrids + agentic (2025–26).

**The seven recurring moves:** compress · linearize the quadratic · decouple capacity from compute (MoE) · change the training signal · spend compute at inference · add a control loop · unify modalities.

**The three enemies the field keeps fighting:** attention's `O(n²)` cost · the KV-cache memory wall · long-horizon agent reliability.

**The alignment ladder:** SFT → RLHF (reward model + PPO) → DPO (no RL) → RLVR (verifiable rewards) → self-verifiable reasoning.

**The fusion arc (vision):** late (CLIP) → cross-attention (Flamingo) → prefix-concat (LLaVA) → early-native (Chameleon).

**The numbers worth remembering:** Chinchilla ≈ 20 tokens/param (compute-optimal training); modern models *overtrain* to ~1000+ tokens/param (inference-optimal); hybrid attention ≈ 3:1 linear-to-full blocks.

When in doubt, return to the framing: **every paper is a fix for a specific pain in a specific predecessor, and it creates a new pain that the next paper attacks.** Learn the pains, and the field becomes a story instead of a list.
