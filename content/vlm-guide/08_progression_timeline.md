# The Progression: A Chained Lineage (2017 → 2026) — Part 1 of 2: Prehistory & Main Spine

This part (1 of 2) covers Part 0 (prehistory) and Part 1 (the main architecture/pretraining/alignment spine). Part 2 covers the five parallel tracks — efficiency, reasoning, multimodal, RAG, and agents — plus the meta-pattern and concepts recap.

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

## You can now

- Explain why the Transformer (2017) was a break from RNNs/LSTMs/seq2seq — parallel training and no long-distance forgetting — and what cost it introduced in exchange (the `O(n²)` attention bill).
- Trace the main spine's causal chain: pretraining paradigms fork (GPT vs BERT, 2018) → scale unlocks in-context learning (GPT-3, 2020) → the scaling recipe gets corrected (Chinchilla, 2022) → alignment turns base models into assistants (RLHF → DPO) → the open recipe (LLaMA) → the efficient-flagship template (DeepSeek) → reasoning-by-RL (o1 → R1).
- For any entry in Part 0 or Part 1, state what it fixed about its predecessor and what limitation remained that motivated the next entry — the "chain of fixes" reading habit this timeline is meant to build.
- Recognize the "dating note": pre-2024 history here is settled; 2025+ entries are trend illustrations, not a list to memorize verbatim.

