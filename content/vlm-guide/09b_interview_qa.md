# Interview Q&A (the full bank) — Part 2 of 2: VLMs, Agents, Deep-Dives & System Design

This is part 2 of the interview Q&A bank. Part 1 covered sections A–F: foundations through RAG. Here we cover VLMs/multimodal, agents, the tricky "modify the architecture" deep-dive bank, ML system design, and rapid-fire questions. Same grading lens as part 1: depth (the *why*), production experience (real numbers), tradeoff awareness, and knowing what frameworks actually hide.

---

## G. VLMs / multimodal

**Q: Walk through a VLM's architecture.**
Vision encoder (pixels → visual features) → projector/connector (map into the LLM's embedding space → visual tokens) → LLM (decoder attends over text + visual tokens) → text out. Most modern open VLMs are LLaVA-style: frozen-ish encoder + small trained MLP projector + pretrained LLM. (the VLM chapter)

**Q: How does ViT turn an image into tokens?**
Split into fixed patches (e.g. 16×16), flatten+linearly-project each patch to a vector (patches are the "tokens"), add 2D positional embeddings + a CLS token, run through transformer layers. A 224² image at patch-16 → 196 tokens. Patch count drives both spatial resolution and token cost. (the VLM chapter)

**Q: CLIP vs SigLIP vs DINO — why does the encoder choice matter?**
CLIP/SigLIP are contrastively trained on image-text pairs, so their features are *already language-aligned* — the default for VLMs (SigLIP's sigmoid loss scales better; SigLIP 2 is multilingual + dense). DINO is self-supervised (no text), strong at spatial/geometric structure where CLIP is weak. The pretraining objective tells you what the encoder is good at; document/OCR VLMs care about resolution + spatial fidelity, so encoder choice matters more there. (the VLM chapter)

**Q: Explain the VLM fusion spectrum.**
Late fusion (CLIP): separate encoders, interact only via similarity — great for retrieval, not generative. Cross-attention fusion (Flamingo): text attends to vision via inserted cross-attn layers, LLM weights mostly intact. Prefix-concat (LLaVA, dominant): project vision to tokens, concatenate with text, one decoder's self-attention does the fusion. Early/native (Chameleon, Emu3, Llama 4): tokenize images into the same vocabulary, train one model from scratch on interleaved streams — can generate images too. Spectrum = barely-interact → one-unified-model. (the VLM chapter)

**Q: Projector choices — MLP vs Q-Former vs pixel-shuffle, and the core tradeoff?**
MLP (LLaVA): keep every patch as a token, simple, strong — but token count = patch count. Q-Former (BLIP-2): learned query tokens cross-attend to extract a *fixed small* number of tokens (e.g. 32), aggressive compression, complex to train. Pixel-shuffle/pooling: merge 2×2 patches → 4× fewer tokens. Core tradeoff the projector mediates: number of visual tokens (cost, since the LLM is O(n²)) vs information/detail preserved (OCR, spatial precision). (the VLM chapter)

**Q: Continuous vs discrete visual tokens?**
Continuous: real-valued vectors from encoder+projector, fed as soft embeddings — can't be *generated* by a discrete LM head. Discrete (VQ-VAE/VQGAN): quantize patches into codebook indices added to the vocabulary, so image and text share one discrete space and the model can *generate* images by predicting image tokens (Chameleon, Emu3). (the VLM chapter)

**Q: How is a LLaVA-style VLM trained?**
Stage 1 (alignment): freeze encoder + LLM, train only the projector on image-caption data. Stage 2 (instruction tuning): unfreeze the LLM (sometimes encoder, gradually), train on multimodal instruction data (VQA, OCR, charts, grounding). Stage 3 (optional): DPO/RLHF/RLVR for preferences and to reduce hallucination. (the VLM chapter)

**Q: Why do VLMs hallucinate objects, and how do you reduce it?**
Strong language priors override weak visual grounding — the model "expects" objects that co-occur in training text. Reduce with better grounding data, DPO against hallucinated descriptions, and higher resolution (small objects/text get lost at low res). Measured by POPE/CHAIR. (the VLM chapter)

---

## H. Agents

**Q: What makes something an "agent" vs a chatbot?**
The model drives a *loop*: it can take actions (tools), observe results, and decide what to do next toward a goal — rather than emitting one answer and stopping. The LLM is the policy/brain; the loop + tools are scaffolding. Four capabilities: reasoning/planning, tool use, memory, coordination. (the agents chapter)

**Q: Explain ReAct.**
Interleave Thought → Action → Observation: the model reasons about what to do, emits a tool call, reads the result, repeats. Grounding reasoning in real observations lets it adapt to errors and changing state instead of hallucinating a plan. It's the substrate of most agents. (the agents chapter)

**Q: How does function calling / tool use actually work?**
The model is given tool schemas (name, description, JSON params). To act, it emits a structured call (name + JSON args), the runtime executes it, the result returns as an observation. The model is post-trained to produce these, and constrained decoding guarantees valid JSON. It picks tools by their *descriptions*, so description quality and good error messages (for recovery) matter a lot; too many tools → tool-retrieval (RAG over the tool catalog). (the agents chapter)

**Q: What is MCP?**
Model Context Protocol — an open standard for exposing tools/data/resources to models uniformly, so any MCP client can use any MCP server. "USB-C for tools": decouples tool providers from agent builders. (the agents chapter)

**Q: Planning without feedback vs with feedback?**
Without feedback (plan-then-execute): generate a full plan upfront — CoT, decomposition, Tree-of-Thoughts. Efficient but brittle if reality diverges. With feedback (interleaved): plan, act, observe, replan — ReAct, Reflexion (verbal self-critique + retry), LATS (tree search + acting). Robust but more tokens/latency. (the agents chapter)

**Q: How does agent memory work, and how is it different from RAG?**
Short-term = the context window (current task). Long-term = persisted across sessions, retrieved when relevant — vector memory (RAG over past interactions), structured/graph memory, or episodic/semantic/procedural splits, often with reflection (summarize experiences into insights). Difference: RAG retrieves from an external *knowledge* corpus; agent memory retrieves from the agent's own *experience*. Same machinery, different content/lifecycle. (the agents chapter)

**Q: When is multi-agent worth it, and when not?**
Worth it when the task genuinely decomposes into parallelizable or cleanly-separable sub-tasks, when role specialization helps, or for context isolation (each subagent gets a clean window). Not worth it for tasks a single well-engineered agent handles — multi-agent adds coordination overhead, error propagation, token cost (often many×), and harder debugging. (the agents chapter)

**Q: What is context engineering and why did it replace prompt engineering?**
The context window is finite *and* models degrade as it fills (lost-in-the-middle, distraction, cost). Context engineering curates exactly what's in the window each step: compression/summarization of old turns, selective retrieval of relevant tools/memories/files, offloading big artifacts outside context (files/handles), structured + cache-friendly prefixes, subagent isolation. A lot of real agent performance comes from this, not a smarter model. (the agents chapter)

**Q: What's the central unsolved problem in agents?**
Long-horizon reliability. Per-step errors compound: 95% per-step success → ~60% over 10 steps. Capability isn't the binding constraint for real deployments; reliability over long trajectories is. Watch for compounding errors, looping, context rot, and recovery from tool failures. (the agents chapter)

---

## I. Tricky / "modify the architecture" / deep-dive

These are the hardest ones — where they hand you a constraint and watch you reason, or push on *why*. Frameworks, not memorized answers.

**Q: Your KV cache OOMs at 100k-token context. What do you change, in what order?** *(classic design-under-constraint)*
Reason from the bottleneck (KV memory). In rough order of cost/effort: (1) switch to GQA/MQA if on MHA — biggest cache cut for least work; (2) quantize the KV cache to FP8/INT4 (KIVI/KVQuant) — 2–4× with a quality check; (3) eviction/sinks — StreamingLLM (keep first "sink" tokens + recent window) or DuoAttention (full cache only for retrieval heads); (4) PagedAttention so you're not wasting on fragmentation; (5) chunked prefill so the long prompt doesn't stall others; (6) architecturally, move to MLA or a linear-attention hybrid (flat memory); (7) step back — does this even need 100k in-context, or should it be RAG? Name the quality tradeoff for each lossy option. (the inference chapter, the LLM chapter)

**Q: Make attention linear. What do you lose, and how do you get it back?** *(deep-dive)*
Linear attention (kernel feature map + running sum) drops `n²`→`n` and removes the growing KV cache, but you lose *exact* content-based retrieval — the softmax attention matrix is what lets a token sharply attend to one specific far-away token; a fixed-size recurrent state blurs that. You get it back with a hybrid: keep a few full-attention layers (the ~3:1 pattern) so exact retrieval is preserved where it matters while linear blocks carry the bulk cheaply. (the LLM chapter)

**Q: Design a module to add a new prior (say, barcode/region awareness) into a transformer's attention. How?** *(architecture-modification, VLM/document flavor)*
Don't retrain from scratch — inject a *bias*. Add a learned additive term to the attention logits before softmax (`QKᵀ/√d + B`), where `B` encodes the prior (e.g. higher bias toward known barcode regions, or a 2D-distance bias for spatial structure). It's parameter-light, differentiable, composes with existing attention, and you can ablate it cleanly. Alternatives: a dedicated cross-attention to a region-feature stream, or a doc-type conditioning token prepended to the sequence. Then justify: minimal params, preserves pretrained weights, easy to A/B. (the foundations, the VLM chapter)

**Q: Your MoE router collapses to 3 experts. Diagnose and fix.**
Diagnosis: no balancing pressure, so the router exploits early-favored experts (rich-get-richer). Fixes: add an auxiliary load-balancing loss (penalize deviation from uniform expert load), or DeepSeek-V3's auxiliary-loss-free bias adjustment (add a per-expert routing bias nudged to balance, no competing loss); also check capacity factors (token dropping) and router learning rate / noise (noisy top-k routing helps exploration). Verify with per-expert utilization histograms. (the LLM chapter)

**Q: Extend a model trained at 4k context to 128k without full retraining. How, and what breaks?** *(deep-dive on RoPE)*
RoPE high-frequency rotations alias/wrap at distances unseen in training, so naive extension degrades. Options: Position Interpolation (squeeze positions into the trained rotation range — simple, some resolution loss), NTK-aware scaling (change the rotation base, preserves high-freq resolution), or YaRN (refined NTK+interpolation, the common "extended to 128k" method), usually with a short continued-pretraining/fine-tune at long context. What breaks even after it *runs*: "lost in the middle" — running at 128k ≠ using 128k well; verify with needle-in-a-haystack/RULER, not just that it doesn't crash. (the foundations, the LLM chapter)

**Q: Reduce a VLM's visual tokens 4× with minimal quality loss. How?**
Pixel-shuffle/spatial pooling (merge 2×2 patches → 1 token) before the projector, or a Q-Former/resampler to a fixed small token count, or token pruning/merging (drop low-information patches). The tradeoff is detail — fine for natural-image captioning, risky for OCR/charts where small text lives in those tokens. So: choose compression by *task*, keep higher token budgets for document understanding, and measure on DocVQA/ChartQA not just captioning. (the VLM chapter)

**Q: Your RLHF-trained model started reward-hacking (verbose, sycophantic, gaming the RM). Diagnose.**
The policy found high-reward regions the reward model scores well but humans don't like — the RM is a proxy and the policy is over-optimizing it (Goodhart). Fixes: stronger KL penalty to the reference model (limit drift), better/retrained RM with adversarial examples, length normalization (penalize verbosity the RM rewards), or switch the signal — DPO (no separate RM to hack) or RLVR (verifiable reward can't be flattered). This is exactly why verifiable rewards became attractive for math/code. (the LLM chapter)

**Q: Why does removing the FFN hurt more than you'd think? What does each sub-layer do?** *(deep-dive)*
Attention *moves* information between positions (mixing across the sequence); the FFN *processes* information within each position (mixing across features) and holds the bulk of the parameters — it's where much factual "knowledge" is stored (key-value memory view). Remove/shrink it and you keep routing but lose per-token computation and storage capacity, so quality drops sharply even though attention is "the famous part." (the foundations)

**Q: A reasoning model "overthinks" simple questions — wastes tokens, sometimes worse answers. What do you do?**
Recognize test-time compute isn't free or monotonic. Options: train/prompt it to allocate thinking by difficulty (budget-aware), add a stop-thinking signal / length penalty in RL, route easy queries to a non-thinking mode (hybrid thinking models like Qwen3 do this), or cap reasoning tokens. The deeper point: more thinking helps on hard reasoning, hurts on simple/factual/latency-sensitive tasks — match the tool to the task. (the LLM chapter)

**Q: You see attention "sink" tokens (huge attention on the first token / BOS). Bug or feature?**
Feature, mostly. Models learn to dump excess attention onto a few initial tokens as a no-op when no real token is relevant (a "register"). It's why StreamingLLM keeps the first few tokens when sliding the window (drop them and quality collapses), and why KV-quantization schemes specially preserve sink tokens. Not a bug to fix — a behavior to accommodate. (the inference chapter)

**Q: Estimate the cost/latency of running 1M documents through a model.** *(the senior gut-check)*
Show the back-of-envelope: tokens per doc × 1M = total input tokens; prefill is compute-bound so estimate via model FLOPs/token and GPU throughput (or just $/1M-token API pricing × volume); if generating, add output tokens × per-token decode latency (bandwidth-bound). Then mention levers: batching/throughput mode, prefix caching if docs share a prompt, a smaller/quantized model, or whether you even need the full model (route easy docs to a cheaper one). The point is you can reason about cost, not the exact number. (the inference chapter)

**Q: Why not just use a bigger context window instead of RAG?** *(trap)*
Because (1) `n²` cost — huge contexts are expensive in compute and KV memory; (2) lost-in-the-middle — models don't use the middle of long contexts well, so stuffing everything in *lowers* effective accuracy; (3) freshness/provenance — RAG gives citations and updates without retraining; (4) you often have far more corpus than any window. The honest answer is "both": retrieve to fill a large-but-finite context *well*. (the LLM chapter, the RAG chapter)

---

## J. ML system design (frameworks for the open-ended ones)

For any "design an X" prompt, structure the answer: **requirements (latency/cost/scale/quality SLOs) → data → architecture → training/eval → serving/inference → monitoring → failure modes & tradeoffs.** Always state assumptions and name tradeoffs; interviewers grade the reasoning, not a "right" diagram.

**Q: Design a production RAG system for internal company docs.**
Requirements: corpus size, freshness, latency, citations, access control. Pipeline: ingestion (parse, structure-aware chunk, embed with a domain-fit model, store in a vector DB with metadata) → query (rewrite, hybrid dense+BM25 retrieve top-50, cross-encoder rerank top-5, build prompt with citations, generate) → guardrails (faithfulness check, "I don't know" fallback). Ops: metadata filtering for permissions, document versioning, periodic re-indexing, prefix caching for the system prompt, eval harness (retrieval recall + faithfulness), monitoring for stale/irrelevant answers. Tradeoffs: chunking strategy, top-k vs latency, rerank cost. (the RAG chapter)

**Q: Design an agent that resolves customer support tickets end-to-end.**
Requirements: success rate, escalation policy, latency, safety. Loop: ReAct over tools (KB search/RAG, account lookup, ticket update, refund API) with structured function calls; planning for multi-step tickets; memory of the customer's history; human-in-the-loop escalation on low confidence or high-risk actions. Reliability: step budgets, loop detection, action confirmation for irreversible operations, trajectory logging. Eval: task-completion on a held-out ticket set + trajectory quality, not just final-answer. Tradeoffs: autonomy vs safety, single vs multi-agent, context engineering for long tickets. (the agents chapter)

**Q: Design on-device deployment of a VLM for document capture (phone).**
Requirements: memory/thermal/battery limits, offline, latency. Choices: small VLM (sub-4B), aggressive quantization (QAT-derived int4/int8 with a layer-sensitivity map), runtime per platform (MLX/CoreML on iOS, MNN/LiteRT on Android), high-enough resolution for OCR with token compression to fit memory, GQA/sliding-window to shrink KV. Gotchas: immature GPU kernels for newer ops (CPU fallback), tokenizer mismatch on conversion (silent garbage), multimodal-RoPE export bugs. Validate transfer on *real* captures, since synthetic document data lacks real geometric/spatial distortions. (the inference chapter, the VLM chapter)

---

## K. Rapid-fire (crisp answers for the quick ones)

- **Q: Why residual connections?** Preserve gradient flow and give each layer a clean "read/edit the residual stream" path; enable very deep stacks. (the foundations)
- **Q: What's perplexity?** `exp(cross-entropy loss)` — average branching factor; lower = more confident/accurate next-token prediction. (the LLM chapter)
- **Q: Zero-shot vs few-shot vs in-context learning?** Zero/few-shot = task from instruction alone / with a few in-prompt examples; in-context learning = conditioning on the prompt to do a task *without weight updates*. (the LLM chapter)
- **Q: Why CoT works?** It lets the model externalize intermediate computation into tokens it can condition on, turning a one-shot guess into a multi-step computation; effective mainly at scale. (the LLM chapter, the progression timeline)
- **Q: Greedy vs beam search — when beam?** Beam keeps b candidate sequences; good for low-entropy tasks (translation), bad for open-ended chat (bland/repetitive). (the LLM chapter)
- **Q: What does the LM head do, and weight tying?** Projects the final hidden state to vocab logits; tying shares it with the input embedding matrix (saves params, often helps) — though some large models decouple them for big tokenizers. (the foundations)
- **Q: Embedding dimensionality tradeoff (incl. Matryoshka)?** Higher dim = more expressive but costlier storage/search; Matryoshka embeddings let you truncate to a shorter prefix with graceful degradation. (the RAG chapter)
- **Q: Sliding-window attention?** Each token attends only to a fixed recent window (O(n) not O(n²)); cheap long context, with stacked layers giving an effectively larger receptive field (Mistral). (the LLM chapter)
- **Q: Multi-token prediction (MTP)?** Predict several future tokens per step during training for a richer signal (DeepSeek-V3); can also seed speculative decoding at inference. (the progression timeline)
- **Q: Distillation in one line?** Train a small student to mimic a big teacher (soft labels or teacher-generated data); how R1's reasoning went into small models. (the inference chapter, the LLM chapter)
- **Q: What's "lost in the middle"?** Models recall info at the start/end of a long context better than the middle — a long window ≠ good long-context use. (the LLM chapter)
- **Q: BLEU/ROUGE limitations?** N-gram overlap metrics; correlate weakly with quality for open-ended generation — supplement with human/LLM eval. (the reading-papers chapter)
- **Q: What's an induction head?** An attention head that implements in-context copying (sees "A B … A" → predicts "B"); a mechanistic basis for in-context learning. (the foundations)

---

**Final prep advice (the meta-answer):** for any question, lead with the one-sentence *why*, then the *mechanism*, then the *tradeoff*, and if you can, a *real number or project* you've shipped. For "modify/design" questions, reason from the bottleneck (the two-phase inference model is the master key for efficiency questions; the four-primitive frame for everything else), enumerate options in order of cost/effort, and always name what each option trades away. That structure is what reads as senior.

---

## You can now

- Answer the VLM bank (architecture walk-through, ViT patching, CLIP/SigLIP/DINO tradeoffs, fusion spectrum, projector choices, hallucination causes) and the agents bank (ReAct, function calling, MCP, memory vs RAG, multi-agent tradeoffs, context engineering, long-horizon reliability) at senior depth.
- Handle the "modify the architecture" / deep-dive questions by reasoning from a named bottleneck (KV-cache OOM, linear attention's retrieval loss, MoE router collapse, RoPE context extension, reward hacking) and enumerating fixes in cost/effort order.
- Structure any open-ended "design an X" system-design answer: requirements → data → architecture → training/eval → serving → monitoring → failure modes, stating assumptions and naming tradeoffs explicitly.
- Fire off the rapid-fire bank (residuals, perplexity, zero/few-shot, CoT, beam search, weight tying, Matryoshka embeddings, sliding-window attention, MTP, distillation, lost-in-the-middle, BLEU/ROUGE limits, induction heads) without hesitation, each with its one-line *why*.
