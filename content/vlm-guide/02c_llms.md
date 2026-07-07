# Large Language Models — Part 3 of 3: Reasoning, Decoding & Long Context

This is part 3 of the Large Language Models lesson. Parts 1–2 covered architecture and training; here we cover reasoning models and test-time compute, decoding strategies, long context, and a checklist for reading any LLM paper end to end.

---

## 6. Reasoning models and test-time compute

A **reasoning model** (o1/o3, DeepSeek-R1, QwQ, Gemini "thinking," Claude with extended thinking) is one trained (usually via RLVR) to produce a long internal **chain-of-thought** before its final answer, and that spends *more tokens thinking* on harder problems.

The core idea is **test-time (inference-time) compute scaling**: instead of only scaling *training*, you scale *thinking at inference*. More reasoning tokens → better answers, a *second* scaling axis orthogonal to model size. Empirically, a smaller model that thinks longer can beat a bigger model that answers immediately, on reasoning tasks.

Ways to spend test-time compute (you'll see these as decoding/search strategies):
- **Long CoT (sequential):** just generate a long reasoning trace. What R1/o1 do internally.
- **Self-consistency:** sample many CoTs, take the majority answer.
- **Best-of-N / rejection sampling:** sample N, pick the best per a verifier/RM.
- **Search:** Tree-of-Thoughts, MCTS over reasoning steps, beam search over thoughts. Higher cost, sometimes higher ceiling.

Open questions you'll see debated: does RLVR *create* new reasoning ability or just *elicit/sharpen* what the base model already had? When should the model stop thinking ("overthinking" wastes compute and can hurt)? How to reward *process* vs *outcome*? These are live research fronts, not settled.

### 6.1 Controlling test-time compute

The theoretical framing is one thing; the practical question is *how do you control how much thinking the model does, and how much does that cost?*

**s1 (Jan 2025) — the minimal proof of concept:** fine-tune Qwen2.5-32B on just **1,000 carefully curated reasoning examples** (quality over quantity — problems chosen for difficulty and diversity, solutions verified). Then apply **budget forcing** at inference:
- To force *more* thinking: append the token "Wait" when the model tries to stop early, pushing it to reconsider.
- To force *less* thinking: hard-stop the chain-of-thought at a token budget and force the final answer.

The result: s1 **beats o1-preview** on competition math (MATH500, AIME) despite costing orders of magnitude less to train. The insight: for a base model with strong pretraining, the reasoning capability is largely latent — you don't need millions of RL steps, you need a clean signal + the ability to steer compute at inference.

**Compute-optimal adaptive allocation:** not all problems need the same compute. A simple algebra problem solved in 50 tokens doesn't benefit from 2,000 tokens of deliberation; a competition geometry problem does. **Adaptive budget allocation** — giving harder problems more thinking budget based on model confidence or estimated difficulty — gives **2–4× efficiency over fixed budgets** at the same average accuracy. The practical pattern: start with a short budget, check if the answer is confident/consistent, extend only if uncertain.

**The elicit-vs-create debate — with evidence both ways:**

*Evidence for "elicit only":* arXiv:2504.13837 (NeurIPS 2025) shows that RLVR improves **sampling efficiency** (the model reaches the correct answer in fewer samples) but **rarely exceeds the base model's pass@k frontier** — the ceiling of what the base model can produce with many samples is unchanged by RLVR. Interpretation: RLVR focuses the model's existing capability toward better single-sample behavior; it doesn't expand the frontier of what's reachable.

*Evidence for "create":* arXiv:2602.08281 shows that RLVR **can compose novel capability from atomic sub-skills** that exist in the base model separately — skills the base model cannot combine into a correct solution even at very high pass@k. RLVR, in this setting, genuinely creates a new capability by learning to orchestrate sub-skills the base model has but cannot chain. Both findings are real; the honest answer is "eliciting *and* occasionally creating, depending on whether the task requires novel composition."

Reading takeaway: "budget forcing" and "adaptive compute allocation" are the practical toolkit; the elicit-vs-create debate tells you what those tools can actually buy.

When to *use* a reasoning model: math, code, logic, multi-step planning — yes. Simple factual/creative/latency-sensitive tasks — often overkill (slower, costlier). This tradeoff is itself a design decision in agent/RAG systems.

---

## 7. Decoding — turning distributions into text

At each step the model gives `P(next token)`. How you pick from it is **decoding**, and it changes outputs a lot:

- **Greedy:** always take argmax. Deterministic, repetitive, often dull.
- **Temperature `T`:** divide logits by `T` before softmax. `T<1` sharpens (more confident/deterministic), `T>1` flattens (more random/creative). `T→0` ≈ greedy.
- **Top-k:** sample only from the `k` most likely tokens.
- **Top-p (nucleus):** sample from the smallest set of tokens whose cumulative probability ≥ `p`. Adapts to how peaked the distribution is. The common default (e.g. `p=0.9–0.95`).
- **min-p:** keep tokens above a fraction of the top token's probability — a newer, often-better adaptive cutoff.
- **Repetition / presence / frequency penalties:** discourage repeating tokens.
- **Beam search:** keep `b` candidate sequences, expand all, keep best `b`. Good for low-entropy tasks (translation), bad for open-ended (bland, repetitive). Rare for chat.
- **Constrained / structured decoding:** mask logits to force valid JSON / a grammar / a regex. How "guaranteed valid JSON" / tool-call formatting works (e.g. via finite-state machines over the vocab). Important for agents.
- **Speculative decoding** (an *acceleration*, not a sampling change) → covered in the inference chapter; it produces the *same* distribution faster.

### 7.1 Constrained decoding engines

Structured decoding works by maintaining a grammar state machine alongside the token stream and masking out any token that would produce an invalid continuation — but the naive implementation re-parses the grammar from scratch each step, which is prohibitively slow.

**XGrammar** solves this with precompiled context-free grammar automata and per-token bitmask caches — overhead is **<40µs per token**, low enough to be invisible in production. XGrammar is now the **default structured-output backend in vLLM, SGLang, and TensorRT-LLM**. When a serving framework advertises "guaranteed valid JSON" or "schema-constrained output," it is almost certainly XGrammar underneath.

**llguidance (Microsoft)** takes a different approach: a **Rust implementation of an Earley parser** that handles the full class of context-free grammars (not just regular/pushdown subsets). It underlies **OpenAI's Structured Outputs** API — the reason you can pass an arbitrary Pydantic schema and always get valid JSON back.

**XGrammar-2 (Jan 2026)** extends XGrammar to **dynamic grammars** — grammars that are not fixed at request time but constructed on-the-fly, e.g. an agent carrying 100+ tool schemas where the valid call structure depends on which tools are loaded for the current task. Static precompilation doesn't work when the grammar changes per request; XGrammar-2 handles incremental grammar updates without the per-token overhead blowing up.

You care because this is the infrastructure that makes **function calling reliable** at scale. Without constrained decoding, tool-call JSON fails ~5–15% of the time at production traffic; with it, failures approach zero. When a paper or system says "reliable tool use," constrained decoding is doing most of the heavy lifting.

Reasoning models usually run at a modest temperature; tool-calling/structured tasks often near-greedy + constraints. Knowing these knobs explains a lot of "why did the output change" and a lot of eval methodology.

---

## 8. Long context

Two separate problems, often conflated:
1. **Can the model *technically* run at length `n`?** Bounded by the `O(n²)` attention cost (compute + KV-cache memory). Addressed by FlashAttention (in the inference chapter), GQA/MLA (smaller cache), linear/SSM hybrids (flat memory, §3.3), and RoPE-extension tricks (PI/NTK/YaRN, from the foundations).
2. **Can it *use* the context well?** Even when it runs, models suffer **"lost in the middle"** — they attend well to the start and end of a long context but miss the middle. Tested by **needle-in-a-haystack** (plant a fact at a random depth, see if it's retrieved). A long *context window* ≠ good long-context *reasoning*; benchmarks like RULER and LongBench probe the gap.

Related serving optimizations (StreamingLLM/attention sinks: keep the first few tokens + a recent window; DuoAttention: full cache only for "retrieval heads") let you run effectively-unbounded streams cheaply — see the inference chapter.

This is also exactly the tension that motivates RAG: rather than stuff everything into a giant context, retrieve only what's relevant. "Long context vs RAG" is a recurring real-world design debate; the honest answer is usually *both* (retrieve to fill a large-but-finite context well).

---

## 9. Reading-an-LLM-paper checklist

- **Architecture:** which attention variant (MHA/GQA/MLA/linear/hybrid)? Dense or MoE (total vs active params)? Anything non-standard in norm/FFN/positional?
- **Training:** base or post-trained? If post-trained: SFT only? DPO? RLVR/GRPO? What's the reward signal?
- **Is it a reasoning model?** Does it scale test-time compute? How is "thinking" trained and bounded?
- **Scale & data:** params, tokens, tokens/param (vs Chinchilla ~20), data mix, synthetic data.
- **Context:** trained length, extended length, *and* how they show it actually uses it (not just runs).
- **The one-sentence contribution**, and **the tradeoff they paid.**

---

## You can now

- Decompose any modern LLM into its four axes — attention variant (MHA/MQA/GQA/MLA), FFN variant (dense vs MoE), sequence mixer (full vs linear/SSM/hybrid), and positional/norm details — and read a model card as a set of deliberate tradeoffs.
- Explain the KV-cache pressure that drives the MHA→MQA→GQA→MLA ladder, and why MoE decouples total parameters from compute-per-token.
- Place any post-training method on the signal hierarchy — SFT imitates, RLHF/DPO prefer, RLVR/GRPO verify — and explain why GRPO drops PPO's critic by using the group mean as a baseline.
- Reason about test-time compute scaling: when a smaller model that thinks longer beats a bigger one, and how budget forcing and adaptive allocation control the cost.
- Choose decoding knobs (temperature, top-p, min-p, constrained decoding) for a task, and separate "can run at length n" from "can actually use length n" in long-context claims.

## Try it

Pick a recent open model's technical report (e.g. Qwen3, DeepSeek-V3/V4, Kimi, GLM) and fill in the §9 reading checklist end to end: name its attention variant and KV-cache strategy, whether it is dense or MoE (total vs active params, tokens/param vs Chinchilla ~20), its exact post-training recipe and reward signal, whether it is a reasoning model and how "thinking" is bounded, and its trained-vs-extended context plus the evidence it actually *uses* that context. Finish with one sentence stating the paper's core contribution and the tradeoff it paid. If any axis is unclear from the report, note that as a gap — that itself is a finding.
