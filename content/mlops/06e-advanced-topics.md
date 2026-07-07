# 06 — Advanced Topics: Everything Else Worth Knowing — Part 5 of 5: The Bookshelf and Closing Notes

This is part 5 of the Advanced Topics reference catalog, and the last part. Here we cover the recommended reading list, a closing note on the patterns underneath all of this, and where to go next in the curriculum.

## Phase 17 — The Bookshelf

### Tier 1 — Read These

1. **Designing Machine Learning Systems** — Chip Huyen. The textbook. Twice.
2. **Designing Data-Intensive Applications** — Martin Kleppmann. The distributed systems bible.
3. **Machine Learning Design Patterns** — Lakshmanan, Robinson, Munn.
4. **Reliable Machine Learning** — Chen et al. (O'Reilly).

### Tier 2 — Strongly Recommended

5. **Practical MLOps** — Noah Gift.
6. **Building Machine Learning Powered Applications** — Emmanuel Ameisen.
7. **The Hundred-Page Machine Learning Book** — Andriy Burkov (foundations refresher).
8. **Deep Learning** — Goodfellow, Bengio, Courville (the formal text).
9. **Streaming Systems** — Akidau, Chernyak, Lax.
10. **Database Internals** — Alex Petrov.

### Tier 3 — LLM-Specific

11. **Hands-On Large Language Models** — Jay Alammar, Maarten Grootendorst.
12. **Build a Large Language Model (From Scratch)** — Sebastian Raschka.
13. **Generative Deep Learning** (2nd ed) — David Foster.

### Papers Worth Reading

- **Attention Is All You Need** (Vaswani et al.) — Transformers.
- **GPT-3, GPT-4 technical reports** — what scale does.
- **LLaMA, LLaMA-2, LLaMA-3 papers** — open model design.
- **InstructGPT** — RLHF for alignment.
- **DPO paper** — the simpler alternative.
- **FlashAttention 1/2/3** — attention optimization.
- **vLLM paper** — efficient LLM serving.
- **Chinchilla** — scaling laws.
- **Mixtral / DeepSeek-V3** — MoE in practice.
- **Constitutional AI** (Anthropic) — RLAIF.
- **The DeepSeek-R1 paper** — GRPO and reasoning training.
- **Mamba / RWKV** — alternatives to attention.

### Blogs and Newsletters

- **Chip Huyen's blog**
- **Eugene Yan**
- **Sebastian Raschka's Magazine**
- **Lilian Weng's blog** (OpenAI)
- **Hugging Face blog**
- **Anthropic / OpenAI / DeepMind research blogs**
- **The Latent Space podcast and newsletter**
- **MLOps Community Slack and newsletter**
- **The Pragmatic Engineer** (Gergely Orosz) for engineering culture
- **Databricks / Snowflake / NVIDIA engineering blogs**

### Conferences (Watch Talks Online)

- **NeurIPS, ICML, ICLR** — research; pick MLOps-adjacent papers
- **MLOps World**
- **NVIDIA GTC** — infrastructure at scale
- **KubeCon + AI/ML Day**
- **Data + AI Summit** (Databricks) — practical at scale
- **Ray Summit** — Ray + distributed ML

---

## Phase 18 — A Closing Note

You'll never finish this curriculum. New tools appear monthly. The point isn't to know everything; it's to internalize the **underlying patterns** so deeply that any new tool slots into your mental model in a day.

Patterns that recur:

- Storage/compute separation, reinvented every cycle
- Lazy evaluation and predicate pushdown in every modern engine
- Eventually-consistent replication with strong-consistency islands
- Idempotency as the primary defense against distributed failures
- Schema evolution as a first-class operational concern
- Cost as a function of bytes scanned, parameters, tokens, compute time
- The training-serving skew bug, in some new disguise, every few months
- "Just batch it" as the answer to most throughput problems
- Caching at every layer

Master these and the rest is vocabulary.

---

## What to Do Next

You've now seen the full landscape. The honest path forward:

1. **Finish the foundations through specialization chapters.** Work through them sequentially.
2. **Pick a specialization** from the next-steps chapter and go deep.
3. **Build two portfolio projects** from the Fortune 50 portfolio chapter. Slowly. Deeply.
4. **Use this chapter as reference** when problems push you into new territory.
5. **Read Designing Machine Learning Systems and DDIA at least twice each.**

The compound interest on solid fundamentals over 12–18 months is genuinely transformative. Most candidates skip them and stay mid-level forever. Don't be most candidates.

When you're ready to think about the role *above* senior IC, continue to the ML architect track.

---
## You can now

- Navigate the Tier 1-3 reading list, the papers, and the blogs/newsletters/conferences worth tracking as your ongoing MLOps curriculum.
- Recognize the recurring patterns underneath this whole reference catalog (storage/compute separation, idempotency, training-serving skew, caching at every layer) so a genuinely new tool slots into your mental model quickly.
- Decide what to do next in the curriculum — finish the foundations sequentially, pick a specialization, build portfolio projects, or continue on to the ML architect track.
