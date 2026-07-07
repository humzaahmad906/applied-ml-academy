# 18 — Interview Bank IV: Breadth & Rapid-Fire — Part 4 of 4: Rapid-fire & how to practice

This is part 4 of 4 of the Interview Bank IV: Breadth & Rapid-Fire lesson. Here we cover the
rapid-fire "intuition/trivia" round — thirty short probes spanning everything in this bank — plus
guidance on how to drill breadth for speed of retrieval.

---

## Part J — Rapid-fire

Short probes an interviewer fires to calibrate breadth. One or two sentences each; know the *why*.

**J1. Why bytes, not characters, for BPE?** Base vocab is 256 byte values, so nothing is ever
"unknown" and every input round-trips; a character vocab is huge (~150k Unicode code points) and still
splits emoji/CJK poorly.

**J2. Why RMSNorm over LayerNorm?** Mean-centering is unnecessary, so dropping it is cheaper with no
quality loss.

**J3. Why pre-norm over post-norm?** It keeps the residual path an identity, making deep stacks
trainable without warmup gymnastics.

**J4. Why is `d_ff ≈ (8/3)·d` for SwiGLU?** SwiGLU uses three weight matrices instead of two, so
shrinking the hidden size to 8/3·d keeps the FFN parameter count matched to a standard 4·d ReLU MLP.

**J5. Why does RoPE rotate q and k but never v?** Position must enter the *score* (the q·k dot product,
which becomes relative-offset-dependent); v is the content being mixed and carries no position.

**J6. Total vs active params in an MoE — which goes in 6ND?** Active. FLOPs track what runs per token,
not what is stored.

**J7. Why does int4 weight-only quantization speed up decode ~4×?** Decode is memory-bound on reading
weights; a quarter the bytes means ~a quarter the weight-read traffic.

**J8. Why is decode memory-bound but prefill compute-bound?** Prefill processes all prompt tokens in
one high-intensity batched matmul; decode reads the whole weight matrix to produce a single token
(intensity near the batch size).

**J9. Why is speculative decoding lossless?** The target verifies drafted tokens in one parallel pass
and a modified rejection rule accepts/corrects them so the output distribution is exactly the target's.

**J10. What determines on-device tokens/sec?** Memory bandwidth ÷ (bytes-per-weight × active
parameters) — bandwidth-bound, which is why quantization is the lever.

**J11. Why is activation quantization harder than weight quantization?** Activations have outlier
channels with huge magnitudes; one scale either clips them or crushes everything else. Weights are
better-behaved.

**J12. Chinchilla's tokens-per-parameter?** ~20. But deployment models overtrain far past it (LLaMA 3
8B ≈ 2000) to cut lifetime inference cost.

**J13. Why does compute-optimal differ from deployment-optimal?** Chinchilla minimizes *training* cost;
serving pays inference on every forward pass forever, so a smaller overtrained model is cheaper overall.

**J14. What's the highest-return data stage?** Deduplication and model-based (classifier) quality
filtering.

**J15. What does the fuzzy-dedup S-curve $1-(1-s^r)^b$ control?** The Jaccard-similarity threshold
above which document pairs become candidates; tune `b`, `r` to place the steep cutoff.

**J16. Why extract from WARC, not WET?** WET's crude pre-extracted text is low quality; re-extracting
main content from raw HTML materially improves the final model.

**J17. Why can't you compare perplexity across model families?** It is per-token and tokenizer-
dependent; different tokenizers cut text into different token counts.

**J18. Why does the multiple-choice scoring protocol matter?** Letter-likelihood vs text-likelihood vs
generate-and-parse, and length normalization, all change rankings on the *same* model.

**J19. Top-p over top-k — why?** Top-p is adaptive: few tokens when the model is confident, many when
unsure, matching the truncation to the distribution's shape.

**J20. When do you turn off repetition penalties?** Code and structured output, which legitimately
repeat tokens (keywords, delimiters, keys).

**J21. Why do LLMs avoid beam search?** Maximizing sequence probability yields bland, degenerate text
on open-ended generation; the mode is not what you want.

**J22. What's an attention sink?** Models park excess softmax mass on the first few tokens; evicting
them in a sliding window corrupts the distribution, so StreamingLLM keeps them.

**J23. What does needle-in-a-haystack test — and not test?** Long-context *retrieval* of one fact; it
does *not* test long-context reasoning or aggregation.

**J24. Why mask the prompt in SFT?** You want capacity spent on generating the answer given the
question, not on generating the question.

**J25. In DPO, what is the implicit reward?** $\beta \cdot \log(\pi/\pi_{\text{ref}})$ — the policy *is* the reward model,
which is how the reward model and RL loop are removed.

**J26. What does GRPO drop versus PPO?** The learned value network; the group's own reward mean/std is
the baseline instead.

**J27. What is the KL leash's job?** To anchor the policy near the SFT reference and prevent reward
hacking; `β` is the main alignment knob.

**J28. Why did MoE routing collapse without balancing?** Selection is self-reinforcing — an expert
picked early trains more, gets better, gets picked more — until a few experts hog all traffic.

**J29. Where do a small model's parameters mostly live?** The embedding + output head (`V×d` each),
which don't shrink with the transformer body — hence small vocab for small models.

**J30. What separates two same-size models today?** Data quality and post-training, not architecture —
architecture has largely converged on the standard pre-norm decoder.

## You can now

- Fire off the *why* behind thirty breadth facts spanning tokenization, MoE, GPUs, scaling laws, data, evaluation, sampling, long context, and alignment in under thirty seconds each.
- Distinguish a "pass" answer (states the mechanism) from a "fail" answer (states the conclusion) on rapid-fire questions.
- Identify which of these facts you fumbled and route yourself back to the full Q&A in Parts A–J for the underlying derivation.

---

## How to practice

Breadth rounds reward *coverage* and *speed of retrieval*, so practice differently from the deep-dive
banks. Take one Part at a time, close the answers, and try to give the two-sentence core of each
question in under thirty seconds — the interviewer is sampling how much of the field you hold and how
fast you can reach it, then following up on whatever you fumble. For the rapid-fire round, drill until
the *why* comes out with the *what*: "top-p over top-k — because it's adaptive to the distribution's
shape" is a pass; "top-p is better" is a fail. When you can answer a Part cold, have someone (or a
model) fire the questions out of order and follow each answer with "why?" one more level down, because
the real breadth interview is not the first question — it is the third follow-up that finds the edge of
what you actually understand. Then go back to the earlier interview banks and notice how these
breadth facts are the foundation the deep questions stand on.
