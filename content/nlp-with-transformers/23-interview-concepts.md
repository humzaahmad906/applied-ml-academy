# 23 — Interview Bank: Concepts and Breadth

This is the breadth bank: the conceptual questions asked across the whole surface of a modern NLP/LLM
interview, from word vectors to RLHF to KV caches. Companions are
[Interview Bank: Implementation Drills](24-interview-implementation.md) (coding drills) and
[Interview Bank: Applied NLP System Design](25-interview-applied-design.md) (system design); use this one for the
"explain X" and "why X over Y" rounds that dominate the phone screen and first onsite panel.

Every answer is written to *pass a real loop*: it names the mechanism, states the tradeoff, and puts a
number on it where a number exists. That is the shape interviewers grade. "Attention lets the model focus"
fails; "scaled dot-product with `1/√d` to keep logit variance ~1, O(n²) in length, which is why a 32k
prompt is memory-bound on the KV cache" passes. Read the answer, close it, and reproduce the *reasoning* —
the loop tests reasoning, not recall. It is July 2026, so model names (Llama 3/4, Qwen3, DeepSeek-R1/V3,
GPT-4o and the o-series, Claude, Gemini) are current, and each section links to the module that teaches it.

---

## NLP fundamentals and word vectors

See [The NLP Landscape in 2026](01-nlp-landscape.md) and [Word Vectors: The Representation That Started It All](02-word-vectors.md).

**Q1. What is the distributional hypothesis and why does it matter?** "Know a word by the company it
keeps" (Firth): words in similar contexts have similar meaning. It matters because it turns semantics into
a prediction problem you optimize with gradient descent — word2vec, GloVe, every contextual embedding since.
Tradeoff: distributional similarity conflates related-but-opposite words ("hot"/"cold" share contexts), so
pure co-occurrence embeddings place antonyms close, which is why intrinsic analogy scores mislead.

**Q2. Derive skip-gram with negative sampling.** Skip-gram maximizes `P(context|center)`; full softmax is
O(V) per step, so negative sampling replaces it: for a true pair and `k` negatives, maximize
`log σ(v_c·v_w) + Σ log σ(−v_{n_i}·v_w)`. Gradient w.r.t. center is `(σ(v_c·v_w)−1)v_c + Σ σ(v_{n_i}·v_w)v_{n_i}`
— pull the true context in, push negatives out. `k=5–20` small corpora, `2–5` large; negatives sampled from
the unigram distribution to the `3/4` power to upweight rare words.

**Q3. word2vec vs GloVe.** Nearly the same solution two ways: word2vec is a local online predictor, GloVe
factorizes the global co-occurrence matrix (`w_i·w_j + b ≈ log X_ij`). Levy & Goldberg showed skip-gram
implicitly factorizes shifted PMI, so both are matrix factorization. GloVe trains faster on huge corpora
(counts precomputed); word2vec streams simpler. Both are static — one vector per type — the limit contextual
models fixed.

**Q4. Where do static embeddings still earn their keep?** Retrieval and RecSys: a frozen table gives O(1)
lookup and cheap nearest-neighbor, no transformer forward pass per item. Cold-start recommendation, candidate
generation before a reranker, and fastText subwords for morphologically rich/low-resource languages still
ship near-static vectors. Tradeoff: no context sensitivity — "bank" is one vector — so they lose on
disambiguation-sensitive tasks.

**Q5. How do you measure embedding bias, and what's the catch?** WEAT (Caliskan 2017): differential cosine
association between target sets (career/family) and attribute sets (male/female names), tested against a
permutation null. It reliably surfaces corpus stereotypes. Catch: projecting out a "bias subspace"
(Bolukbasi) mostly hides bias from the metric — Gonen & Goldberg showed the cluster structure survives.
Measurement is easy; removal is not.

---

## Tokenization

See [Tokenization: Turning Text into Model Inputs](03-tokenization.md).

**Q6. Walk through BPE.** Start from bytes/characters, count adjacent symbol pairs, merge the most frequent
into a new token, add to vocab, repeat to the target size (50k–256k); at inference apply merges in learned
order. Greedy and frequency-driven: common sequences ("ing", " the") become single tokens, rare ones stay
fragmented. Tradeoff: deterministic and fast but not probabilistically optimal — Unigram often segments better.

**Q7. BPE vs WordPiece vs Unigram.** BPE merges by raw pair frequency; WordPiece merges by likelihood gain
(`freq(pair)/(freq(a)·freq(b))`), preferring informative merges; Unigram starts huge and prunes tokens that
least hurt a unigram-LM likelihood, keeping a probabilistic model with multiple segmentations. Modern LLMs
use byte-level BPE (Llama, GPT) or Unigram (T5, Gemma); byte-level guarantees no OOV since every byte is
representable.

**Q8. Do all languages cost the same tokens?** No — fertility (tokens/word) is far higher for
underrepresented languages. English ~1.3 tokens/word; the same content in Hindi/Arabic/Burmese can be 2–5x
more with an English-centric tokenizer, so non-English users pay more per call, fit less context, and see
higher latency. Fix: a balanced tokenizer training mixture and larger vocab; tradeoff is a bigger embedding
matrix and softmax.

**Q9. What are glitch tokens?** Tokens carved out by the BPE training corpus (e.g. usernames like
`SolidGoldMagikarp`) that appeared rarely in the *model's* training data, so their embeddings are essentially
untrained and prompting with them produces bizarre output. They exist because tokenizer and model see
different data. Lesson: tokenizer and model are locked together, and early vocab decisions are expensive to
change.

**Q10. Why do LLMs struggle with arithmetic and character tasks?** Tokenization hides structure. "1234" may
be one token or split inconsistently, so the model never sees clean digit positions; "count the r's in
strawberry" is hard because letters are buried in subwords. Fixes: right-to-left digit grouping, single-digit
tokenization, byte-latent/tokenizer-free research. The interview tell: blame the tokenizer, not "the model
can't reason."

---

## Transformer architecture

See [The Transformer Architecture](04-transformer-architecture.md).

**Q11. Explain scaled dot-product attention and the scaling factor.** Soft dictionary lookup:
`softmax(QKᵀ/√d_k)V` — queries score keys by dot product, softmax weights, weighted sum of values. The `√d_k`
divisor exists because a d-dim dot product has variance `d_k`; without it, logits grow with dimension, softmax
saturates near one-hot, gradients vanish. Dividing keeps logit variance ~1. Cost is O(n²·d) — the quadratic
that dominates long-context economics.

**Q12. Why multi-head instead of one big head?** Splitting into `h` heads lets each attend in a different
subspace — syntax, coreference, locality — then concatenate and project. Same total FLOPs as one wide head but
far more expressive, because patterns aren't averaged into one distribution. Tradeoff: more heads means
smaller per-head dim, and beyond a point extra heads add little — the redundancy GQA/MQA exploit at inference.

**Q13. Explain RoPE and why it beat learned positions.** Rotary embeddings rotate Q and K by an angle
proportional to absolute position before the dot product; since the score depends on the *difference* of
angles, RoPE injects *relative* position into the score with zero added params. It beat learned absolute
embeddings because it extrapolates (with interpolation/YaRN), adds no params, and encodes what attention
needs. Tradeoff: raw RoPE still degrades far past training length.

**Q14. Pre-norm vs post-norm.** Post-norm (2017 original) normalizes after the residual add, putting the norm
on the residual path and making deep stacks unstable — needs LR warmup and careful init. Pre-norm normalizes
inside the branch, keeping the residual path a clean identity, so gradients flow and you train 100+ layers
without gymnastics. Every modern LLM is pre-norm + RMSNorm; tradeoff is a slight quality dip at equal depth,
often fixed with a final norm.

**Q15. What does GQA buy you?** Grouped-Query Attention shares one K/V projection across a group of query
heads (e.g. 8 KV heads for 64 query heads) — the middle ground between MHA (`g=h`) and MQA (`g=1`). The win is
inference memory: the KV cache shrinks by the query-to-KV ratio (8 KV heads = 8x smaller cache) with almost no
quality loss. Decisive because at long context the KV cache, not the weights, dominates serving memory.

**Q16. Why did decoder-only win?** Causal-LM decoder-only is the simplest thing that scales: every token is a
training signal, the architecture is uniform, and in-context learning emerges for free. Encoder-decoder (T5)
still edges out fixed transduction (translation/summarization) but doubles complexity; encoders alone (BERT)
win cheap discriminative production NLP. 2026 answer: decoder-only for assistants, encoders for classification/
retrieval, encoder-decoder narrowly for seq2seq.

---

## Pretraining

See [Pretraining: Objectives, Data, and Compute](05-pretraining.md).

**Q17. Causal vs masked vs span-corruption LM — what survived?** Causal (GPT) predicts next token L→R;
masked (BERT) predicts ~15% masked tokens bidirectionally; span corruption (T5) generates masked spans. MLM
gives better understanding-task representations but can't generate and "wastes" the 85% unmasked; causal LM
won for general models because it's a pure generative objective that scales and yields ICL — every position is
a prediction. MLM still wins for embedding/classification encoders.

**Q18. State the compute rule and Chinchilla.** Training FLOPs ≈ `6ND` (~2 forward + 4 backward per token per
param). Chinchilla (Hoffmann 2022): for fixed compute, scale params and tokens together (~20 tokens/param) —
GPT-3-era models were badly undertrained, so compute-optimal means smaller-and-more-data. The 2026 twist:
inference cost pushes *past* optimal — Llama 3 8B on 15T tokens is ~1875 tokens/param — because you serve it
billions of times and want it cheap.

**Q19. What does modern pretraining data look like?** A filtered, deduplicated mixture: web crawl
(CommonCrawl) → quality filtering (classifier-scored, MinHash near-dedup) → deliberate blend of code, math,
books, multilingual (mixture weights strongly shape skills). Llama-3-style recipes stage it: bulk, then a
high-quality annealing stage, then long-context extension. The number: quality filtering can beat 2x the
tokens — garbage web text hits diminishing returns fast.

**Q20. BERT or GPT for a production task?** Discriminative, high-volume (classification/NER/retrieval),
latency/cost-sensitive → fine-tuned encoder (DeBERTa-v3, 100–400M): 10–100x cheaper per inference than an API
LLM, runs on CPU, you control calibration. Open-ended generation, needs world knowledge, or too little labeled
data → decoder LLM. Senior move: quote the cost delta and note many "LLM" problems are secretly classification.

---

## Applied tasks and transfer learning

See [Transfer Learning: The Applied-NLP Workhorse](06-transfer-learning-tasks.md).

**Q21. NER as token classification — the subword gotcha?** Per-token BIO tagging (`B-PER`/`I-PER`/`O`) with a
linear head on encoder representations. Gotcha: the tokenizer splits words into subwords, so align word labels
to subwords — label the first subword, set the rest to `-100` (ignored by loss). Evaluate with span-level F1
(`seqeval`), not token accuracy, which is inflated by easy `O` tokens. Mis-alignment is the most common bug.

**Q22. Bi-encoder vs cross-encoder.** Bi-encoder (SBERT) embeds query and doc independently, so you precompute
doc vectors and do fast nearest-neighbor over millions — but the two never interact, capping accuracy.
Cross-encoder concatenates query+doc through the transformer for a far better score, but it's O(N) forward
passes per query so it can't scale a corpus. Production uses both: bi-encoder retrieves top-100, cross-encoder
reranks to 10.

**Q23. ROUGE/BLEU — what they measure and how they lie.** BLEU (MT) is n-gram precision + brevity penalty;
ROUGE (summarization) is mostly n-gram recall — both surface overlap with references. They lie: a correct
paraphrase with no lexical overlap scores low, a fluent-but-wrong output with overlap scores high; neither
measures faithfulness. chrF is more robust for rich morphology; in 2026, COMET (MT) and LLM-as-judge correlate
far better. Quote n-gram metrics as a cheap regression signal, not truth.

**Q24. When does a 100M encoder beat a frontier API LLM?** Narrow task, high volume. A fine-tuned DeBERTa
often matches/beats zero-shot frontier accuracy at ~1–5ms CPU latency (vs hundreds of ms + a fee), data stays
in your VPC (privacy), and probabilities are calibrated for thresholding. The LLM wins on few-shot
flexibility, unlabelable tasks, and long-tail reasoning. Rule of thumb: above ~10k requests/day on a stable
taxonomy, fine-tune the small model.

**Q25. What is distillation and why distill?** Train a small student to match a large teacher, typically via
KL on temperature-softened logits, which carry "dark knowledge" (relative class probabilities) richer than
hard labels. You distill to cut serving cost/latency while keeping most quality: DistilBERT is ~40% smaller,
~60% faster, ~97% of BERT's GLUE. Tradeoff: the student caps below the teacher and you need teacher
access/logits. Sequence-level distillation (train on teacher generations) is the LLM-era version.

---

## Post-training: SFT, RLHF, DPO

See [Post-Training: Turning a Base Model into an Assistant](07-post-training.md).

**Q26. Base model vs assistant — what does SFT do?** A base model is a next-token predictor; it continues
your prompt, it doesn't obey it. SFT trains on `(instruction, response)` pairs in a chat template, teaching
the *format* and *behavior* of an assistant: follow instructions, use turns, stop when done. SFT teaches style
and format cheaply but can't reliably make the model prefer good over merely-plausible answers — that's what
preference optimization adds.

**Q27. Walk through RLHF with equations.** Three stages: (1) SFT. (2) Train a reward model on preference pairs
via Bradley-Terry: `loss = −log σ(r(x,y_w) − r(x,y_l))`. (3) PPO to maximize `E[r(x,y)] − β·KL(π_θ‖π_ref)` —
the KL anchor stops the policy drifting into reward-hacked gibberish. Tradeoff: PPO is unstable, holds 4
models in memory (policy, ref, RM, value), and is finicky — exactly why DPO exists.

**Q28. Derive DPO's intuition; when over RLHF?** The RLHF objective has a closed-form optimum
`π*(y|x) ∝ π_ref·exp(r/β)`; invert to express reward via the policy, substitute into Bradley-Terry, and the
reward model cancels — you optimize the policy directly on preference pairs with a classification-style loss.
Use DPO for RLHF-quality alignment without an RM or PPO: stable, only policy+ref. RLHF/online RL still edges
it out at the frontier where exploration and an online signal matter.

**Q29. What is reward hacking and how do you catch it?** The policy exploits RM flaws to score high without
being better — runaway length (RMs love verbosity), sycophancy, formatting tricks. Catch it via KL from the
reference (spiking KL + rising reward is the tell), a length-controlled held-out eval, and human-checking
high-reward samples. Mitigate with length penalties, length-controlled win rates (AlpacaEval 2 LC), and a
tight KL anchor.

**Q30. LIMA claimed 1,000 examples suffices — the real lesson?** LIMA (Zhou 2023): a strong base fine-tuned on
~1,000 curated, diverse, high-quality examples matched much larger SFT sets — the "superficial alignment
hypothesis": pretraining installs knowledge, SFT surfaces format and style. The lesson isn't literally 1,000;
it's that SFT data *quality and diversity* dominate quantity — a few thousand pristine examples beat 100k noisy
ones. Capabilities come from pretraining; post-training elicits them.

---

## Prompting and PEFT

See [Adaptation: Prompting and Parameter-Efficient Fine-Tuning](08-prompting-peft.md).

**Q31. What is in-context learning, mechanistically?** Few-shot examples let the model perform an untrained
task with no weight updates. Mechanistically it's implicit task inference — demonstrations locate the task in
the learned distribution and condition generation — and induction heads (copy patterns from earlier context)
are part of the circuitry. Surprise (Min 2022): example *label correctness* matters less than the label space
and format — demos specify task shape, not the mapping.

**Q32. Why is prompt engineering empirical?** Because behavior is a high-dimensional, non-smooth function of
exact tokens: reordering few-shot examples or changing "Answer:" to "A:" can swing accuracy double digits, and
sensitivity differs per model. No theory predicts the winner, so you measure with a small eval set. Tradeoff:
prompt tuning is cheap and instant but brittle and model-specific — a change helping GPT-4o may hurt Llama, so
prompts don't transfer free.

**Q33. Do the LoRA memory math for 7B.** Full FT bf16: ~14 GB weights + ~14 GB grads + ~56 GB Adam ≈ 84 GB
before activations (multi-GPU). LoRA freezes the base and trains a low-rank `ΔW=(α/r)BA`, so grads/optimizer
exist only for ~0.3–0.6% of params: ~15 GB, fits a 24 GB card. QLoRA quantizes the frozen base to 4-bit NF4
(~3.5 GB) + bf16 adapters: ~5 GB, fits an 8 GB card. That collapse is why applied fine-tuning is PEFT.

**Q34. LoRA rank/alpha and the "all-linear" finding.** Rank `r` is adapter capacity (`8–16` sweet spot,
`32–64` heavy); `α` scales the update by `α/r`. Fix the `α/r` ratio (`α=2r`) and tune `r` alone so capacity
doesn't silently change your effective LR. QLoRA's key finding: *which* layers you target beats raising rank —
target all linear layers (attention q/k/v/o **and** MLP gate/up/down). LoRA matches full FT except when the
update genuinely isn't low-rank (large domain shift).

**Q35. Prompt vs RAG vs PEFT vs full FT.** Match the tool to what's missing. Instructions/format → prompt
(cheapest, instant). Knowledge/facts that change → RAG (retrieve at inference, no retrain, citable).
Behavior/style/skill on stable data → PEFT/LoRA (a few GPU-hours, cheap to serve many adapters). Fundamental
capability + large shift + lots of data → full FT. Cost: prompt << RAG ≈ LoRA << full FT. Classic mistake:
fine-tuning to inject facts — that's RAG's job.

---

## RAG and agents

See [RAG and Agents: Grounding Models in the World](09-rag-agents.md).

**Q36. Why RAG instead of fine-tuning knowledge in?** Three reasons: knowledge cutoff (can't know
post-training events), grounding (retrieved passages let you cite and cut hallucination), and private/changing
data (no retrain per update). RAG separates the parametric model (reasoning/language) from the non-parametric
store (facts), so you edit the index, not the weights. Tradeoff: you own a retrieval system that can miss, and
the answer is only as good as the retrieved context.

**Q37. Sparse vs dense retrieval — why hybrid?** Sparse (BM25) matches exact terms with TF-IDF weighting —
great for codes/names, zero training, blind to synonyms. Dense embeds query+docs into a shared space and
matches by cosine — captures meaning, sometimes misses exact terms, needs a trained model. Hybrid fuses both
(Reciprocal Rank Fusion) because their failure modes are complementary. HNSW gives ~O(log n) approximate
nearest-neighbor for the dense side.

**Q38. Three RAG failure modes and fixes.** (1) Retrieval miss — answer not in top-k; fix with hybrid search,
better chunking, a reranker. (2) Lost-in-the-middle (Liu 2023) — models neglect evidence in the middle of long
context; fix by reranking so the best evidence sits at the edges and keeping context tight. (3) Conflicting/
distractor evidence — a near-duplicate derails the model; fix with a cross-encoder reranker and prompting the
model to cite and reconcile.

**Q39. Explain the ReAct loop; where do agents break?** ReAct (Yao 2022) interleaves reasoning and acting: the
model emits a thought, chooses a tool call (function name + JSON args against a schema), observes the result,
loops until it can answer. It breaks because errors compound: at 95% per-step reliability, a 10-step task is
`0.95^10 ≈ 60%` reliable — multiplicative decay is why long-horizon agents are fragile. Mitigate with fewer
steps, verification/reflection, and constrained schemas.

**Q40. What is MCP and why does it matter in 2026?** The Model Context Protocol is an open standard connecting
models to tools and data via a common interface — build a tool ("server") once, use it in any MCP client. It
matters because it turned tool integration from an N×M problem into N+M, and by 2026 it's the default way
production agents reach databases, APIs, and files. Tradeoff: standard-adoption cost and the security surface
of exposing tools to a model.

---

## Evaluation and benchmarking

See [Evaluation: The Skill That Gets You Hired](10-evaluation.md).

**Q41. Define perplexity; what it does and doesn't track.** Exponentiated average negative log-likelihood per
token: `exp(−(1/N)Σ log p(x_i))` — the model's average branching factor, how surprised it is by held-out text
(lower better). It tracks raw LM fit and is a clean pretraining signal, but not instruction-following,
factuality, or usefulness, and it isn't comparable across tokenizers (different vocab/token count). Great
perplexity, bad assistant is possible — hence task benchmarks and judges post-training.

**Q42. Name the benchmark canon.** MMLU / MMLU-Pro: broad multiple-choice knowledge (MMLU saturated, hence
Pro). GSM8K: grade-school math (saturated). MATH: competition math. HumanEval/MBPP: Python by unit tests. GPQA:
Google-proof graduate science. SWE-bench: resolve real GitHub issues (the agentic coding bar). MMMU: multimodal
college reasoning. Senior point: the easy ones are saturated and contaminated, so the field moved to harder,
contamination-resistant, agentic benchmarks.

**Q43. What is contamination and how do you detect it?** Test data leaked into pretraining, so scores reflect
memorization not capability. Detect via n-gram overlap between benchmark and corpus, canary/exact-match
strings, and behavioral tests (score gap on perturbed vs original items, or completing a test item from a
partial prompt). Stance: distrust public leaderboards, weight recent/private held-out sets, build your own
unseen eval. Contamination is *the* reason leaderboards mislead.

**Q44. LLM-as-judge — biases and mitigations.** A strong model scores outputs (pairwise or rubric). Biases:
position (favors first — swap order and average), length (favors longer — length-controlled win rates),
self-preference (prefers own family — different/ensemble judge), verbosity/formatting. Cheap, fast, ~0.8+
correlation with humans when careful, but not ground truth. AlpacaEval 2 LC and Chatbot Arena ELO are the
standard judged comparisons.

**Q45. Build a product eval a team can trust.** Three layers: (1) a frozen, versioned golden set of real
inputs + expected behaviors acting as a regression gate in CI, so changes can't silently regress; (2)
automated per-task metrics (exact match, F1, faithfulness-via-judge) on every change; (3) online metrics
(thumbs, task completion, escalation) for what offline misses. The freeze is essential — adding samples is
fine, removing/rebalancing invalidates history. Eval-driven development is the hiring signal.

---

## Reasoning models

See [Reasoning Models: CoT, Verifiers, and RL with Verifiable Rewards](11-reasoning.md).

**Q46. Why does chain-of-thought improve accuracy?** CoT gives serial compute and externalized working
memory: a single forward pass has fixed depth, but generating intermediate steps as tokens spreads a
multi-step computation across many passes and conditions each step on prior tokens — state offloaded into the
context. That's why it helps multi-step arithmetic/logic and barely helps single-step recall. Tradeoff: more
tokens = more latency/cost, and the visible chain isn't guaranteed to be the true reasoning.

**Q47. Self-consistency — how does it scale?** Sample `k` independent CoT solutions at nonzero temperature and
majority-vote the answer (Wang 2022). It works because wrong paths disagree while correct ones converge, so
voting denoises. Accuracy rises with `k` with diminishing returns — big gains k=1→8, then plateau — so you
trade ~k× inference for a few points. Simplest test-time compute scaling; needs no verifier.

**Q48. Process vs outcome supervision — which is better?** Outcome rewards only the final answer; process
(Lightman 2023, "Let's Verify Step by Step") rewards each reasoning step via a process reward model. Process
gives denser, better-targeted signal — catches right-answer-via-wrong-reasoning, credits partial progress,
trains better verifiers — and beats outcome-only on MATH. Tradeoff: step-level labels are far costlier. Outcome
rewards are cheap and fine when the answer is auto-verifiable.

**Q49. Explain GRPO vs PPO.** Group Relative Policy Optimization (DeepSeek): per prompt, sample a *group* of
completions, score with a (often verifiable/rule-based) reward, and use the group mean as the baseline —
advantage = `(reward − mean)/std`. That removes PPO's separate value/critic network, cutting memory and
complexity at scale. It's the workhorse of RL-with-verifiable-rewards. Tradeoff: needs a reliable reward, and
naive GRPO has length/formatting pathologies DAPO-style fixes address.

**Q50. Summarize the DeepSeek-R1 recipe and the "aha".** R1-Zero: pure RL with verifiable rewards on a base
model — no SFT first — and long CoT plus self-correction ("wait, let me reconsider") *emerges* on its own: the
"aha moment." R1 adds a cold-start SFT stage and a final RL stage to fix readability/language-mixing. Headline:
RL alone can teach reasoning, and you can distill the traces into small dense reasoners. Job takeaway: buy
reasoning tokens only when the task needs multi-step verification.

---

## Inference and decoding

See [Inference and Decoding: Sampling, KV Cache, and Speculative Decoding](12-inference-decoding.md).

**Q51. Greedy, beam, temperature, top-k, top-p, min-p.** Greedy takes the argmax — deterministic, fine short,
repetitive open-ended. Beam keeps b sequences — good for MT, collapses to generic/repetitive on open-ended and
slow, so chat skips it. Temperature scales logits (>1 flattens, <1 sharpens). Top-k samples the k highest;
top-p (nucleus) samples the smallest set with cumulative prob ≥ p, adapting to distribution shape; min-p
thresholds relative to the top token, more robust across contexts. Typical chat: temp 0.7, top-p 0.9.

**Q52. KV cache memory math — why it dominates.** Without a cache, generating token `t` recomputes attention
over all prior tokens (O(n²) total); the KV cache stores each layer's K/V so decode only computes the new query
against cached K/V (O(n)). Per token: `2(K,V) × layers × kv_heads × head_dim × 2 bytes`. For a 70B-class model
at 32k that's ~10 GB *per user* — rivaling the weights, so concurrency is memory-bound on the cache. Hence
GQA/MLA and PagedAttention.

**Q53. Why is speculative decoding lossless?** A small fast draft proposes `k` tokens; the large target
verifies all `k` in one forward pass (cheap since decode is memory-bound) and accepts the longest prefix
consistent with its own distribution via a rejection-sampling rule. Because the acceptance test exactly
preserves the target's distribution, output is identical in distribution — lossless. Speedup depends on
acceptance rate, commonly 2–3x. Tradeoff: two models, and a bad draft gives low acceptance.

**Q54. What breaks at long context; how do you extend it?** RoPE was trained to a max length; far past it,
scores go off-distribution and quality collapses. Position Interpolation scales positions to fit the trained
range; YaRN/NTK-aware do it per-frequency, often with a short fine-tune, to reach 128k+. Cost also grows:
prefill O(n²), KV cache O(n). RAG-vs-long-context: RAG is cheaper and citable for retrieval-shaped needs; long
context wins when the task needs global reasoning over the whole document.

**Q55. TTFT vs tokens/sec — what dominates each?** Prefill (processing the prompt) is compute-bound and sets
time-to-first-token, scaling with prompt length. Decode (one token at a time) is memory-bandwidth-bound — you
re-read weights + KV cache each step — and sets inter-token latency / tokens-per-second. A chat UI optimizes
TTFT (responsiveness); a batch job optimizes throughput. Continuous batching (vLLM/PagedAttention) fills the
memory-bound decode with concurrent requests to raise throughput.

---

## Interpretability

See [Interpretability: Reading What the Model Is Doing](13-interpretability.md).

**Q56. Is attention an explanation?** No — attention shows *where* information was read, not *why* an output
was produced, and Jain & Wallace (2019) found different attention distributions giving the same prediction, so
it isn't faithful. It's a useful debugging *hint* (spotting an induction head copying) but not attribution.
The honest answer distinguishes correlation (attention maps) from causation (ablations, activation patching)
and reaches for causal methods when the question is "why."

**Q57. What are induction heads and why do they matter?** Attention heads implementing "if I saw `A B`
earlier and now see `A`, predict `B`" — pattern completion over context (Olsson 2022). They matter because
they're a concrete, discovered mechanism for in-context learning, and their formation coincides with a phase
change in ICL ability during training. The flagship evidence that transformers contain interpretable circuits,
not just an inscrutable blob.

**Q58. SAEs and superposition in one breath.** Superposition: models pack more features than neurons as
near-orthogonal directions, so neurons are polysemantic (fire for unrelated concepts). Sparse Autoencoders
learn an overcomplete sparse dictionary that decomposes activations into monosemantic *features* you can
interpret and steer — the leading 2026 approach to reading internal concepts. Limit: interpretations are
partial, SAEs have reconstruction error, coverage is far from complete.

**Q59. What is the logit lens?** Project intermediate residual-stream activations through the final unembedding
to see what the model "would predict" if it stopped at that layer — revealing that a prediction often forms
several layers before the end and is refined after. Cheap, no training, a great first probe. Caveat: early
layers aren't really in the output basis, so early readings are unreliable; the tuned lens corrects for that.

---

## Multimodality

See [Multimodality: When the Model Also Sees and Hears](14-multimodality.md).

**Q60. How does a LLaVA-style VLM connect vision to an LLM?** Late fusion via a projector: a frozen vision
encoder (CLIP/SigLIP ViT) makes patch embeddings, a small trained MLP maps them into the LLM's token embedding
space, and the projected image tokens are prepended to text — the LLM treats them like any tokens. Staged
training: align the projector on captions (LLM frozen), then visual-QA instruction-tune. Cheap and modular;
tradeoff — the LLM never natively learned vision, so fine spatial/OCR tasks lag native-multimodal models.

**Q61. Late vs early fusion vs diffusion-hybrid.** Late (LLaVA): separate encoders glued at the token level —
modular, cheap, bolted-on. Early (Chameleon): tokenize images into discrete tokens and train one transformer
on interleaved image+text from scratch — unified and better at mixed generation, but expensive and needs a
good image tokenizer. Transfusion blends autoregressive text with a diffusion objective for high-quality image
generation. 2026 trend is native any-to-any; late fusion still dominates "add vision to my LLM."

**Q62. Why is document AI the killer app, and what's hard?** Enterprises drown in PDFs/forms/scans, so
OCR-free document understanding (layout + text + tables directly) automates high-value back-office work —
invoices, contracts, claims. Hard because it needs 2D *layout* reasoning (tables, multi-column, checkboxes),
robustness to scan noise, and long documents that blow the context window. DocVQA measures it but gaps
real-world mess. See [vlm-guide](../vlm-guide/) for engineering depth.

---

## Risks and safety

See [Risks and Safety: What Can Go Wrong and Who Owns It](15-risks-and-safety.md).

**Q63. Why do models hallucinate, and what reduces it?** They're trained to produce fluent, probable
continuations, not to know what they don't know, and the objective rewards a confident guess over abstention —
confabulation is baked in. Reducers by leverage: grounding + forced citations, teaching abstention/calibration
in post-training, constrained decoding for structured fields. You reduce, not eliminate — hallucination is
intrinsic to next-token generation, so mitigation is grounding and verification, not a cure.

**Q64. Explain prompt injection; why unsolved?** An attacker puts instructions in *data* the model consumes (a
web page, retrieved doc, email), and the model can't reliably separate trusted developer instructions from
untrusted content in one token stream, so it follows the injected one. Unsolved because there's no hard
instruction/data boundary in a single stream, and it turns dangerous once agents have tools. Mitigate with
input filtering, privilege separation, human-in-the-loop — see [ai-security](../ai-security/). Treat retrieved
content as untrusted.

**Q65. LLM-specific privacy risks.** Memorization: models regurgitate verbatim training text including PII, and
Carlini et al. recovered real sequences — bigger models and duplicated data memorize more. Membership
inference tests whether a record was in training. RAG adds a leak surface (another tenant's docs). Mitigate
with dedup, PII scrubbing, differential privacy (at a quality cost), and strict tenant isolation. The number:
dedup dramatically cuts memorization, which is why every modern recipe dedups.

**Q66. 2026 governance reality for an engineer.** The EU AI Act is phasing in obligations by risk tier
(transparency, documentation; for GPAI models, technical docs and training-data summaries), so "just ship it"
isn't the default. Practically you own a model card (capabilities, evals, limits), respect licenses ("open
weights" like Llama's community license is *not* OSI open source — check field-of-use/scale clauses), and log
for auditability. Treat licenses and documentation as part of the deliverable.

---

## Rapid-fire (one line each)

**R1. Why √d_k in attention?** Keep logit variance ~1 so softmax doesn't saturate and gradients don't vanish.
**R2. Encoder or decoder for retrieval embeddings?** Encoder — bidirectional context, mean/CLS-pooled.
**R3. Training FLOPs-per-token rule?** ~6N (N = parameters).
**R4. Chinchilla tokens-per-param?** ~20; modern models overtrain past it for cheap inference.
**R5. What does GQA reduce?** KV cache size, by the query-to-KV-head ratio.
**R6. RoPE: absolute or relative position?** Relative, via the rotation-angle difference in the dot product.
**R7. RMSNorm vs LayerNorm?** RMSNorm drops mean-centering — cheaper, works as well.
**R8. Default LoRA targets?** All linear layers (attention + MLP); matters more than raising rank.
**R9. DPO removes which RLHF part?** The separate reward model and the PPO loop.
**R10. GRPO's baseline?** The group mean reward — no value/critic network.
**R11. Is speculative decoding lossy?** No — rejection sampling preserves the target distribution exactly.
**R12. Beam search for chatbots?** No — generic, repetitive text on open-ended generation.
**R13. Fastest way to cut hallucination?** Ground with retrieval and force citations.
**R14. Bi- vs cross-encoder roles?** Bi-encoder retrieves fast; cross-encoder reranks accurately.
**R15. What is lost-in-the-middle?** Long-context models neglect evidence in the middle of the prompt.
**R16. Perplexity comparable across tokenizers?** No — different vocab, different token counts.
**R17. Metric for NER?** Span-level F1 (seqeval), not token accuracy.
**R18. What does SFT teach that pretraining doesn't?** Format and instruction-following, not new knowledge.
**R19. Prefill vs decode bottleneck?** Prefill compute-bound (TTFT); decode memory-bandwidth-bound (tok/s).
**R20. Is attention a faithful explanation?** No — where info was read, not why; use causal methods.

---

## Explain it to a PM (no jargon)

**P1. Why does the same request cost more in some languages?** The model reads text in chunks and bills per
chunk. Its "alphabet" of chunks was built mostly from English, so English breaks into few chunks while Hindi or
Arabic breaks into 3–5x more — same meaning, more chunks, so it costs more, fits less per request, and runs
slower. If non-English users matter, budget for it or pick a model with a more balanced vocabulary.

**P2. Why does it confidently make things up?** It's built to produce the most *plausible-sounding* next words,
not to check facts — like a fluent person answering off the top of their head. When it doesn't know, it guesses
smoothly instead of saying "I'm not sure." The reliable fix isn't "make it smarter"; it's feeding it the actual
source documents and requiring it to cite them, so answers are grounded in something we can verify.

**P3. Fine-tuning vs RAG in plain terms?** Fine-tuning changes the model's *habits* — tone, format, behavior.
RAG changes what it *knows right now* by handing it the relevant documents at question time. If our facts change
weekly (prices, policies, docs), that's RAG — update the library, no retraining. For a consistent voice or a
specialized skill, that's fine-tuning. Injecting fresh facts by fine-tuning is the common, expensive mistake.

**P4. Why not trust benchmark scores in the press release?** Benchmarks are exams, and some models effectively
saw the answers during training — the questions leaked into their study material, so a high score can mean good
memory, not real skill. The only scores to trust for our product come from a test set we built and keep
private. Ask "was this measured on our data or a public leaderboard?" before believing a number.

**P5. Why is a tool-using agent riskier than a chatbot?** A chatbot only talks; an agent can *act* — click,
send, query, run code. Two problems compound: small mistakes multiply (95% reliable per step is only ~60% over
ten steps), and if it reads a document with hidden malicious instructions it may follow them — and now it has
hands. Agents doing anything sensitive need guardrails and a human checkpoint, not blind autonomy.

---

## How to practice these

Cover the answer, read only the question, and say your answer *out loud* — the loop grades whether you can
narrate mechanism → tradeoff → number under mild pressure, and silent recognition hides the gaps speaking
exposes. When you fumble one, trace it to its module ([04](04-transformer-architecture.md) for architecture,
[07](07-post-training.md) for alignment, [12](12-inference-decoding.md) for serving) and re-derive the number
rather than memorize it. The interviewer pushes one level deeper than the bank — "why √d and not d?", "the KV
cache at 128k?" — so for every answer, prepare the one follow-up you'd ask from the other side of the table.
