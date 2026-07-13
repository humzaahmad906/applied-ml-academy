# 05 — Pretraining: Objectives, Data, and Compute

Everything useful a language model knows — grammar, facts, code, the shape of an
argument — arrives during pretraining, the self-supervised phase where the model
predicts held-out text over trillions of tokens. Fine-tuning and post-training
(module [07-post-training.md](07-post-training.md)) only steer a capability that
pretraining already installed; they cannot conjure one that isn't there. So the
questions that decide whether a model is any good are pretraining questions: what
do you make it predict, what do you feed it, and how much compute do you spend.
This lesson covers the three surviving pretraining objectives, why the field
converged on decoder-only, what the data pipeline actually looks like, the
`FLOPs ≈ 6ND` accounting that governs every training budget, and the modern
Llama-3-style recipe. Scaling-law fitting itself is a sibling topic — we summarize
it here and send you to [scaling laws](../language-modeling/08-scaling-laws.md)
for the derivation.

## Three objectives, one idea

Every pretraining objective is the same trick: hide part of the text, make the
model reconstruct it, and let backprop turn "predict the missing tokens" into
"learn how language works." They differ only in *what* gets hidden and *which
tokens can attend to which*.

**Causal language modeling (CLM)** — the GPT objective. Predict the next token
given all previous tokens, left to right. The loss is the mean negative
log-likelihood of each true next token:

$$
\mathcal{L}_{\text{CLM}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

Attention is causally masked (module
[04-transformer-architecture.md](04-transformer-architecture.md)), so position
`t` sees only positions `< t`. The crucial property: **every token is a training
signal**. A sequence of length `T` gives you `T` prediction problems in one
forward pass. That density is why CLM is so compute-efficient.

**Masked language modeling (MLM)** — the BERT objective (Devlin et al., 2018).
Corrupt ~15% of input tokens (replace with `[MASK]`, a random token, or leave
unchanged) and predict the originals, using *bidirectional* attention — every
token sees the whole sequence, left and right:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log p_\theta(x_i \mid x_{\setminus \mathcal{M}})
$$

Bidirectionality is the strength and the weakness. It builds richer
representations of a *fixed* input — great for classification, tagging,
retrieval — but the model can't generate autoregressively, and only the ~15%
masked positions produce a loss, so you extract far less signal per token than
CLM. The BERT lineage is a series of fixes to that recipe. **RoBERTa** (Liu et
al., 2019) showed BERT was badly undertrained: drop the next-sentence-prediction
task, train longer on more data with dynamic masking, and it jumps. **DeBERTa**
(He et al., 2020) adds disentangled attention (content and position handled
separately) and a better mask decoder; **DeBERTa-v3** swaps MLM for
ELECTRA-style replaced-token detection and remains, in 2026, the encoder you
reach for when you want the strongest small classifier or tagger.

**Span corruption** — the T5 objective (Raffel et al., 2019). Mask contiguous
*spans*, replace each with a sentinel token, and have an encoder-decoder
reconstruct the dropped spans as a short target sequence. This unifies every
task as text-to-text: translation, summarization, classification all become
"encode this, decode that." It keeps the bidirectional encoder for input
understanding while adding a decoder for generation.

## Why decoder-only won

For a few years the field ran all three in parallel. By 2026 the frontier is
decoder-only CLM, essentially everywhere, and it's worth knowing why — it's a
common interview probe.

- **Signal density.** CLM learns from every token; MLM from ~15%. Per unit of
  compute, next-token prediction extracts strictly more supervision, and
  pretraining is compute-bound.
- **One model, all tasks, no head-swapping.** A decoder generates, so
  classification, extraction, translation, and dialogue all reduce to "produce
  the right continuation." In-context learning
  ([08-prompting-peft.md](08-prompting-peft.md)) falls out for free — you never
  needed a task-specific head. Encoder-decoder can do this too, but at ~2× the
  parameters for the same capability, since it maintains two stacks.
- **Scaling and inference.** A single stack with a KV cache
  ([12-inference-decoding.md](12-inference-decoding.md)) is simpler to scale,
  shard, and serve than a split encoder-decoder. The engineering ecosystem
  compounded around it.

The honest caveat: bidirectional encoders are *not* obsolete. For pure
understanding tasks at fixed input — classification, NER, retrieval embeddings —
a 100M-parameter DeBERTa still beats a decoder LM of the same size, because
bidirectional context is genuinely more informative when you don't need to
generate. That decision — encoder vs decoder at work — is the whole point of
module [06-transfer-learning-tasks.md](06-transfer-learning-tasks.md). The rule
of thumb: **generate → decoder; represent → encoder.**

## What pretraining data actually is

The romantic version is "train on the internet." The real pipeline is mostly
janitorial, and data quality now separates good models from bad ones more than
architecture does.

The raw source is web crawl — **CommonCrawl**, petabytes of HTML — plus curated
corpora: code (GitHub), books, Wikipedia, arXiv, StackExchange, filtered forum
text. Turning that into training tokens is a filtering funnel that discards the
large majority of bytes:

1. **Extraction and language ID.** Strip HTML to text; keep documents in your
   target languages. Multilingual mixes are deliberate — token budget per
   language is a design choice, and low-resource languages pay a fertility
   penalty (module [03-tokenization.md](03-tokenization.md)).
2. **Quality filtering.** Heuristics (line-length, symbol ratios, boilerplate
   removal à la C4) plus model-based classifiers that score "does this look like
   the kind of text we want." Modern pipelines (FineWeb-Edu, DCLM) lean heavily
   on a small classifier trained to spot educational/high-quality text; this
   single step moves benchmark scores more than most architecture tweaks.
3. **Deduplication.** Near-duplicate documents (MinHash/LSH) and exact
   substring dedup. Duplicated data wastes compute and worsens memorization and
   test-set contamination — the model regurgitates and leaks eval data
   ([10-evaluation.md](10-evaluation.md)).
4. **Decontamination.** Explicitly remove documents overlapping known benchmark
   test sets, or your eval numbers are fiction.
5. **Mixture weighting.** Upweight high-value domains (code, math) beyond their
   natural web frequency. The mix is tuned empirically against downstream
   benchmarks; it is one of the most guarded parts of any frontier recipe.

The takeaway for the job: when a model underperforms, the data pipeline is a more
likely culprit than the model code. The sibling course
([../language-modeling/11-data.md](../language-modeling/11-data.md)) goes deep on
building this pipeline.

## The compute you'll actually budget: FLOPs ≈ 6ND

You must be able to estimate training cost on the back of an envelope. The
standard approximation: training a dense transformer with `N` parameters on `D`
tokens costs

$$
C \approx 6ND \quad \text{FLOPs.}
$$

Where the 6 comes from: a forward pass through the model is ~`2N` FLOPs per token
(each parameter participates in one multiply and one add — a multiply-accumulate
is 2 FLOPs). The backward pass costs roughly twice the forward (you compute
gradients with respect to both activations and weights), so ~`4N`. Forward plus
backward is ~`6N` FLOPs per token, times `D` tokens gives `6ND`. Attention's
`O(T²)` cost is a lower-order term at typical hidden sizes and gets folded into
the approximation.

Make it concrete. An 8B model on 15T tokens:
`6 × 8e9 × 15e12 ≈ 7.2e23` FLOPs. On H100s at ~40% of the ~1e15 bf16
FLOPs/s realized in practice, that's `7.2e23 / (4e14) ≈ 1.8e9` GPU-seconds ≈
~500K GPU-hours ≈ a few weeks on ~1000 GPUs. That single formula tells you the
cluster size and calendar time before you write a line of training code, and it's
the number interviewers expect you to produce cold.

## Chinchilla, at the summary level

Given a fixed compute budget `C ≈ 6ND`, you trade model size `N` against tokens
`D`. Make the model bigger and you must train on fewer tokens; train on more
tokens and the model must be smaller. **Where is the optimum?** Hoffmann et al.
(2022) — "Chinchilla" — fit the loss surface and found the compute-optimal point
scales **both `N` and `D` roughly in proportion to `√C`**, i.e. about **20
tokens per parameter**. Their headline result: a 70B model trained on 1.4T
tokens beat the 175B GPT-3 trained on 300B tokens, at equal compute, because
GPT-3 was badly *undertrained* — too many parameters, too few tokens.

Two practical corrections you must carry into 2026:

- Chinchilla optimizes *training* compute. In production you also pay
  **inference** compute on every request, forever. So teams deliberately
  "overtrain" small models far past 20 tokens/param (Llama 3 8B saw ~15T tokens,
  ~1875 tokens/param) — a slightly suboptimal *training* trade that buys a much
  cheaper, faster model to serve. Knowing *why* the industry ignores Chinchilla's
  literal ratio is a stronger interview answer than reciting it.
- The 20× number is a fitted heuristic, not a law of nature; it shifts with data
  quality and architecture.

The full loss-surface derivation, the `L(N, D)` power law, and how you fit the
coefficients live in [scaling laws](../language-modeling/08-scaling-laws.md) —
that's the deep treatment; this section is the working summary.

## Emergent transfer

The reason any of this matters: a model trained *only* to predict the next token
turns out to do arithmetic, translate, summarize, and follow instructions —
none of which appeared as explicit objectives. Predicting text well enough
requires modeling the processes that generated it. Some capabilities appear to
switch on sharply with scale ("emergence," Wei et al., 2022), though later work
(Schaeffer et al., 2023) showed much of the sharpness is an artifact of
discontinuous metrics like exact-match accuracy; on smoother metrics the gains
are more continuous. Either way, the practical fact stands: **you cannot
fine-tune in a capability the base model lacks.** If GPT-scale pretraining didn't
give the model a skill, a few thousand fine-tuning examples won't either — they
surface and shape it, not create it.

## The modern recipe (Llama-3 style)

Frontier open recipes (Llama 3, Qwen3, DeepSeek-V3) have converged on a
multi-stage pretraining shape:

1. **Main stage — quality-filtered tokens at scale.** Trillions of tokens
   (~15T for Llama 3), heavy on classifier-filtered web text plus code and math,
   at the base context length (e.g. 8K), with the data mixture tuned against
   downstream benchmarks. This is where most of the compute goes.
2. **Annealing / high-quality finish.** In the final few percent of training,
   decay the learning rate to near zero *while* upweighting the highest-quality
   data (curated math, code, instruction-like text). This "annealing" phase
   reliably lifts benchmark scores for cheap — the last tokens the model sees
   matter disproportionately.
3. **Long-context extension.** A dedicated stage that continues training on long
   documents while scaling the RoPE base frequency (position interpolation /
   YaRN, module [12-inference-decoding.md](12-inference-decoding.md)) to stretch
   the usable context from 8K to 128K+. Done late and briefly because long-context
   data is scarce and the `O(T²)` attention cost is punishing.

Only after all of this does post-training (SFT, preference optimization;
[07-post-training.md](07-post-training.md)) turn the base model into an
assistant.

## Which do I use at work?

You will almost never pretrain from scratch — it's a frontier-lab activity. Your
decision is which *pretrained* family to adapt:

- **Encoder (BERT/RoBERTa/DeBERTa-v3).** Fixed-input understanding at low cost:
  classification, NER, retrieval embeddings, rerankers. A 100M-param DeBERTa-v3
  fine-tune beats calling an API LLM on latency, cost, and often accuracy for a
  narrow task. This is still most production NLP —
  [06-transfer-learning-tasks.md](06-transfer-learning-tasks.md).
- **Decoder (Llama/Qwen/GPT/Claude).** Anything generative, few-shot, or
  open-ended: chat, summarization, extraction with reasoning, tool use.
- **Encoder-decoder (T5/BART).** Still a fine, cheap choice for constrained
  seq2seq (fixed-format summarization, translation) when you don't need a
  general assistant, though decoders have eaten most of this territory.

## What interviews ask here

- Why did decoder-only CLM win over BERT-style MLM and T5 span corruption?
  (Signal density — every token trains; unified generation; simpler scaling and
  serving.)
- Derive `FLOPs ≈ 6ND` and use it to estimate GPU-hours for an 8B model on 15T
  tokens. (2N forward + 4N backward per token × D tokens.)
- State Chinchilla's result and why production models deliberately violate it.
  (~20 tokens/param compute-optimal; overtrain small models to cut inference
  cost.)
- What's the difference between MLM and CLM, and when would you still pick an
  encoder in 2026? (Bidirectional vs causal; fixed-input understanding tasks.)
- Walk through the pretraining data pipeline. (Extract → language ID → quality
  filter → dedup → decontaminate → mixture weighting.)
- What is annealing / the high-quality finishing stage and why does it help?

## Where this shows up on the job

- **Model selection.** Choosing encoder vs decoder vs encoder-decoder for a task
  is a weekly decision; getting it right saves 10–100× on serving cost.
- **Budget estimation.** Sizing a continued-pretraining or from-scratch run for
  a domain model means doing the `6ND` math to quote cluster size and cost before
  anyone approves it.
- **Debugging quality.** When a fine-tune underperforms, tracing it to
  base-model capability limits or data-pipeline defects (dedup, contamination)
  rather than blaming the training code.
- **Continued pretraining.** Adapting an open base to a specialized domain
  (legal, biomedical, a low-resource language) reuses every idea here — objective,
  data filtering, and token budget.
