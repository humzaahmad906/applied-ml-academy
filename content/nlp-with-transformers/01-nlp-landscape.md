# 01 — The NLP Landscape in 2026

Natural language processing stopped being a collection of specialized subfields and became, for most
practical purposes, one thing: predict the next token well enough and a startling range of language
tasks fall out of the same model. That collapse is recent and it is not total — plenty of production
NLP in 2026 still runs on fine-tuned 100M-parameter encoders, not a frontier LLM — but it reframes
everything. This module gives you the map: what the tasks actually are, how the field arrived here
from rule systems through statistical and recurrent models, why the transformer won, and where the
jobs sit. You need this map before the mechanisms, because interviewers test whether you understand
the *shape* of the field, not just whether you can quote an attention equation.

## What NLP is, concretely

NLP is the engineering of systems that read, transform, or produce human language. The reason it is
hard, and the reason it resisted clean solutions for decades, is that language is **discrete,
compositional, ambiguous, and long-range dependent** all at once. "The trophy didn't fit in the
suitcase because it was too big" — *it* is the trophy; flip "big" to "small" and *it* becomes the
suitcase. No amount of local pattern matching resolves that; you need something like world knowledge
and coreference. Language also has a heavy tail: a handful of words cover most tokens (Zipf's law),
but the long tail of rare words, names, and novel compounds never stops, which is why fixed
vocabularies fail and [subword tokenization](03-tokenization.md) exists.

Everything downstream is a way of coping with those four properties. Word vectors cope with discreteness
by embedding tokens in a continuous space. Attention copes with long-range dependence by making every
position reachable in one step. Pretraining copes with ambiguity by absorbing enough text that world
knowledge is implicit in the weights.

## The task taxonomy

Interviewers expect you to slot any product request into a small taxonomy and name the right
architecture and metric for each. Memorize this table; it is the vocabulary of the field.

- **Text classification** — one label (or a small set) per document. Sentiment, spam, intent, topic,
  toxicity. Metric: accuracy / macro-F1; watch class imbalance. The workhorse of applied NLP.
- **Sequence labeling (NER, POS, chunking)** — one label per token. Named-entity recognition tags
  spans (person, org, location, product) using **BIO** encoding. Metric: entity-level span F1
  (`seqeval`), not token accuracy — a model can get 97% of tokens right and miss half the entities.
- **Extractive QA** — given a question and a passage, return the answer *span*. Metric: exact match
  and token-F1. SQuAD-style. Contrast with **generative/abstractive QA**, where the model writes a
  new answer and you evaluate it very differently.
- **Summarization** — compress a document. Extractive (pick sentences) vs abstractive (generate).
  Metric: ROUGE, BLEU, chrF — all of which correlate weakly with human judgment (module
  [06](06-transfer-learning-tasks.md) is honest about their lies).
- **Machine translation (MT)** — map one language to another. The task that drove the encoder-decoder
  transformer into existence. Metric: BLEU / chrF / COMET.
- **Dialogue / assistants** — multi-turn, open-ended generation conditioned on a system prompt and
  history. No single ground truth, so evaluation shifts to preference and [LLM-as-judge](10-evaluation.md).
- **Retrieval / semantic search** — rank documents by relevance to a query. Powered by sentence
  embeddings; the foundation of [RAG](09-rag-agents.md). Metric: recall@k, MRR, nDCG.

Two orthogonal axes cut across all of these. **Understanding vs generation**: classification and NER
*read*; summarization and dialogue *write*. **Span-level vs sequence-level**: does the label attach to
a token, a span, or the whole document? Those two questions plus the metric determine your architecture
before you write a line of code.

## How the field got here (the compressed arc)

**Rules (1950s–1980s).** Hand-written grammars, finite-state transducers, expert systems. Precise,
interpretable, and brittle — every exception is a new rule, and language is nothing but exceptions.
These systems never scaled past narrow domains, but their descendants survive: regexes, `spaCy`
rule matchers, and grammar-constrained decoding all live in production today.

**Statistical (1990s–2000s).** The move that unlocked progress was giving up on hand-coded meaning and
counting instead. n-gram language models estimate $P(w_t \mid w_{t-1}, \dots, w_{t-n+1})$ from corpus
frequencies. Hidden Markov Models and, later, Conditional Random Fields (Lafferty et al., 2001) gave
strong sequence labeling. IBM's alignment models powered a decade of statistical MT. The ceiling: the
Markov assumption throws away everything beyond a short window, and data sparsity means most n-grams
are never seen, forcing elaborate smoothing.

**Neural / recurrent (2013–2017).** Two ideas broke the statistical ceiling. First, **distributed
representations** — [word vectors](02-word-vectors.md) — replaced sparse one-hot tokens with dense
vectors where similar words sit near each other, so a model could generalize from "dog" to "cat"
without seeing every combination. Second, **recurrent networks** (LSTM, GRU) processed sequences step
by step, carrying a hidden state, and in principle remembered arbitrary history. Seq2seq with attention
(Bahdanau et al., 2014; Sutskever et al., 2014) drove MT and summarization and defined the era.

**Transformer (2017–now).** "Attention Is All You Need" (Vaswani et al., 2017) threw out recurrence
entirely and kept only attention. The reasons it won are worth stating precisely, because it is the
single most common interview question in the field.

## Why RNNs and LSTMs lost

Three failures, in order of importance.

**Parallelism.** An RNN's hidden state at step $t$ depends on step $t-1$, so the forward pass is
inherently sequential — you cannot compute position 500 until you have computed 499. On modern
accelerators, which are enormous parallel matrix machines, that is fatal. A transformer computes all
positions' representations simultaneously; every token attends to every other in one batched matmul.
This is *the* reason transformers scaled: not that they are smarter per parameter, but that you can
train them on far more data in the same wall-clock time. Parallelism over the sequence turned compute
and data into the bottleneck instead of the architecture, and that is what scaling laws (module
[05](05-pretraining.md)) then exploited.

**Long-range credit assignment.** LSTMs were designed to fight vanishing gradients, and they help, but
information between two distant tokens still has to survive a long chain of gated updates. Signal
decays; gradients over hundreds of steps get noisy. Attention makes the path length between any two
positions **constant** — one hop — so long-range dependencies are learned as easily as short ones. The
cost is $O(n^2)$ compute in sequence length (module [04](04-transformer-architecture.md) and
[12](12-inference-decoding.md) deal with this at length), a tradeoff the field happily accepts.

**Transfer.** The deepest reason. RNN-era NLP mostly trained a fresh model per task from a modest
labeled set. Transformers made **pretrain-once, adapt-many** practical: train on a mountain of raw text
with a self-supervised objective, then fine-tune or prompt for each task. ELMo and ULMFiT showed the
promise; BERT and GPT made it the default. Transfer is why a 2026 team ships a strong classifier in an
afternoon instead of collecting 50k labels. RNNs could transfer in principle, but the architecture
never made large-scale pretraining cheap enough to matter.

Recurrence is not dead — state-space models (Mamba) and linear-attention hybrids are an active research
line precisely because $O(n^2)$ hurts at long context — but for the working NLP engineer in 2026, the
transformer is the substrate, and the recurrent era is background you should be able to explain, not
tooling you reach for.

## Two production eras: pipeline vs LLM

The other axis you must hold in your head is *how systems are built*, and it split into two eras that
now coexist.

**The pipeline era (still ~half of production NLP).** Decompose a problem into stages, each a
specialized model: tokenize → embed → a fine-tuned encoder for classification or NER → business logic.
These systems are cheap (a DeBERTa-small runs thousands of docs/sec on one GPU), fast (single-digit
milliseconds), calibrated, private (runs in your VPC), and debuggable. When the task is narrow and
high-volume — ticket routing, PII detection, content moderation, extraction — a fine-tuned 100M
encoder often *beats* a frontier API on latency, cost, and sometimes accuracy. Module
[06](06-transfer-learning-tasks.md) puts real numbers on this.

**The LLM era.** One large generalist model, adapted by [prompting](08-prompting-peft.md),
[retrieval](09-rag-agents.md), or [light fine-tuning](08-prompting-peft.md), handles open-ended tasks
a pipeline never could: multi-step reasoning, flexible generation, tasks you never collected data for.
The cost is latency, price per token, weaker calibration, and less control. This is where net-new
capability lives, and where most 2026 hiring is concentrated.

The senior signal in interviews is refusing the false choice. Real systems are **hybrid**: a cheap
encoder classifies and routes, escalating only the hard cases to an LLM; RAG grounds an LLM in your
private docs; a small fine-tuned model handles the 90% and an API absorbs the tail. "Which is better"
is the junior question; "which per stage, and what's the cost/latency/quality budget" is the senior one.

Make it concrete. A support-ticket system (the shape of this course's [capstone](26-capstone.md)) might
run a fine-tuned DeBERTa intent classifier at ~5 ms/ticket to route 95% of traffic to canned flows,
call a token-classification model to pull out order IDs and product names, and only for the residual
"other / complex" bucket assemble a RAG prompt over the product docs and pay for an LLM call. The
expensive generalist touches a small fraction of volume; the cheap specialists carry the load. Costing
that split — how many tickets hit the LLM, at what price, versus the accuracy you'd lose by routing more
to the encoder — is exactly the tradeoff an applied interview will push you to reason about out loud.

## Where the jobs are

Three broad tracks, with real differences in day-to-day work and what the loop screens for.

- **Applied NLP / ML engineer.** Ship language features: fine-tune encoders, build retrieval and RAG,
  wire evals, own latency and cost. The largest bucket. Interviews test the task taxonomy above,
  fine-tuning mechanics, [evaluation](10-evaluation.md), and system design. This course is aimed
  squarely here.
- **LLM / infra engineer.** Serve and optimize large models: [inference](12-inference-decoding.md),
  KV cache, quantization, throughput, training systems. Interviews go deep on GPU systems and
  parallelism — the sibling `language-modeling` course owns that depth.
- **Research / research engineer.** Push the frontier: new objectives, [post-training](07-post-training.md)
  methods, [reasoning](11-reasoning.md), [interpretability](13-interpretability.md). Interviews expect
  paper fluency and the ability to design an experiment and eval.

Most job titles blur these, and the boundaries move every quarter. The durable advice: be excellent at
the applied core — tasks, representations, adaptation, evaluation — and read enough research to know
what is real. That combination is what the following modules build.

## What interviews ask here

- Why did transformers replace RNNs? (Parallelism first, then constant-path long-range credit, then
  cheap large-scale pretraining and transfer.)
- Given task X, what architecture and metric? (Map it: token-level → sequence labeling + span F1;
  doc-level → classification + macro-F1; open-ended → generation + judge/preference.)
- When would you *not* use an LLM? (High-volume narrow tasks where a fine-tuned encoder wins on
  latency, cost, calibration, privacy — bring numbers.)
- Walk the arc from n-grams to transformers and name the ceiling each stage hit.
- What is the distributional hypothesis and why does it matter for representation learning?
- Pipeline vs LLM system design for a given product, with a cost/latency budget.

## Where this shows up on the job

- **Scoping.** The first thing you do with any language request is slot it into the task taxonomy and
  decide pipeline vs LLM vs hybrid — that decision sets your data, model, metric, and budget.
- **Cost/latency architecture.** Routing cheap cases to a small model and escalating the tail to an LLM
  is the single most common way teams cut inference spend without losing quality.
- **Talking to non-ML stakeholders.** You constantly translate a product ask ("understand our support
  tickets") into concrete tasks (intent classification + entity extraction + retrieval) with distinct
  models, metrics, and SLAs — the framing in this module is that translation.
