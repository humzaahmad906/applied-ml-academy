# 16 — The Modern NLP Stack: Tools, Models, and a 90-Day Plan

Every module in this course taught a mechanism. This one attaches the 2026 tool to each
mechanism, tells you which model to reach for and when, gives you the names to drop per
interview topic, and lays out a 90-day plan to convert all of it into a portfolio. The
skill an interviewer is actually probing is not "do you know what attention is" — it's
"when you hit this problem at work, do you know the three tools people use, which one you
pick, and why." That mapping is the difference between someone who read the papers and
someone who ships.

Treat this as a reference you keep open while you read the other modules, not a chapter
you save for last.

## The stack, mapped to each module

The modern applied-NLP toolchain is small and consolidated. Here is what owns each layer,
tied back to the module that teaches the concept.

**Representation — [word vectors](02-word-vectors.md), [tokenization](03-tokenization.md).**
`tokenizers` (the Rust-backed HuggingFace library) trains and runs BPE / WordPiece /
Unigram; `sentencepiece` is the other standard, still common in Google-lineage models.
For static embeddings and sentence embeddings, `sentence-transformers` is the default —
`all-MiniLM-L6-v2` for a fast 384-dim baseline, `bge`/`gte`/`e5` families when you need
stronger retrieval quality. `gensim` still exists for classic word2vec/GloVe but you'll
rarely train those in production.

**Architecture and pretraining — [transformer](04-transformer-architecture.md), [pretraining](05-pretraining.md).**
`transformers` is the hub of everything: model definitions, `AutoModel`/`AutoTokenizer`,
generation. `datasets` handles streamed, memory-mapped corpora. Actual large-scale
pretraining is a systems problem (FSDP/DeepSpeed, Megatron-style parallelism) and lives in
the `language-modeling` sibling course — reference it, don't re-implement it on a T4.

**Fine-tuning and post-training — [transfer learning](06-transfer-learning-tasks.md), [post-training](07-post-training.md), [prompting/PEFT](08-prompting-peft.md).**
`transformers` `Trainer` for encoder fine-tuning; `TRL` for `SFTTrainer`, `DPOTrainer`,
`GRPOTrainer`; `PEFT` for LoRA/QLoRA adapters; `bitsandbytes` for 4-bit quantized bases.
`Unsloth` is the fast single-GPU path (custom Triton kernels, ~2× speed, less VRAM);
`Axolotl` is the YAML-reproducible team path. The run mechanics at length are the
`fine-tuning-llms` sibling course's job.

**Retrieval and agents — [RAG and agents](09-rag-agents.md).**
Vector search: `FAISS` (in-process, the workhorse for labs and mid-scale), or a managed
store (Qdrant, Weaviate, pgvector, LanceDB) in production. Sparse: `bm25s`/Elasticsearch.
Reranking: `sentence-transformers` cross-encoders (`ms-marco-MiniLM`) or a hosted reranker.
Orchestration: `LangChain` and `LlamaIndex` are ubiquitous but weigh their abstraction cost
— for anything you need to debug under load, a few hundred lines of direct code often beats
the framework, and interviewers respect the candidate who knows *when* the framework earns
its complexity. `MCP` (Model Context Protocol) is the 2026 standard for wiring tools to
models; know what it is. The deep RAG/agent engineering depth is the `vlm-guide` course.

**Evaluation — [evaluation](10-evaluation.md).**
`lm-evaluation-harness` (EleutherAI) is the canonical academic-benchmark runner;
`inspect` (UK AISI) is the modern framework for structured, agentic, and safety evals.
For LLM-as-judge product evals you'll often write your own harness — `ragas` for RAG-
specific metrics is a reasonable starting point. The point from module 10 stands: your
golden set and regression gate matter more than any leaderboard.

**Reasoning and inference — [reasoning](11-reasoning.md), [inference/decoding](12-inference-decoding.md).**
Serving: `vLLM` (PagedAttention, continuous batching) is the default high-throughput
server; `SGLang` is the strong alternative, especially for structured/agentic workloads
with prefix caching. Edge/local: `llama.cpp` (GGUF, CPU/Metal) and `MLX` on Apple silicon;
`Ollama` wraps llama.cpp for a one-command local experience. Structured decoding:
`outlines`/`xgrammar` for grammar- and schema-constrained generation.

**Interpretability, multimodality, risk — [interp](13-interpretability.md), [multimodal](14-multimodality.md), [risk](15-risks-and-safety.md).**
`TransformerLens` and `nnsight` for mechanistic work; `SAELens` for sparse autoencoders.
Multimodal models load through `transformers` (`AutoModelForVision2Seq`) or their own repos.
For risk/governance you're producing artifacts, not running a library: model cards,
eval reports, license audits. Prompt-injection and jailbreak depth is the `ai-security`
course.

**Observability (cross-cutting).**
Once anything is in production you need traces and a cost dashboard: `Langfuse`,
`Arize Phoenix`, or a vendor's native tracing. Cost-per-request and p95 latency are the
numbers your manager asks about, so instrument them from day one.

## The model landscape: open vs API

Two families, and the interview question is always "which and why."

**Open-weight.** You download the weights and run them yourself.
- **Llama 3 / 4** (Meta) — the ecosystem default; everything supports it. Community-license,
  not OSI-open.
- **Qwen3** (Alibaba) — strong across sizes, excellent multilingual, permissive Apache-2.0
  for most sizes; the current go-to for a self-hosted assistant.
- **DeepSeek-V3 / R1** — a large MoE base and its reasoning sibling; R1 popularized open
  reasoning via RLVR (see [reasoning](11-reasoning.md)).
- **Mistral / Mixtral** — efficient dense and MoE models, Apache-2.0 lineage.
- **Gemma** (Google) — small, strong, good on-device story.
- **OLMo** (AI2) — genuinely open: weights *and* data *and* training code. The one to cite
  when someone conflates "open weight" with "open source."
- Small end for labs and edge: `Qwen2.5-0.5B`, `SmolLM2`, `Gemma-2-2B`.

**API / closed.** You call an endpoint; you never see the weights.
- **GPT-4o and the o-series** (OpenAI) — the o-series being the reasoning line.
- **Claude** (Anthropic) — strong long-context, coding, and tool use.
- **Gemini** (Google) — very long context and native multimodality.

## The selection playbook

Pick along five axes; the honest answer in an interview names the tradeoff, not a favorite.

1. **Privacy and data residency.** If data can't leave your infrastructure (health, legal,
   regulated enterprise), open weights self-hosted is the only option. This single
   constraint decides many real deployments before quality even enters.
2. **Cost at volume.** APIs are cheapest at low/spiky volume — no idle GPU. Past a steady
   throughput threshold, self-hosting an open model on `vLLM` wins on cost per token, and a
   fine-tuned small open model can undercut a frontier API by 10–100× on a narrow task. Do
   the arithmetic: tokens/day × price/token vs GPU-hour × utilization.
3. **Latency and control.** Self-hosting lets you tune batching, quantization, and
   speculative decoding for your p95; APIs give you their latency and their rate limits.
4. **Capability ceiling.** For open-ended frontier reasoning, the closed flagships still
   lead. For a bounded task — classification, extraction, domain QA — a fine-tuned open 7B
   (or a 300M encoder, per [transfer learning](06-transfer-learning-tasks.md)) often
   matches or beats an API LLM at a fraction of the cost, with better calibration.
5. **Licensing.** "Open weight" ≠ "open source" (module 15). Check the actual license for
   commercial use, output-training clauses, and monthly-active-user caps before you build a
   product on it. Apache-2.0 (Qwen, Mistral) is the safe default; Llama's community license
   has conditions; verify per model.

The default decision path most teams follow: **prototype on an API** (fastest to a working
demo), **measure** with a real eval, then **migrate the settled task to a fine-tuned open
model** if privacy, cost, or latency demands it. Start expensive and flexible, end cheap
and fixed.

## What to name-drop per interview topic

Interviewers listen for the current, specific reference. One good name beats three vague
ones. Per topic:

- **Tokenization** — BPE (Sennrich 2016), byte-level BPE (GPT-2), SentencePiece Unigram;
  fertility and the multilingual tax.
- **Architecture** — scaled dot-product and √d (Vaswani 2017), RoPE (Su 2021), GQA
  (Ainslie 2023), RMSNorm, FlashAttention (Dao 2022), SwiGLU.
- **Pretraining** — Chinchilla compute-optimal (Hoffmann 2022), FLOPs ≈ 6ND, decoder-only
  won; RoBERTa/DeBERTa for the encoder lineage.
- **Transfer / production NLP** — SBERT (Reimers 2019), bi- vs cross-encoder, DeBERTa-v3,
  distillation (DistilBERT), seqeval for NER.
- **Post-training** — InstructGPT/RLHF (Ouyang 2022), Bradley-Terry reward, DPO (Rafailov
  2023), LIMA ("quality > quantity"), Tülu/OLMo open recipes.
- **Prompting/PEFT** — in-context learning (Brown 2020), LoRA (Hu 2021), QLoRA (Dettmers
  2023), "LoRA without regret."
- **RAG/agents** — BM25, dense retrieval + HNSW, lost-in-the-middle (Liu 2023), ReAct (Yao
  2022), MCP.
- **Evaluation** — MMLU/GSM8K/HumanEval/SWE-bench/GPQA, contamination, LLM-as-judge biases,
  Chatbot Arena ELO.
- **Reasoning** — self-consistency (Wang 2022), process supervision (Lightman 2023), GRPO,
  DeepSeek-R1 and R1-Zero, test-time compute scaling.
- **Inference** — KV cache, PagedAttention/vLLM (Kwon 2023), speculative decoding
  (Leviathan 2023), YaRN for context extension.
- **Interpretability** — induction heads (Olsson 2022), logit lens, superposition and SAEs
  (Anthropic 2023–24), "attention is not explanation."
- **Multimodality** — LLaVA, CLIP-style encoders, Chameleon (early fusion), MMMU.

Say the mechanism, then the name, then the tradeoff. Names alone read as memorization;
names attached to a mechanism read as understanding.

## A 90-day portfolio plan

Ninety days, working alongside a job, turns this course into evidence you can point at. The
capstone ([26-capstone.md](26-capstone.md)) is the centerpiece; the rest is scaffolding.

**Days 1–30 — foundations and first artifacts.** Work modules 01–06 and do
[lab 17](17-lab-embeddings-tokenizers.md) and [lab 18](18-lab-transformer-from-scratch.md).
Ship two small public repos: a from-scratch skip-gram + tokenizer comparison, and a
from-scratch transformer block trained on TinyStories with attention visualizations. These
prove you understand the internals, which is what junior/mid loops probe hardest.

**Days 31–60 — adaptation and retrieval.** Work modules 07–10 and do
[lab 19](19-lab-finetune-encoder.md), [lab 20](20-lab-sft-dpo.md), and
[lab 21](21-lab-rag-eval.md). Ship a fine-tuned encoder with a real latency-vs-API cost
table, and a RAG system with a golden-set eval and a documented failure-mode fix. This is
the exact shape of most applied-NLP work, so it's the most persuasive part of the portfolio.

**Days 61–90 — the capstone and interview prep.** Build the Production Support Intelligence
System from [module 26](26-capstone.md): intent classifier + entity extraction + RAG
answerer + eval harness + cost/latency budget + model card. In parallel, drill
[bank 23](23-interview-concepts.md), [bank 24](24-interview-implementation.md), and
[bank 25](25-interview-applied-design.md) — a set of questions each evening, closed-book.
Finish with a README, a short demo video, and an eval report. The capstone plus two
supporting repos is a portfolio that survives a senior loop's follow-up questions.

Throughout, write up each artifact as you go — a paragraph on what you built, the number
you measured, and the tradeoff you chose. That writeup is what you paste into the "tell me
about a project" answer, and it's the difference between "I did RAG" and "I built RAG,
measured hit@5 at 0.82, found lost-in-the-middle degrading answers, and fixed it with a
cross-encoder rerank that cost 40 ms."

## What interviews ask here

- "You have a text-classification task at 10M requests/day — API LLM or self-hosted?" —
  do the cost arithmetic, name latency/privacy/calibration, land on a fine-tuned small
  encoder.
- "When would you *not* use LangChain/LlamaIndex?" — when you need to debug under load or
  control the pipeline; frameworks earn their complexity only past a certain size.
- "What's the difference between open weight and open source?" — weights vs weights + data
  + code + a real license; cite OLMo vs Llama's community license.
- "Which serving stack and why?" — vLLM/SGLang for throughput, llama.cpp/MLX for edge; name
  PagedAttention and continuous batching.
- "How do you pick between Qwen, Llama, and an API model for a new product?" — walk the
  five-axis playbook: privacy, cost-at-volume, latency, capability ceiling, license.
- "Walk me through a project you shipped." — the 90-day artifacts, each with a measured
  number and a named tradeoff.

## Where this shows up on the job

- **Every build starts with a make-or-buy decision** — API vs open, framework vs direct
  code — and being fluent in the tradeoff axes is what makes you the person who scopes the
  project rather than just implements it.
- **Cost and latency dashboards are permanent** — once something serves traffic, tokens/day
  × price and p95 latency are recurring conversations with your manager, so the tooling
  choices you make on day one follow you.
- **Model migration is routine** — teams prototype on an API and migrate settled tasks to
  fine-tuned open models; owning that migration (eval parity, cost proof, rollout) is
  high-visibility, promotable work.
- **The portfolio is the interview** — the repos and capstone you build here are the
  concrete answer to "have you actually done this," which is the question that separates
  offers from rejections.
