# NLP with Transformers — A Job-Focused Specialization

Natural language processing after the recurrent era, taught for people who want to
build and ship it — and pass the interview loop that gates those jobs. The arc is the
whole modern stack: word vectors and tokenization as the durable basics, the
transformer in depth, pretraining and post-training, adaptation by prompting and PEFT,
retrieval and agents, evaluation, reasoning models, inference and decoding,
interpretability, multimodality, and the risk/safety surface you are accountable for on
the job. Then six hands-on labs on free Colab, three interview banks, and a portfolio
capstone.

This is the *NLP* lens on large models: language as data, the task taxonomy,
representation, adaptation, evaluation, and the map from research paper to production
system. Where a topic is really a systems or engineering topic, this course teaches the
concept and points you at the sibling course that owns the depth.

## Who this is for

You already write PyTorch, you know backprop, and you have seen RNNs and LSTMs. This
course assumes recurrent nets — it does not teach them. They get a short "why they lost"
treatment (parallelism, long-range credit assignment, transfer) and then we move on.
What this course *does* teach seriously, because interviews test it, is the NLP
foundation: distributional semantics, tokenization, the task taxonomy, and how modern
transformer systems are actually assembled and evaluated. If word2vec, BPE, attention,
and LoRA are things you want to be able to derive and defend rather than merely invoke,
you are the reader.

## The module table

Modules 01–16 are the content. Labs 17–22 each pair with an earlier module and run on a
free Colab T4. Banks 23–25 are interview prep. Module 26 is the portfolio capstone.

| # | File | Owns |
|---|---|---|
| 00 | 00-README.md | This overview, roadmap, time budget, how to use it |
| 01 | [The NLP Landscape in 2026](01-nlp-landscape.md) | What NLP is in 2026; task taxonomy; the rules→statistical→neural→transformer arc; why RNN/LSTM lost; where the jobs are |
| 02 | [Word Vectors: The Representation That Started It All](02-word-vectors.md) | Distributional hypothesis; word2vec skip-gram + negative sampling (loss + gradient); GloVe; intrinsic eval and its lies; embedding bias; why embeddings still matter |
| 03 | [Tokenization: Turning Text into Model Inputs](03-tokenization.md) | BPE step by step; WordPiece vs Unigram/SentencePiece; byte-level BPE; vocab tradeoffs; multilingual fertility; failure modes and glitch tokens |
| 04 | [The Transformer Architecture](04-transformer-architecture.md) | Attention from first principles; scaled dot-product and the √d; multi-head; RoPE; pre- vs post-norm; SwiGLU; encoder/decoder/enc-dec; O(n²); GQA/RMSNorm refinements |
| 05 | [Pretraining: Objectives, Data, and Compute](05-pretraining.md) | Causal LM vs masked LM vs span corruption; why decoder-only won; pretraining data; FLOPs ≈ 6ND and Chinchilla in brief; the modern Llama-style recipe |
| 06 | [Transfer Learning: The Applied-NLP Workhorse](06-transfer-learning-tasks.md) | Fine-tuning encoders that still run production NLP: classification, NER, extractive QA, summarization/MT, sentence embeddings, bi- vs cross-encoders; when a 100M encoder beats an API LLM |
| 07 | [Post-Training: Turning a Base Model into an Assistant](07-post-training.md) | Base vs assistant; SFT and chat templates; RLHF (reward model + PPO + KL); reward hacking; DPO derivation; rejection sampling; data quality > quantity; safety tuning |
| 08 | [Adaptation: Prompting and Parameter-Efficient Fine-Tuning](08-prompting-peft.md) | In-context learning; prompt sensitivity; CoT; structured outputs; PEFT memory math; LoRA/QLoRA; the prompt vs RAG vs PEFT vs full-FT decision |
| 09 | [RAG and Agents: Grounding Models in the World](09-rag-agents.md) | Why retrieval; BM25 vs dense; the RAG pipeline; failure modes; agents (ReAct, tool calling, planning); agentic RAG; MCP |
| 10 | [Evaluation: The Skill That Gets You Hired](10-evaluation.md) | Perplexity; the benchmark canon and saturation; contamination; LLM-as-judge biases; building a product eval and regression gate |
| 11 | [Reasoning Models: CoT, Verifiers, and RL with Verifiable Rewards](11-reasoning.md) | What "reasoning" means; CoT and self-consistency; process vs outcome supervision; test-time compute; GRPO/RLVR; the DeepSeek-R1 recipe; distilling reasoning |
| 12 | [Inference and Decoding: Sampling, KV Cache, and Speculative Decoding](12-inference-decoding.md) | Decoding search (greedy/beam/temperature/top-k/top-p/min-p); KV cache math; speculative decoding; long-context/RoPE scaling; constrained decoding; latency anatomy |
| 13 | [Interpretability: Reading What the Model Is Doing](13-interpretability.md) | Behavioral → attributional → mechanistic; why attention ≠ explanation; induction heads, logit lens, SAEs; practical uses and honest limits |
| 14 | [Multimodality: When the Model Also Sees and Hears](14-multimodality.md) | Vision encoders + adapters vs early fusion; image tokenization; VLM training stages; document AI as the killer job; speech; any-to-any |
| 15 | [Risks and Safety: What Can Go Wrong and Who Owns It](15-risks-and-safety.md) | Hallucination, bias, privacy, security, misuse; safety eval and red-teaming; 2026 governance (EU AI Act, model cards, licenses); what you own on the job |
| 16 | [The Modern NLP Stack: Tools, Models, and a 90-Day Plan](16-modern-stack.md) | The 2026 production stack mapped to every module; open vs API model landscape and selection; what to name-drop per topic; a 90-day portfolio plan |
| 17 | [Lab: Embeddings and Tokenizers](17-lab-embeddings-tokenizers.md) | Lab 1 (Colab): train skip-gram from scratch, WEAT-lite bias, BPE vs SentencePiece fertility on English/Urdu/code |
| 18 | [Lab: Transformer From Scratch](18-lab-transformer-from-scratch.md) | Lab 2 (Colab): implement SDPA + MHA + RoPE + SwiGLU block, train a tiny decoder on TinyStories, visualize attention |
| 19 | [Lab: Fine-Tune an Encoder (the production workhorse)](19-lab-finetune-encoder.md) | Lab 3 (Colab): fine-tune DeBERTa/DistilBERT for classification + NER, calibration, latency-vs-API cost table |
| 20 | [Lab: SFT then DPO on a Small Model](20-lab-sft-dpo.md) | Lab 4 (Colab): SFT then DPO a small model with TRL+LoRA, judged before/after, chat-template gotchas |
| 21 | [Lab: RAG with a Real Eval Harness](21-lab-rag-eval.md) | Lab 5 (Colab): chunk→embed→FAISS→rerank RAG, hit@k + faithfulness judge, distractor failure and fix, golden-set gate |
| 22 | [Lab: Reasoning & Decoding](22-lab-reasoning-decoding.md) | Lab 6 (Colab): decoding playground, CoT vs direct on GSM8K, self-consistency scaling, best-of-n, KV-cache timing |
| 23 | [Interview Bank: Concepts and Breadth](23-interview-concepts.md) | Breadth bank: 60+ Q&A across modules 01–16, rapid-fire, "explain to a PM" |
| 24 | [Interview Bank: Implementation Drills](24-interview-implementation.md) | Coding drills with solutions: BPE, SDPA, RoPE, top-p, beam, BIO decode, LoRA layer, DPO loss, retriever |
| 25 | [Interview Bank: Applied NLP System Design](25-interview-applied-design.md) | Applied design rounds: the framework + 6 worked cases + junior-vs-senior answers |
| 26 | [Capstone: Production Support Intelligence System](26-capstone.md) | Portfolio capstone: Production Support Intelligence System, milestones, rubric, 3 alt tracks |

## How to use this

Read a module, then do its lab. The pairing is: module 02–03 → [lab 17](17-lab-embeddings-tokenizers.md);
module 04 → [lab 18](18-lab-transformer-from-scratch.md); module 06 → [lab 19](19-lab-finetune-encoder.md);
module 07–08 → [lab 20](20-lab-sft-dpo.md); module 09–10 → [lab 21](21-lab-rag-eval.md);
module 11–12 → [lab 22](22-lab-reasoning-decoding.md). Reading without building leaves
you able to recognize an idea but not defend it under follow-up questions, which is
exactly where interviews live.

Read [The Modern NLP Stack: Tools, Models, and a 90-Day Plan](16-modern-stack.md) alongside the core modules, not at the
end — it attaches a current tool name and tradeoff to each concept as you learn it, so
you never learn the mechanism without its 2026 production counterpart.

Drill the interview banks last. Cover each answer, reconstruct the mechanism, tradeoff,
and one number out loud, then check yourself. Bank 23 is breadth, 24 is implement-from-
scratch coding, 25 is applied design. Do the capstone as the portfolio piece you point at
in interviews when someone asks "what have you actually shipped."

## A realistic time budget

Working full time, plan roughly one content module per two or three evenings, with its
lab on a weekend. That puts the sixteen content modules and six labs at about 10–14 weeks.
The banks are another two to three weeks of active recall, and the capstone is a focused
two-week build. Call it a solid quarter end to end. You can compress it: if you only need
applied NLP interview readiness, modules 01–10 plus banks 23 and 25 plus the capstone are
the core, and the reasoning/interpretability/multimodality modules are depth you can defer.

## What you can run on

A free Colab T4 (16 GB) covers every lab. Nothing here needs a paid GPU, a cluster, or a
local card. Small models only — DistilBERT, DeBERTa-v3-small, Qwen2.5-0.5B-Instruct,
SmolLM2, all-MiniLM embeddings — and each lab is scoped to run in under ~25 minutes. Where
a topic genuinely needs multi-GPU systems work (kernels, parallelism, scaling-law fitting),
that depth lives in the sibling `language-modeling` course and this one points you there
rather than pretending a T4 can do it.
