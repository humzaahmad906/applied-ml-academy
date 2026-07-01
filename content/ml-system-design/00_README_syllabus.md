# ML System Design — A 2026 Course for Junior ML Engineers

**Goal:** Take you from "I can train a model in a notebook" to "I can design, defend, and operate a production ML system" — the exact skill tested in FAANG-level ML system design interviews and required of senior ML engineers.

**Audience:** Junior ML engineers (0–3 years). Assumes you know Python, PyTorch basics, and core ML (loss functions, transformers at a high level, train/val/test). Everything else is built up from first principles.

**Why this course exists:** Most ML system design material is frozen in 2021 (feature stores + XGBoost + "deploy with Flask"). In 2026, interviews and real jobs test *two* genres: (1) classic predictive ML systems (recommendations, search, fraud) and (2) LLM/GenAI systems (serving, RAG, agents, fine-tuning). This course covers both, weighted toward what is actually being used at the frontier right now: disaggregated inference, RLVR post-training, context/harness engineering, generative recommenders, and eval-driven development.

---

## Course map

| # | Module | One-line summary |
|---|--------|------------------|
| 01 | Foundations & the Interview Framework | Anatomy of a production ML system + how to structure any "Design X" answer |
| 02 | Data Engineering & Feature Platforms | Pipelines, feature stores, labeling, synthetic data, LLM data curation |
| 03 | Training & Post-Training Infrastructure | Distributed training, parallelism, SFT → DPO → RLVR/GRPO, PEFT |
| 04 | Inference Optimization & On-Device ML | Inference arithmetic, quantization, speculative decoding, edge stacks |
| 05 | LLM Serving Systems | Continuous batching, PagedAttention, prefix caching, P/D disaggregation |
| 06 | Retrieval & RAG Systems | Embeddings, ANN indexes, hybrid search, rerankers, RAG evaluation |
| 07 | Agentic Systems in Production | Workflows vs agents, tools/MCP, context engineering, security, agent evals |
| 08 | Classic ML Systems | Multi-stage recsys, two-tower retrieval, ranking, fraud, search |
| 09 | Evaluation, Observability & MLOps | Offline/online eval, A/B testing, LLM-as-judge, drift, tracing, CI for models |
| 10 | Capstones & the Interview Playbook | Full mock questions with model answers, capstone projects, 12-week plan |

Every module ends with: **References** (papers, engineering blogs, repos), a **hands-on project** sized for a single workstation/consumer GPU, and **interview Q&A with full answers**.

---

## How to use this course

1. **Read a module, then immediately do its project.** System design knowledge that hasn't been load-tested by a real implementation evaporates in interviews. The projects are deliberately small enough to finish in 1–2 weekends each.
2. **Answer the interview questions out loud before reading the answers.** The gap between "I recognize this" and "I can produce this under pressure" is the entire game.
3. **Keep a design journal.** After each module, write a one-page design doc applying the module to a system you invent. The interview is a writing/speaking exercise as much as a knowledge exercise.
4. **Numbers matter.** Memorize the back-of-envelope constants in the training, inference, and serving chapters (FLOPs per token, KV cache size, GPU bandwidth). Senior candidates are separated from junior ones by their ability to do capacity math live.

## Suggested 12-week schedule

- **Weeks 1–2:** Modules 01–02 + projects
- **Weeks 3–4:** Module 03 + project (this is the heaviest module)
- **Weeks 5–6:** Modules 04–05 + one combined serving project
- **Weeks 7–8:** Modules 06–07 + projects
- **Week 9:** Module 08 + recsys project
- **Week 10:** Module 09 + instrument your earlier projects
- **Weeks 11–12:** Module 10 — capstone + 4 timed mock interviews (45 min each, recorded)

## How the course is structured

Every chapter is self-contained: it opens with why the topic matters, builds the concepts from first principles, then closes with a hands-on project and interview Q&A. Read the chapters in order the first time through — later chapters assume the vocabulary and back-of-envelope constants introduced earlier. The capacity-math constants (FLOPs per token, KV-cache size, GPU bandwidth) recur across the training, inference, and serving chapters; the multi-stage funnel introduced in the classic-ML chapter reappears in retrieval and agent designs. Treat the recurring patterns — cascades, feedback loops, eval-driven development — as the through-line of the whole course.

To stay current, keep reading production engineering write-ups from the teams shipping these systems: model-serving platforms, recommendation and search stacks, and post-training recipes are all documented publicly and evolve quickly. The techniques in this course are the durable core; the specific tools around them will keep changing.
