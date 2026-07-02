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

**Core modules 01–10 are the spine — read them in order.** The modules below extend it and are meant to be read after the core, in sequence; a time-constrained learner can treat 13–14 as electives (see the learning paths below).

| # | Module | One-line summary |
|---|--------|------------------|
| 11 | Economics & Cost Modeling | Cloud/API pricing, build-vs-buy crossover math, a TCO worksheet |
| 12 | DevOps & Deployment | Dockerfiles, Kubernetes, autoscaling, a full CI/CD pipeline for models |
| 13 | Case Studies | Five end-to-end system walkthroughs assembling everything you've learned |
| 14 | Domain Variations | How the core patterns bend for healthcare, finance, autonomous, manufacturing |
| 15 | Cumulative Capstone | One end-to-end build across all modules — the portfolio centerpiece |
| 16 | Career, Portfolio & Interview Strategy | Portfolio structure, interview tactics, career progression, reading list |
| 17 | Course Delivery & Roadmap | Learning design, the capacity-math cheat sheet, staying current, cohort guidance |

Every core module ends with: **References** (papers, engineering blogs, repos), a **hands-on project** sized for a single workstation/consumer GPU, and **interview Q&A with full answers**.

---

## Prerequisite self-check

Before starting, confirm you can honestly answer yes to these. If not, close the gap first — the course assumes them and moves fast.

- **Python:** comfortable with classes, decorators, async/await basics, and reading library source. If not → any solid intermediate-Python course.
- **PyTorch:** you can write a training loop by hand (forward, loss, `backward()`, optimizer step), and you know what a `DataLoader` and a `Dataset` are. If not → the official PyTorch 60-minute blitz + one from-scratch training script.
- **Core ML:** you know cross-entropy vs MSE, train/val/test discipline, overfitting, and the transformer at a block-diagram level (attention, MLP, residual, layernorm). If not → any modern intro-to-DL course and Karpathy's "Let's build GPT".
- **Systems basics:** you know what a REST endpoint, a database index, and a container are. If not → the DevOps chapter (module 12) will be rough; skim a Docker primer first.

You do **not** need prior distributed-training, CUDA, or MLOps experience — those are built up from first principles.

## Learning paths

The course is one spine with three entry emphases. All learners do the core (01–10); the paths differ in what they linger on and what they skim.

- **SWE → ML:** you have systems, you need modeling intuition. Linger on 03 (training), 04 (inference), 08 (classic ML fundamentals) and the Foundations Boxes throughout. Skim nothing in 01–02; those reframe what you already know in ML terms. 12 (DevOps) will feel easy — use it as a checkpoint.
- **Data Scientist → MLE:** you have modeling, you need productionization. Linger on 02 (feature platforms), 05 (serving), 09 (eval/observability/MLOps), 11 (economics), 12 (DevOps). The Tool Survival Guides are your fastest ramp.
- **Experienced MLE → interview prep:** you have breadth, you need pressure-tested recall and tradeoff fluency. Prioritize the Interview Q&A in every module, the War Stories, module 10 (mock questions), module 13 (case studies as retellable narratives), and module 16 (interview strategy). Use module 17's cheat sheet for the capacity-math constants.

Every learner finishes with the **cumulative capstone (module 15)** — it is the single artifact that proves end-to-end capability and anchors the portfolio (module 16).

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

The **extension modules (11–17)** come after the core, read in order. They are self-paced rather than fitted to the 12-week clock: pull in **11 (economics)** and **12 (DevOps)** whenever a project needs cost math or a real deployment; read **13 (case studies)** and **14 (domain variations)** once the core patterns are solid; run **15 (cumulative capstone)** as the integrating build (start planning it around week 10); and use **16 (career)** and **17 (delivery & roadmap)** as you prepare to interview. See the learning paths above for which of these matter most for your target role.

## How the course is structured

Every chapter is self-contained: it opens with why the topic matters, builds the concepts from first principles, then closes with a hands-on project and interview Q&A. Read the chapters in order the first time through — later chapters assume the vocabulary and back-of-envelope constants introduced earlier. The capacity-math constants (FLOPs per token, KV-cache size, GPU bandwidth) recur across the training, inference, and serving chapters; the multi-stage funnel introduced in the classic-ML chapter reappears in retrieval and agent designs. Treat the recurring patterns — cascades, feedback loops, eval-driven development — as the through-line of the whole course.

To stay current, keep reading production engineering write-ups from the teams shipping these systems: model-serving platforms, recommendation and search stacks, and post-training recipes are all documented publicly and evolve quickly. The techniques in this course are the durable core; the specific tools around them will keep changing.
