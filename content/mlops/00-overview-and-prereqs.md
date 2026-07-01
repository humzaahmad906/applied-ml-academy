# 00 — Overview and Prerequisites

A complete MLOps curriculum, structured the same way a strong data engineering curriculum would be — foundations, tools, projects, advanced topics, and the architect track. By the end, you should be able to walk into a Fortune 50 MLOps / ML Platform Engineer interview and credibly defend two or three production-grade systems you've built end-to-end.

**MLOps** is not "ML plus DevOps." It's the discipline of taking models out of notebooks and running them as reliable software products: reproducible training, versioned data, monitored serving, automated retraining, governance, and cost discipline. Most ML projects fail in production not because the model was wrong but because the surrounding system was missing. MLOps is the surrounding system.

## The Eight Chapters in This Curriculum

| Chapter | What It Covers | Estimated Time |
|---|---|---|
| **Overview and Prerequisites** | Orientation, prereqs, study habits | 1 day |
| **Beginner Guide** | Foundations — ML refresher, Docker for ML, experiment tracking, model packaging, simple deployment | 4–6 weeks |
| **Medium Guide** | Productionization — model registry, feature pipelines, CI/CD/CT, monitoring basics, orchestration | 4–6 weeks |
| **Advanced Guide** | Scale — distributed training, Kubernetes-native serving, real-time inference, online feature serving, capstone | 6–8 weeks |
| **Next Steps** | Specialization — SageMaker / Vertex / Databricks, advanced monitoring, LLMOps, governance | 6–8 weeks |
| **Fortune 50 Projects** | 7 portfolio projects engineered to demonstrate F50-level competence | 6+ months total |
| **Advanced Topics** | Post-graduate — distributed systems, training/inference optimization, security, LLM internals | Reference |
| **ML Architect Track** | ML architect / staff+ track — strategy, decisions, organization, migrations | Reference |

**Total realistic timeline:** 6 months of focused part-time study (10–15 hrs/week) to finish foundations through specialization. Then 6–12 months building 2–3 portfolio projects deeply. So roughly 12–18 months from a serious start to credibly interviewing for F50 MLOps / ML Platform roles.

If you already have strong data engineering skills, halve those estimates. MLOps stands on top of DE — you can't run a model pipeline without a data pipeline.

## What an MLOps / ML Platform Engineer Actually Does

The role title varies wildly: MLOps Engineer, ML Platform Engineer, ML Infrastructure Engineer, AI Infrastructure Engineer, Applied ML Engineer (production-leaning). The work is the same:

- **Data and feature pipelines** for training and serving — including point-in-time correctness for features
- **Training infrastructure** — running distributed training on GPUs without it being painful
- **Experiment tracking, model registry, lineage** — every artifact reproducible, traceable, signed
- **Serving infrastructure** — batch, online, streaming inference at the latency and throughput business needs
- **CI/CD/CT** — continuous integration, continuous delivery, continuous training (the third one is the MLOps-specific addition)
- **Monitoring** — data drift, concept drift, prediction drift, system metrics, business metrics
- **Governance** — model cards, audit trails, fairness/bias checks, regulatory compliance (especially EU AI Act, NYC Local Law 144, etc.)
- **Cost** — GPU utilization, idle clusters, inference cost per request, training cost per experiment

You are **not** primarily writing models. Data scientists and ML researchers write models. You make their models real.

## Prerequisites — What You Must Have Before Module 1

Non-negotiable. Without these you'll thrash in tooling instead of learning MLOps.

### 1. Python — Solid Intermediate Level

You must be comfortable with:

- Functions, classes, modules, decorators, context managers
- List/dict/set comprehensions, generators, `yield from`
- Virtual environments (`uv` is the 2026 standard; `venv` and `pip` are the floor)
- Typing — basic `typing` module use (`List`, `Dict`, `Optional`, `Callable`)
- `pydantic` data models (they're everywhere in modern ML/serving code)
- Reading JSON, CSV, Parquet (pandas/polars/pyarrow)
- Basic concurrency: `asyncio`, `concurrent.futures`
- Logging via `logging.getLogger(__name__)` — never `print` in production code
- Packaging — `pyproject.toml`, the difference between a script, a module, and an installable package

You do not need to be a Python expert. You need to be able to write a 300-line service or training script without googling syntax every five minutes.

**If you're not there yet:** Spend a week on intermediate Python — closures, decorators, context managers, typing — then build a small FastAPI service that wraps a scikit-learn model. That single exercise covers 80% of the Python you need.

### 2. Math and Statistics — Working Level

You don't need a PhD. You need to be unintimidated by:

- Probability basics — random variables, distributions, expectation, variance
- Linear algebra basics — vectors, matrices, dot products, norms, eigenvalues at a conceptual level
- Calculus basics — gradients, partial derivatives (chain rule is enough; you won't be solving integrals)
- Statistics — mean, median, stddev, percentiles, hypothesis testing intuition, p-values
- Distance/similarity metrics — Euclidean, cosine, Manhattan, Hamming
- Loss functions — MSE, cross-entropy, log loss; *why* each one is shaped that way

You will not derive backpropagation. You will reach for `torch.nn.CrossEntropyLoss` and know enough to debug when its inputs are the wrong shape.

**If you're not there yet:** Start with visual linear algebra and calculus — the "why shapes matter" intuition is more important than formal proofs. Then work through applied statistics and ML intuition content until you can explain the bias-variance tradeoff and how a gradient descent step works. That covers what you need.

### 3. ML Fundamentals — Intermediate Level

You should be able to explain, in your own words:

- Supervised vs unsupervised vs reinforcement vs self-supervised learning
- Train/validation/test split, cross-validation, why you need a held-out test set
- Bias-variance tradeoff, overfitting vs underfitting, regularization (L1/L2/dropout)
- Common classical algorithms — linear/logistic regression, decision trees, random forests, gradient boosting (XGBoost/LightGBM), k-NN, k-means
- Common deep learning architectures — MLP, CNN, RNN/LSTM (briefly), Transformers (the big one)
- Evaluation metrics — accuracy, precision, recall, F1, AUC-ROC, PR-AUC; for regression: MAE, MSE, RMSE, MAPE; for ranking: NDCG, MAP
- Class imbalance handling — resampling, class weights, threshold tuning
- The basic embedding idea — words/items/users as dense vectors

You don't need to be able to derive these. You need to recognize them and pick appropriate ones for a given problem.

**If you're not there yet:** Work through a structured ML course covering supervised learning fundamentals and then a hands-on deep learning course that builds intuition through code rather than theory. Cover both classical methods and neural networks before continuing.

### 4. SQL — Solid Foundation

ML pipelines start with data, which lives in databases. You need:

- `SELECT`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`
- All four `JOIN` types
- Subqueries and CTEs (`WITH`)
- Aggregation functions, `CASE WHEN`, `DISTINCT`
- Window functions (`ROW_NUMBER`, `LAG`, `LEAD`, `SUM OVER`, `AVG OVER`) — required for point-in-time-correct feature computation, which we'll cover in depth
- Time-based filtering (you'll write a lot of `WHERE event_time BETWEEN ... AND ...`)

**If you're not there yet:** Work through a structured SQL tutorial that covers basic SELECT through window functions. Don't skip windows — they're the heart of feature engineering at scale.

### 5. Command Line and Git — Comfortable

- Shell navigation, piping, redirection, environment variables, SSH
- `git clone/add/commit/push/pull`, branches, merges, rebases, basic conflict resolution
- Reading a diff, reading a git log, blame, bisect at a basic level
- `tmux` or `screen` for long-running training sessions on remote machines

### 6. Linux and Networking Basics

You'll be running things on Linux servers and inside Linux containers constantly:

- Filesystem layout, permissions, `chmod`, `chown`
- Process management (`ps`, `top`, `htop`, `kill`)
- Networking — ports, `localhost` vs `0.0.0.0`, basic firewalls, DNS, the OSI model in passing
- Disk and memory diagnostics — `df`, `du`, `free`, `iostat`
- GPU diagnostics — `nvidia-smi`, basic CUDA terminology (CUDA version, driver version, compute capability)

### 7. A Computer That Won't Hold You Back

- 16GB RAM minimum, 32GB strongly recommended
- 100GB free disk space
- macOS or Linux (WSL2 on Windows; don't try MLOps on raw Windows)
- **GPU access** — you don't need a local GPU early on. For deep learning, use Colab (free T4), Kaggle (free P100/T4), or rent an A100/H100 hour by the hour on RunPod / Lambda / Modal / Paperspace when needed. Buying a 4090 is fine if you have the budget.

You will also need accounts on:

- **AWS** (most common F50 cloud) — set up with billing alerts at $5, $20, $50
- **GCP** (close second) — $300 free credit covers a lot
- **Hugging Face** — for models and datasets
- **Weights & Biases** — free tier is generous; experiment tracking
- **GitHub** — for projects, Actions, free Codespaces

Set up budget alerts on day one. GPU costs add up fast.

## Study Habits That Actually Work

### Type the code, including the YAML and Dockerfiles

Every example. Don't copy-paste. Your hands need to learn the syntax — `apiVersion: apps/v1` should be muscle memory by month two. YAML for Kubernetes, GitHub Actions, dbt, MLflow, KServe, etc., is half the job.

### Build a `notes/` directory

One folder, one markdown file per topic. Write down:

- What the topic taught you in your own words
- What broke and how you fixed it
- Three questions you couldn't answer yet
- A code snippet you want to remember

After six months, this folder is more valuable than any course material.

### Use AI as a tutor, not a crutch

In MLOps especially, AI agents are very good at writing boilerplate config but very bad at reasoning about production tradeoffs. Use them for syntax. Don't use them for design decisions until you understand the design space yourself — otherwise you'll deploy a load-balanced GPU inference service and have no idea why it falls over.

A good prompt pattern: *"Here's an error I'm getting. Here's what I think is happening. Here's what I tried. Can you tell me what concept I'm missing?"*

### Build in public

A weekly LinkedIn post or blog entry on what you built. Three benefits:

1. Writing forces you to clarify your understanding
2. You build an audience of MLOps people before you need one
3. When you apply for jobs, "here are 20 posts about my journey" beats a bullet-point resume

### Optimize for full pipelines, not individual components

The single most common beginner mistake in MLOps is going deep on one tool (MLflow! Kubeflow! BentoML!) without ever connecting it to a real end-to-end flow. The interview answer "I deeply understand MLflow" is much weaker than "I have a project where data lands in S3, features are computed in a pipeline, models train on a schedule, get registered, deployed via a canary, monitored, and retrained — and I can show you the repo."

Always go end-to-end before going deep.

### Don't compare to others

The MLOps community is full of people brushing up from senior ML or senior infra roles. Your pace is your pace. The only comparison that matters is you-today vs. you-six-months-ago.

## How This Curriculum Differs from a Typical MLOps Course

Most short MLOps courses teach one tool stack in 8–12 hours. They're useful but they don't:

1. Tell you what to skip if you already know it
2. Show you what's covered in introductory material vs. critical-but-missing in industry
3. Give you projects substantial enough to anchor a F50 interview
4. Bridge from "I followed a tutorial" to "I'm employable at senior level"
5. Cover the architect track that exists above senior IC

This curriculum fills those gaps: opinionated sequencing, industry-calibrated depth, and portfolio projects sized for senior-level interviews.

## When to Move to the Beginner Guide

When you've checked off everything in the **Prerequisites** section above, start the Beginner Guide.

If a prereq is shaky, fix it first. Two weeks of solid prep saves three months of confused thrashing later.

If you have a data engineering background already: you have a substantial head start. Skim the Beginner Guide quickly (it overlaps with DE foundations), and budget more time for the ML-specific content in the Medium and Advanced guides.

If you have an ML research / data science background: you'll move quickly through the ML content but will be slow on infrastructure (containers, Kubernetes, networking). Don't underestimate the infra side — it's where the role lives.
