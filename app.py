import os
import re
import json
import secrets
import hmac
from functools import wraps
from datetime import datetime, date
from pathlib import Path

import markdown as md
from flask import (Flask, render_template, request, redirect, url_for,
                   session, abort, flash)
from werkzeug.security import generate_password_hash, check_password_hash

import art
import data

# ---------------------------------------------------------------- config
BRAND = "Applied ML Academy"
MONOGRAM = "A"
EST_YEAR = "2026"
PROGRAM_LINE = "Professional Certification Program"
INSTRUCTOR = "Humza Ahmad"
INSTRUCTOR_TITLE = "Program Director & Lead Instructor"
COSIGNER = "A. Rahman"
COSIGNER_TITLE = "Dean of Curriculum"

PASS_MARK = int(os.environ.get("PASS_MARK", "80"))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "change-me")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Data lives in Firestore (see data.py) so it survives redeploys. Credentials come
# from FIREBASE_CREDENTIALS / GOOGLE_APPLICATION_CREDENTIALS / FIRESTORE_EMULATOR_HOST.


# ---------------------------------------------------------------- courses
# Static metadata per course; module lists are loaded from content/<slug>/*.md.
COURSES = [
    {"code": "LANGMDL", "title": "LLM Foundation-Model Engineer Nanodegree",
     "program": "Nanodegree", "role": "Foundation-Model Engineer",
     "prereqs": ["linear-algebra", "calculus-gradients", "pytorch-essentials",
                 "deep-learning-foundations", "vlm-guide"],
     "blurb": "Become the engineer who builds language models, not just calls them. "
              "Tokenizer, transformer, kernels, parallelism, scaling, inference, and "
              "alignment — a modern LLM end to end. The deepest track we offer.",
     "hours": "48", "level": "Advanced", "tag": "LLMs & Systems", "accent": "#c6a04e",
     "order": 4, "path_note": "Go deep — build a modern LLM end to end. The hardest track.",
     "slug": "language-modeling",
     "outcomes": ["Implement a BPE tokenizer, transformer, and training loop from scratch",
                  "Reason about FLOPs, memory, and parallelism for real training runs",
                  "Optimize inference: KV cache, quantization, speculative decoding",
                  "Align a base model with SFT, DPO, and GRPO",
                  "Answer frontier-lab interview questions across four banks"]},
    {"code": "LLMVLM", "title": "GenAI Engineer Nanodegree",
     "program": "Nanodegree", "role": "GenAI Engineer",
     "prereqs": ["python-foundations", "ml-foundations", "pytorch-essentials",
                 "apis-web-services", "software-engineering-practices"],
     "blurb": "Become a generative-AI engineer: attention, KV cache, RAG, and agents — "
              "and the tradeoffs behind each. Foundations through frontier, senior-level "
              "throughout. The vocabulary and mental models the rest build on.",
     "hours": "36", "level": "Foundations", "tag": "Generative AI", "accent": "#8b5cf6",
     "order": 1, "path_note": "Start here — the vocabulary and mental models the rest build on.",
     "slug": "vlm-guide",
     "outcomes": ["Explain attention, KV cache, and modern decoder design",
                  "Design retrieval-augmented generation pipelines",
                  "Build and reason about agentic systems",
                  "Read papers and speak the vocabulary fluently"]},
    {"code": "MLSYS", "title": "ML Systems Architect Nanodegree",
     "program": "Nanodegree", "role": "ML Systems Architect",
     "prereqs": ["ml-foundations", "deep-learning-foundations",
                 "software-engineering-practices", "apis-web-services", "mlops"],
     "blurb": "Design ML systems the way a staff engineer does: data platforms, training "
              "and serving infra, RAG, agents, recsys, and the interview playbook. Learn "
              "to frame any 'Design X' problem and defend the tradeoffs.",
     "hours": "40", "level": "Advanced", "tag": "System Design", "accent": "#0ea5e9",
     "order": 5, "path_note": "Capstone — design end-to-end systems and prep for interviews.",
     "slug": "ml-system-design",
     "outcomes": ["Frame any ML system-design interview with a repeatable structure",
                  "Design feature platforms and training/serving infrastructure",
                  "Architect retrieval, agents, recsys, search, and fraud systems",
                  "Handle evaluation, observability, and MLOps at scale"]},
    {"code": "MLOPS", "title": "MLOps Engineer Nanodegree",
     "program": "Nanodegree", "role": "MLOps Engineer",
     "prereqs": ["python-foundations", "cli-git", "software-engineering-practices",
                 "apis-web-services", "docker-containers", "cloud-linux"],
     "blurb": "Take models to production and keep them alive: serving, monitoring, CI/CD "
              "for models, and the failure modes nobody warns you about. Practitioner to "
              "architect track.",
     "hours": "40", "level": "Intermediate → Advanced", "tag": "MLOps", "accent": "#ec4899",
     "order": 3, "path_note": "Take models to production: serving, monitoring, CI/CD.",
     "slug": "mlops",
     "outcomes": ["Serve, monitor, and version models in production",
                  "Build CI/CD pipelines for ML",
                  "Diagnose the failure modes that break deployed models",
                  "Grow from practitioner to ML architect"]},
    {"code": "PRINML", "title": "Principal ML Engineer Nanodegree",
     "program": "Nanodegree", "role": "Principal ML Engineer",
     "prereqs": ["ml-system-design", "mlops"],
     "blurb": "Operate at org scale with the judgment of a 15-year veteran: technical "
              "strategy, platform architecture, unit economics, migrations, governance, "
              "and influence without authority. The senior-to-principal delta, compressed.",
     "hours": "50", "level": "Expert", "tag": "Career", "accent": "#dc2626",
     "order": 6, "path_note": "Beyond staff — operate at org scale with principal-level judgment.",
     "slug": "principal-ml-engineer",
     "outcomes": ["Write technical strategy documents that move an entire org",
                  "Architect ML platforms, training fleets, and serving stacks at org scale",
                  "Make build-vs-buy and migration decisions with defensible unit economics",
                  "Carry the production failure pattern library of a 15-year veteran",
                  "Pass principal-level (L7+/E7+) interview loops and build the portfolio to prove it"]},
    {"code": "DATAENG", "title": "Data Engineer Nanodegree",
     "program": "Nanodegree", "role": "Data Engineer",
     "prereqs": ["python-foundations", "sql-databases", "pandas-analysis",
                 "software-engineering-practices"],
     "blurb": "Build the data layer every model and system sits on top of: pipelines, "
              "warehouses, and the architecture behind them — from first principles to "
              "Fortune-100 scale and the data-architect track.",
     "hours": "36", "level": "Beginner → Advanced", "tag": "Data", "accent": "#f59e0b",
     "order": 2, "path_note": "The data layer every model and system sits on top of.",
     "slug": "data-engineering",
     "outcomes": ["Build reliable batch and streaming data pipelines",
                  "Model data warehouses and lakehouses",
                  "Design data architecture at Fortune-100 scale",
                  "Follow the path to data architect"]},
]
# Open foundation track: free, login required (so notes/progress work), no certificate.
# Content is authored course-by-course; until a course has content files its syllabus
# is shown as "lessons coming soon". `order` is the recommended beginner sequence.
def _syl(*pairs):
    return [{"title": t, "desc": d} for t, d in pairs]


FOUNDATIONS = [
    {"code": "PY", "title": "Python for Data & ML", "slug": "python-foundations",
     "tag": "Programming", "accent": "#3776ab", "level": "Beginner", "hours": "12", "order": 1,
     "foundation": True, "certificate": False,
     "blurb": "Start from zero: Python syntax through NumPy, the language every other course assumes.",
     "syllabus": _syl(
        ("Setup and the REPL", "Install Python, run code, use a notebook and an editor."),
        ("Variables, types, and control flow", "Numbers, strings, booleans, if/for/while."),
        ("Data structures", "Lists, dicts, sets, tuples and when to use each."),
        ("Functions and modules", "Write reusable functions; organize code into modules."),
        ("Files and error handling", "Read/write files; handle exceptions cleanly."),
        ("Object-oriented basics", "Classes, methods, and when objects help."),
        ("Comprehensions and iterators", "Pythonic loops, generators, and lazy iteration."),
        ("NumPy and vectorized math", "Arrays, broadcasting, and why loops are slow."))},
    {"code": "LINALG", "title": "Linear Algebra for ML", "slug": "linear-algebra",
     "tag": "Math", "accent": "#16a34a", "level": "Beginner", "hours": "10", "order": 2,
     "foundation": True, "certificate": False,
     "blurb": "Vectors, matrices, and matrix multiplication — the language tensors are written in.",
     "syllabus": _syl(
        ("Vectors and geometry", "What a vector is; addition, scaling, and intuition."),
        ("Matrices and operations", "Shapes, transpose, and basic operations."),
        ("Matrix multiplication", "The one operation that dominates ML compute."),
        ("Norms and distances", "Measuring size and similarity."),
        ("Dot products and projections", "Similarity, angles, and why attention uses them."),
        ("Eigenvalues and SVD", "Decompositions, intuitively, and where they show up."),
        ("Linear algebra in NumPy", "Do all of the above in code."))},
    {"code": "CALC", "title": "Calculus & Gradients", "slug": "calculus-gradients",
     "tag": "Math", "accent": "#0d9488", "level": "Beginner", "hours": "9", "order": 3,
     "foundation": True, "certificate": False,
     "blurb": "Derivatives, the chain rule, and gradient descent — how models actually learn.",
     "syllabus": _syl(
        ("Functions and limits", "The setup you need, no more."),
        ("Derivatives", "Slope, rate of change, and notation."),
        ("The chain rule", "Composing functions — the heart of backprop."),
        ("Partial derivatives and gradients", "Slopes in many dimensions."),
        ("Gradient descent", "Follow the slope downhill to minimize loss."),
        ("Backprop intuition", "How the chain rule trains a network."))},
    {"code": "PROB", "title": "Probability & Statistics for ML", "slug": "probability-stats",
     "tag": "Math", "accent": "#7c3aed", "level": "Beginner", "hours": "11", "order": 4,
     "foundation": True, "certificate": False,
     "blurb": "Distributions, Bayes, and estimation — the reasoning under every metric and loss.",
     "syllabus": _syl(
        ("Probability basics", "Events, conditional probability, independence."),
        ("Random variables and distributions", "Discrete and continuous; the common ones."),
        ("Expectation and variance", "Averages, spread, and what they tell you."),
        ("Bayes' theorem", "Updating beliefs from evidence."),
        ("Sampling and the CLT", "Why sample means behave nicely."),
        ("Estimation and maximum likelihood", "Fitting parameters to data."),
        ("Hypothesis testing basics", "p-values and what they do and don't mean."))},
    {"code": "CLI", "title": "Command Line & Git", "slug": "cli-git",
     "tag": "Tooling", "accent": "#475569", "level": "Beginner", "hours": "8", "order": 5,
     "foundation": True, "certificate": False,
     "blurb": "The terminal, shell, and version control — the daily workflow of every engineer.",
     "syllabus": _syl(
        ("The terminal and shell", "Where work happens; navigating the filesystem."),
        ("Files and text tools", "cat, grep, pipes, and redirection."),
        ("Bash scripting basics", "Variables, loops, and small automations."),
        ("Git fundamentals", "Commits, history, and the staging area."),
        ("Branching and merging", "Work in parallel without chaos."),
        ("GitHub and pull requests", "Collaborate and review changes."))},
    {"code": "SQL", "title": "SQL & Databases", "slug": "sql-databases",
     "tag": "Data", "accent": "#0891b2", "level": "Beginner", "hours": "10", "order": 6,
     "foundation": True, "certificate": False,
     "blurb": "Query and model relational data — the backbone of data engineering and analytics.",
     "syllabus": _syl(
        ("The relational model", "Tables, rows, keys, and relationships."),
        ("SELECT and filtering", "Get exactly the rows you want."),
        ("Joins", "Combine tables the right way."),
        ("Aggregation and GROUP BY", "Summarize data."),
        ("Subqueries and CTEs", "Compose complex queries readably."),
        ("Indexing and performance", "Why some queries are slow."),
        ("Schema design basics", "Model a domain without pain later."))},
    {"code": "PANDAS", "title": "Data Analysis with Pandas", "slug": "pandas-analysis",
     "tag": "Data", "accent": "#db2777", "level": "Beginner", "hours": "10", "order": 7,
     "foundation": True, "certificate": False,
     "blurb": "Load, clean, and explore real data with DataFrames — the analyst's core tool.",
     "syllabus": _syl(
        ("Series and DataFrames", "The two objects everything is built on."),
        ("Loading and inspecting data", "CSV, JSON, and a first look."),
        ("Selection and filtering", "Slice data by label and condition."),
        ("Cleaning and missing data", "Handle nulls, types, and duplicates."),
        ("GroupBy and aggregation", "Split-apply-combine."),
        ("Merging and reshaping", "Join, pivot, and melt."),
        ("Plotting and EDA", "See the data before you model it."))},
    {"code": "MLF", "title": "Machine Learning Foundations", "slug": "ml-foundations",
     "tag": "Machine Learning", "accent": "#ea580c", "level": "Beginner → Intermediate", "hours": "14", "order": 8,
     "foundation": True, "certificate": False,
     "blurb": "Classical ML end to end: regression, trees, evaluation, and the traps that fool beginners.",
     "syllabus": _syl(
        ("What machine learning is", "Supervised, unsupervised, and the workflow."),
        ("Data splits and leakage", "Train/validation/test done right."),
        ("Linear and logistic regression", "The workhorse models."),
        ("Decision trees and ensembles", "Trees, random forests, gradient boosting."),
        ("Evaluation metrics", "Accuracy, precision/recall, ROC, and when each lies."),
        ("Overfitting and regularization", "Bias, variance, and controlling them."),
        ("Cross-validation", "Trustworthy estimates of performance."),
        ("A first end-to-end project", "From raw data to an evaluated model."))},
    {"code": "DLF", "title": "Deep Learning & Neural Nets", "slug": "deep-learning-foundations",
     "tag": "Deep Learning", "accent": "#6366f1", "level": "Beginner → Intermediate", "hours": "14", "order": 9,
     "foundation": True, "certificate": False,
     "blurb": "From a single neuron to a trained network: forward pass, backprop, and optimization.",
     "syllabus": _syl(
        ("From linear models to neurons", "Why stack and nonlinearize."),
        ("Activation functions", "ReLU and friends, and what they do."),
        ("The forward pass", "How a network computes an output."),
        ("Backpropagation", "Gradients through the whole network."),
        ("Loss functions and optimizers", "What you minimize and how."),
        ("Training loops and batching", "Epochs, mini-batches, learning rate."),
        ("Regularization", "Dropout, weight decay, early stopping."),
        ("Your first neural network", "Train one end to end."))},
    {"code": "PYT", "title": "PyTorch Essentials", "slug": "pytorch-essentials",
     "tag": "Deep Learning", "accent": "#ee4c2c", "level": "Beginner → Intermediate", "hours": "10", "order": 10,
     "foundation": True, "certificate": False,
     "blurb": "The framework the deep courses assume: tensors, autograd, modules, and a real training loop.",
     "syllabus": _syl(
        ("Tensors", "Creation, shapes, and operations."),
        ("Autograd", "Automatic differentiation, demystified."),
        ("nn.Module and layers", "Build models from components."),
        ("Datasets and DataLoaders", "Feed data efficiently."),
        ("The training loop", "Forward, loss, backward, step."),
        ("Saving and loading", "Checkpoints and inference."),
        ("Devices and GPUs", "Move work to the accelerator."))},
    {"code": "DOCK", "title": "Docker & Containers for ML", "slug": "docker-containers",
     "tag": "Ops", "accent": "#2496ed", "level": "Beginner", "hours": "8", "order": 11,
     "foundation": True, "certificate": False,
     "blurb": "Package anything to run anywhere — the unit of deployment for modern ML.",
     "syllabus": _syl(
        ("Why containers", "The 'works on my machine' problem, solved."),
        ("Images vs containers", "The two core concepts."),
        ("Dockerfile basics", "Build a reproducible image."),
        ("Running: ports and volumes", "Talk to a container and persist data."),
        ("Docker Compose", "Run multi-service stacks."),
        ("Packaging an ML model", "Containerize an inference service."),
        ("Registries and sharing", "Push, pull, and deploy."))},
    {"code": "CLOUD", "title": "Cloud & Linux Fundamentals", "slug": "cloud-linux",
     "tag": "Ops", "accent": "#d97706", "level": "Beginner", "hours": "10", "order": 12,
     "foundation": True, "certificate": False,
     "blurb": "Linux and cloud building blocks — compute, storage, networking, and IAM — without the jargon.",
     "syllabus": _syl(
        ("Linux essentials", "The filesystem, shell, and packages."),
        ("Users, permissions, processes", "Who can do what, and what's running."),
        ("Compute: virtual machines", "Rent a computer in the cloud."),
        ("Storage and object stores", "Disks, buckets, and when to use each."),
        ("Networking basics", "IPs, ports, DNS, and firewalls."),
        ("IAM and security", "Identities, roles, and least privilege."),
        ("Cost and trade-offs", "Pick the right service without overspending."))},
    {"code": "API", "title": "APIs & Web Services", "slug": "apis-web-services",
     "tag": "Programming", "accent": "#0891b2", "level": "Beginner", "hours": "12", "order": 13,
     "foundation": True, "certificate": False,
     "blurb": "Build and consume HTTP services — the interface between your model and the "
              "world. HTTP, REST, and a full FastAPI app serving predictions. The prereq "
              "every MLOps, GenAI, and Systems track assumes.",
     "syllabus": _syl(
        ("How the web works: HTTP fundamentals", "Methods, status codes, headers, and the request/response cycle."),
        ("JSON and REST: designing resource APIs", "REST conventions and JSON so your endpoints are predictable."),
        ("Consuming APIs from Python", "Call APIs with requests and httpx: auth, timeouts, retries, pagination."),
        ("Your first FastAPI application", "Routes, parameters, and the dev server with FastAPI."),
        ("Request and response models with Pydantic", "Validate inputs and shape outputs with Pydantic v2."),
        ("Errors, dependencies, and middleware", "Handle errors, inject dependencies, add cross-cutting behavior."),
        ("Serving an ML model behind an endpoint", "Load a model once and serve validated predictions."),
        ("Authentication, rate limiting, and security", "API keys, bearer tokens, CORS, and rate limits."),
        ("Async basics and concurrency for APIs", "When async def helps, when it doesn't, and how to use it."),
        ("Testing, docs, and API versioning", "pytest + TestClient, OpenAPI docs, and versioning."))},
    {"code": "SWE", "title": "Software Engineering Practices", "slug": "software-engineering-practices",
     "tag": "Tooling", "accent": "#57534e", "level": "Beginner", "hours": "12", "order": 14,
     "foundation": True, "certificate": False,
     "blurb": "Write code a team can maintain, test, and ship: packaging, dependency "
              "management, testing, types, linting, logging, config, and CI. The practices "
              "every Nanodegree assumes but nobody else teaches.",
     "syllabus": _syl(
        ("Project structure and Python packaging", "pyproject.toml, src layout, and editable installs."),
        ("Virtual environments and dependency management", "Isolate and pin dependencies with uv for reproducible builds."),
        ("Testing with pytest: the fundamentals", "Assertions, test discovery, and testing as you go."),
        ("Fixtures, parametrize, and test organization", "Reusable setup, many-case tests, and tests that scale."),
        ("Type hints and static analysis", "Annotate code and catch bugs before runtime with a type checker."),
        ("Linting, formatting, and pre-commit hooks", "Automate code quality with ruff and pre-commit."),
        ("Logging done right", "Levels, configuration, and structured logs instead of print()."),
        ("Error handling and exceptions", "Specific and custom exceptions, and the fail-loud principle."),
        ("Configuration and secrets management", "Env vars, .env, pydantic-settings, and never committing secrets."),
        ("Code review and pull-request discipline", "What a good PR and a good review actually look like."),
        ("CI/CD with GitHub Actions", "Run tests and lint on every push — an intro to continuous integration."))},
    {"code": "CV", "title": "Computer Vision & CNNs", "slug": "computer-vision",
     "tag": "Deep Learning", "accent": "#14b8a6", "level": "Beginner → Intermediate", "hours": "13", "order": 15,
     "foundation": True, "certificate": False,
     "prereqs": ["deep-learning-foundations", "pytorch-essentials"],
     "blurb": "How machines see: from pixels to convolutions to vision transformers. Bridges the gap "
              "between the MLPs of deep-learning foundations and the ViT-based models modern vision and "
              "vision-language systems assume — convolutions, the classic architectures, transfer learning, "
              "detection, segmentation, and a full end-to-end image classifier.",
     "outcomes": ["Explain why CNNs beat MLPs on images and read any conv stack's shapes and receptive fields",
                  "Build, train, and evaluate a CNN in PyTorch on a real dataset with augmentation",
                  "Transfer-learn from a pretrained backbone — knowing when to freeze vs fine-tune",
                  "Understand detection, segmentation, and vision transformers, and bridge into the VLM course"],
     "syllabus": _syl(
        ("Images as tensors", "Pixels, channels, H×W×C, batches, and normalization; why MLPs waste image structure."),
        ("The convolution operation", "Kernels, feature maps, stride and padding; parameter sharing and translation equivariance."),
        ("Building a CNN", "Conv-activation-pooling stacks, receptive fields, and a small PyTorch CNN."),
        ("Classic architectures", "LeNet → AlexNet → VGG → ResNet skip connections, with a nod to EfficientNet and ConvNeXt."),
        ("Training and transfer learning", "Augmentation, training on CIFAR, and fine-tuning a pretrained backbone."),
        ("Detection and segmentation", "Bounding boxes, IoU, NMS, one- vs two-stage detectors, and semantic vs instance segmentation."),
        ("Vision transformers", "Patches as tokens, ViT vs CNN, inductive bias vs data, and the bridge to vision-language models."),
        ("Project: build an image classifier", "End-to-end: load, augment, transfer-learn, evaluate, and dodge the common pitfalls."))},
    {"code": "RL", "title": "Reinforcement Learning Foundations", "slug": "reinforcement-learning",
     "tag": "Deep Learning", "accent": "#10b981", "level": "Intermediate", "hours": "9", "order": 16,
     "foundation": True, "certificate": False,
     "prereqs": ["deep-learning-foundations", "pytorch-essentials"],
     "blurb": "Agents, rewards, and policies from the MDP up to PPO — the RL grounding that RLHF and GRPO "
              "assume but never teach. The bridge from deep learning to how LLMs are aligned.",
     "outcomes": ["Frame a decision problem as an MDP and reason about return, value, and the Bellman equation",
                  "Implement Q-learning, DQN, REINFORCE, and PPO and know when each applies",
                  "Explain advantage, baselines, and GAE, and why PPO's clip makes it the workhorse",
                  "Connect RL to LLMs: the RLHF pipeline, GRPO, RLVR, and KL-to-reference"],
     "syllabus": _syl(
        ("The RL problem", "Agents, rewards, the MDP, return and discounting, exploration vs exploitation."),
        ("Value functions and Bellman", "State/action values, the Bellman equation, policy vs value, DP intuition."),
        ("Q-learning and DQN", "Tabular Q-learning, function approximation, replay buffers and target networks."),
        ("Policy gradients", "Parameterize the policy, the policy-gradient theorem, REINFORCE, variance and baselines."),
        ("Actor-critic and PPO", "Advantage, GAE, actor-critic, and PPO's clipped objective."),
        ("RL for LLMs", "RLHF (reward model + PPO), GRPO, RLVR/verifiable rewards, and KL-to-reference."),
        ("Project: train an agent", "Build a CartPole actor-critic end to end, with the loop and the pitfalls."))},
    {"code": "TS", "title": "Time Series & Forecasting", "slug": "time-series-forecasting",
     "tag": "Machine Learning", "accent": "#0e7490", "level": "Beginner → Intermediate", "hours": "9", "order": 17,
     "foundation": True, "certificate": False,
     "prereqs": ["ml-foundations", "pandas-analysis"],
     "blurb": "Forecast the future honestly: from ARIMA and exponential smoothing to gradient boosting, "
              "deep models, and TS foundation models — backtested the right way.",
     "outcomes": ["Spot trend, seasonality, and stationarity, and split time series without leaking the future",
                  "Forecast with classical (ETS/ARIMA), gradient-boosting, and modern deep/foundation models",
                  "Engineer leak-free lag, rolling, and calendar features and reframe forecasting as regression",
                  "Evaluate honestly with MASE and walk-forward backtesting, and ship calibrated intervals"],
     "syllabus": _syl(
        ("Time series fundamentals", "Autocorrelation, trend/seasonality/cycles, stationarity, and the never-shuffle rule."),
        ("Classical methods", "Moving averages, exponential smoothing (ETS), and ARIMA with ACF/PACF."),
        ("Feature engineering for time series", "Lags, rolling stats, calendar features, and the leakage traps."),
        ("Machine learning for forecasting", "Forecasting as regression; gradient boosting; recursive vs direct multi-step."),
        ("Deep learning for time series", "RNN/LSTM, N-BEATS/N-HiTS, TFT, PatchTST, and TS foundation models."),
        ("Evaluation and backtesting", "MAE/RMSE/MAPE/sMAPE/MASE, rolling/expanding backtests, prediction intervals."),
        ("Project: forecasting retail demand", "Baseline to ML, backtested honestly, with intervals."))},
    {"code": "EXP", "title": "Experimentation & Causal Inference", "slug": "experimentation-causal",
     "tag": "Machine Learning", "accent": "#059669", "level": "Beginner → Intermediate", "hours": "9", "order": 18,
     "foundation": True, "certificate": False,
     "prereqs": ["probability-stats", "ml-foundations"],
     "blurb": "From p-values to product decisions: design and analyze A/B tests, dodge the pitfalls that "
              "fake wins, and estimate causal effects when you can't randomize.",
     "outcomes": ["Design an A/B test end to end: hypothesis, OEC and guardrails, power, and sample size",
                  "Analyze results with the right test and read significance vs practical importance honestly",
                  "Diagnose and defend against peeking, SRM, multiple comparisons, and novelty effects",
                  "Estimate causal effects from observational data and target treatment with uplift models"],
     "syllabus": _syl(
        ("Why experiment?", "Correlation vs causation, counterfactuals, and why randomization is the gold standard."),
        ("Designing an A/B test", "Hypotheses, OEC and guardrail metrics, MDE, power, and sample size."),
        ("Running and analyzing", "Proportion and t-tests, confidence intervals, statistical vs practical significance."),
        ("Pitfalls", "Peeking, multiple comparisons, novelty, SRM, Simpson's paradox; sequential testing and CUPED."),
        ("Causal inference basics", "Confounding, DAGs, the backdoor idea, and when you can't A/B test."),
        ("Observational methods", "Propensity scores, difference-in-differences, instrumental variables, synthetic control."),
        ("Uplift and ML", "Heterogeneous effects, T/S/X-learners, and targeting who to treat."))},
    {"code": "DSA", "title": "DSA for ML Interviews", "slug": "dsa-interview-prep",
     "tag": "Programming", "accent": "#e11d48", "level": "Beginner → Intermediate", "hours": "16", "order": 19,
     "foundation": True, "certificate": False,
     "prereqs": ["python-foundations"],
     "blurb": "The coding-round prep the catalog was missing: data structures, algorithms, and the pattern "
              "playbook that turns O(n²) brute force into O(n) — Python-first, aimed at what ML-engineer "
              "coding interviews actually test.",
     "outcomes": ["Analyze time and space complexity of any loop or recursion, and state it on demand",
                  "Recognize and apply the core patterns: hashing, two pointers, sliding window, BFS/DFS, and DP",
                  "Write clean, idiomatic Python using deque, heapq, Counter, defaultdict, and lru_cache",
                  "Run a coding round end to end with the UMPIRE method — clarify, plan, code, and test out loud"],
     "syllabus": _syl(
        ("Big-O and complexity", "Time/space classes, analyzing loops and recursion, amortized cost, and why it matters."),
        ("Arrays and strings", "List/str internals, slicing cost, in-place ops, prefix sums, and 2D grids."),
        ("Hashing", "dict/set, the seen-set and complement patterns, Counter, and defaultdict — O(n²) into O(n)."),
        ("Two pointers and sliding window", "The two workhorse patterns: pair-sum, longest substring, and when to use each."),
        ("Stacks, queues, and linked lists", "list vs deque, the monotonic stack, list reversal, and Floyd's cycle detection."),
        ("Trees and graphs", "Traversals, DFS/BFS templates, adjacency lists, topological sort, and why ML systems are graphs."),
        ("Recursion and dynamic programming", "Recursion → memoization → tabulation; knapsack, edit distance, LIS, lru_cache."),
        ("Interview strategy", "UMPIRE, communicating under pressure, complexity trade-offs, and what ML coding rounds test."))},
]

# Cloud provider deep-dives: detailed, certificate-eligible, in the Cloud category
# (not part of the 5-step recommended path). Ordered after the core certificate track.
CLOUD = [
    {"code": "AWS", "title": "AWS for ML Engineers", "slug": "aws-for-ml",
     "tag": "Cloud", "accent": "#ff9900", "level": "Beginner → Advanced", "hours": "34", "order": 6,
     "certificate": True, "program": "Specialization", "role": "AWS ML Engineer",
     "prereqs": ["cloud-linux", "docker-containers"],
     "blurb": "Amazon Web Services from first principles to building and shipping ML systems — "
              "IAM, compute, storage, containers, SageMaker AI, and Bedrock.",
     "outcomes": ["Navigate AWS core services (IAM, EC2, S3, VPC) confidently",
                  "Train and deploy models with Amazon SageMaker AI",
                  "Serve models with real-time, serverless, and batch inference",
                  "Architect a production ML system on AWS end to end"]},
    {"code": "AZURE", "title": "Azure for ML Engineers", "slug": "azure-for-ml",
     "tag": "Cloud", "accent": "#0078d4", "level": "Beginner → Advanced", "hours": "34", "order": 7,
     "certificate": True, "program": "Specialization", "role": "Azure ML Engineer",
     "prereqs": ["cloud-linux", "docker-containers"],
     "blurb": "Microsoft Azure end to end for ML — Entra ID, compute, storage, AKS, "
              "Azure Machine Learning, and Azure AI Foundry / OpenAI.",
     "outcomes": ["Work with Azure core services and Entra ID securely",
                  "Train and deploy with Azure Machine Learning",
                  "Use Azure AI Foundry and Azure OpenAI for GenAI systems",
                  "Architect a production ML system on Azure end to end"]},
    {"code": "GCP", "title": "GCP for ML Engineers", "slug": "gcp-for-ml",
     "tag": "Cloud", "accent": "#34a853", "level": "Beginner → Advanced", "hours": "34", "order": 8,
     "certificate": True, "program": "Specialization", "role": "GCP ML Engineer",
     "prereqs": ["cloud-linux", "docker-containers"],
     "blurb": "Google Cloud from fundamentals to production ML — IAM, Compute Engine, "
              "Cloud Storage, BigQuery, GKE, and Vertex AI with Gemini.",
     "outcomes": ["Navigate GCP core services (IAM, Compute Engine, Cloud Storage)",
                  "Use BigQuery and Vertex AI for data and training",
                  "Deploy models and Gemini-based GenAI on Vertex AI",
                  "Architect a production ML system on GCP end to end"]},
    {"code": "K8SML", "title": "Kubernetes for ML", "slug": "kubernetes-for-ml",
     "tag": "Cloud", "accent": "#326ce5", "level": "Intermediate → Advanced", "hours": "16", "order": 9,
     "certificate": True, "program": "Specialization", "role": "ML Platform Engineer",
     "prereqs": ["docker-containers", "cloud-linux"],
     "blurb": "Run ML on Kubernetes end to end — GPUs and MIG, batch queues and gang scheduling, "
              "distributed training, and autoscaling GPU inference. The cloud-agnostic layer under "
              "every serious ML platform.",
     "outcomes": ["Schedule, share, and autoscale GPUs on Kubernetes (device plugin, MIG, time-slicing)",
                  "Run queued, gang-scheduled batch and distributed training (Kueue, Kubeflow Trainer, KubeRay)",
                  "Serve models with autoscaling and canary rollouts (KServe, vLLM)",
                  "Operate a shared GPU cluster with storage, secrets, and GPU-aware observability"],
     "syllabus": _syl(
        ("Kubernetes fundamentals for ML", "Pods, deployments, services, namespaces, kubectl — and why ML lands on k8s."),
        ("GPUs on Kubernetes", "NVIDIA device plugin/GPU Operator, requesting GPUs, MIG and time-slicing, node pools."),
        ("Scheduling and autoscaling", "Requests/limits, HPA, KEDA, Cluster Autoscaler vs Karpenter, scale-to-zero."),
        ("Batch training and queues", "Jobs, Kueue admission/quota, Volcano gang scheduling, fair-share on shared GPUs."),
        ("Distributed training", "Kubeflow Trainer/TrainJob, KubeRay, and multi-node NCCL that actually scales."),
        ("Serving models", "KServe, vLLM and disaggregated serving, canary rollouts, GPU inference autoscaling."),
        ("Storage, data, and observability", "PVCs/CSI, datasets from object storage, secrets, Prometheus/Grafana + DCGM."),
        ("Project: train and serve on k8s", "Capstone: a queued fine-tune to an autoscaling, canaried endpoint."))},
    {"code": "AISEC", "title": "AI Security & Guardrails", "slug": "ai-security",
     "tag": "Cloud", "accent": "#b91c1c", "level": "Intermediate → Advanced", "hours": "10", "order": 10,
     "certificate": True, "program": "Specialization", "role": "AI Security Engineer",
     "prereqs": ["vlm-guide", "apis-web-services"],
     "blurb": "Secure the ML systems everyone else just ships. Prompt injection, jailbreaks, data "
              "poisoning, model-file RCE, guardrail models, agent least-privilege, and red-teaming — "
              "the adversarial layer the flagship GenAI tracks leave out.",
     "outcomes": ["Map any AI system's attack surface to the OWASP LLM Top 10 (2025)",
                  "Defend against prompt injection, jailbreaks, data/model poisoning, and leakage",
                  "Deploy layered guardrails and least-privilege agent architectures that hold when filters fail",
                  "Red-team with PyRIT/garak and ship against NIST AI RMF and EU AI Act obligations"],
     "syllabus": _syl(
        ("The AI threat landscape", "Why ML/LLM systems are uniquely exploitable, the OWASP LLM Top 10, and the attack-surface map."),
        ("Prompt injection", "Direct vs indirect/second-order injection, the lethal trifecta, and why it stays unsolved."),
        ("Jailbreaks and model abuse", "Jailbreaks vs injection, shallow refusal training, the attack taxonomy, and abuse without a jailbreak."),
        ("Data, privacy, and leakage", "Memorization, membership inference, poisoning and backdoors, and model-file supply-chain RCE."),
        ("Guardrails and defenses", "Input/output filtering, Llama Guard and moderation APIs, constrained decoding, and when guardrails fail."),
        ("Securing agents and tools", "Least privilege, sandboxing, human-in-the-loop, MCP security, and dual-LLM/CaMeL isolation patterns."),
        ("Red-teaming and governance", "PyRIT/garak, safety evals, NIST AI RMF, EU AI Act, and a shipping checklist."))},
    {"code": "FT", "title": "Fine-Tuning LLMs in Practice", "slug": "fine-tuning-llms",
     "tag": "Cloud", "accent": "#a855f7", "level": "Intermediate → Advanced", "hours": "12", "order": 11,
     "certificate": True, "program": "Specialization", "role": "LLM Fine-Tuning Engineer",
     "prereqs": ["pytorch-essentials", "vlm-guide"],
     "blurb": "Adapt open LLMs to your task, hands-on: LoRA/QLoRA on one GPU, a real TRL SFT run, DPO "
              "preference tuning, honest evaluation, and serving adapters with vLLM — decision to deployment.",
     "outcomes": ["Decide when to fine-tune vs prompt vs RAG, and prove it with an eval",
                  "Train LoRA/QLoRA adapters with HuggingFace TRL, PEFT, and unsloth",
                  "Go beyond SFT with DPO preference tuning and know when it helps",
                  "Evaluate, quantize with a parity check, and serve adapters on vLLM"],
     "syllabus": _syl(
        ("When to fine-tune (and when not to)", "Fine-tune vs prompt vs RAG decision framework; form not facts; the cost reality."),
        ("Data preparation", "ChatML/ShareGPT formats, chat templates, quality over quantity, splits, synthetic-data caution."),
        ("LoRA and QLoRA", "Full FT vs PEFT, low-rank adapters, 4-bit base, rank/alpha/target-modules, and the memory math."),
        ("The training run", "A real TRL SFT run — hyperparameters, packing, monitoring, overfitting and catastrophic forgetting."),
        ("Preference tuning", "Beyond SFT: DPO (and a nod to GRPO), preference data, DPOTrainer, when it pays off."),
        ("Evaluation", "Did it actually improve? task evals, LLM-as-judge, regression vs base, avoiding contamination."),
        ("Serving and project", "Merge vs keep adapter, vLLM LoRA serving, quantize-after-FT with a parity check, end-to-end project."))},
]

ALL_COURSES = COURSES + CLOUD + FOUNDATIONS
COURSE_BY_SLUG = {c["slug"]: c for c in ALL_COURSES}

# ---------------------------------------------------------------- categories
# Browse-by-category taxonomy across every course. Order = display order.
CATEGORIES = [
    ("Programming & Tools", ["python-foundations", "cli-git", "apis-web-services",
                             "software-engineering-practices", "dsa-interview-prep"]),
    ("Mathematics", ["linear-algebra", "calculus-gradients", "probability-stats"]),
    ("Data & Analytics", ["sql-databases", "pandas-analysis", "data-engineering"]),
    ("Machine Learning", ["ml-foundations", "deep-learning-foundations", "pytorch-essentials",
                          "computer-vision", "reinforcement-learning", "time-series-forecasting",
                          "experimentation-causal"]),
    ("LLMs & Systems", ["vlm-guide", "language-modeling", "fine-tuning-llms", "ai-security",
                        "ml-system-design", "principal-ml-engineer"]),
    ("Cloud & Infrastructure",
     ["cloud-linux", "docker-containers", "kubernetes-for-ml", "aws-for-ml", "azure-for-ml",
      "gcp-for-ml", "mlops"]),
]
# ---------------------------------------------------------------- roadmaps
# Goal-oriented career paths. Each ordered step points at a course slug with a
# one-line reason; steps mix free foundations with paid Nanodegrees. `kind` is
# derived at render time from the course's program/foundation flags, so a step
# only needs (slug, why). `cap` is the slug of the destination Nanodegree.
def _step(slug, why):
    return {"slug": slug, "why": why}


ROADMAPS = [
    {"key": "ml-engineer", "role": "ML Engineer", "accent": "#ea580c",
     "tagline": "Train, evaluate, and ship models that work.",
     "who": "You can code and want to become a hireable machine-learning engineer — "
            "from math foundations to designing real systems.",
     "outcome": "Build and evaluate models end to end and design ML systems for interviews.",
     "cap": "ml-system-design",
     "steps": [
        _step("python-foundations", "The language every ML tool assumes."),
        _step("linear-algebra", "The language tensors are written in."),
        _step("calculus-gradients", "How models actually learn."),
        _step("probability-stats", "The reasoning under every metric and loss."),
        _step("ml-foundations", "Classical ML end to end, and the traps that fool beginners."),
        _step("deep-learning-foundations", "From a neuron to a trained network."),
        _step("pytorch-essentials", "The framework the deep courses assume."),
        _step("software-engineering-practices", "Write ML code that's tested, typed, and maintainable."),
        _step("ml-system-design", "Design end-to-end ML systems like a staff engineer.")]},
    {"key": "genai-engineer", "role": "GenAI Engineer", "accent": "#8b5cf6",
     "tagline": "Build with LLMs, RAG, and agents — then build the models themselves.",
     "who": "You want to work at the frontier: applied generative AI first, then the "
            "internals of the models underneath.",
     "outcome": "Ship RAG and agent systems, and build a modern LLM from scratch.",
     "cap": "language-modeling",
     "steps": [
        _step("python-foundations", "Baseline fluency for everything that follows."),
        _step("ml-foundations", "The ML vocabulary GenAI builds on."),
        _step("pytorch-essentials", "Tensors and training loops you'll need."),
        _step("apis-web-services", "Consume and expose APIs — the backbone of RAG and agents."),
        _step("software-engineering-practices", "Ship tested, maintainable code for real GenAI systems."),
        _step("vlm-guide", "Attention, RAG, and agents — the applied frontier."),
        _step("language-modeling", "Build a modern LLM end to end. The deepest track.")]},
    {"key": "mlops-engineer", "role": "MLOps Engineer", "accent": "#ec4899",
     "tagline": "Take models to production and keep them alive.",
     "who": "You want to own serving, monitoring, and CI/CD for models — the infra "
            "side of ML — with the cloud skills that back it.",
     "outcome": "Serve, monitor, and version models in production on real infrastructure.",
     "cap": "mlops",
     "steps": [
        _step("python-foundations", "Scripting and services start here."),
        _step("cli-git", "The daily workflow of every engineer."),
        _step("software-engineering-practices", "Testing, packaging, logging, and CI — code a team can ship."),
        _step("apis-web-services", "Serve a model behind an endpoint — the skill MLOps builds on."),
        _step("cloud-linux", "Compute, storage, networking, and IAM without the jargon."),
        _step("docker-containers", "The unit of deployment for modern ML."),
        _step("mlops", "Serving, monitoring, CI/CD, and the failure modes nobody warns you about."),
        _step("aws-for-ml", "Optional specialization — do it on AWS end to end.")]},
    {"key": "data-engineer", "role": "Data Engineer", "accent": "#f59e0b",
     "tagline": "Build the data layer every model sits on top of.",
     "who": "You want to build reliable pipelines and warehouses — the foundation "
            "beneath every model and analytics system.",
     "outcome": "Design batch and streaming pipelines and warehouses at scale.",
     "cap": "data-engineering",
     "steps": [
        _step("python-foundations", "The glue language of data work."),
        _step("sql-databases", "Query and model relational data — the backbone."),
        _step("pandas-analysis", "Load, clean, and explore real data."),
        _step("software-engineering-practices", "Pipeline code needs testing, logging, and reproducible envs."),
        _step("data-engineering", "Pipelines, warehouses, and architecture to Fortune-100 scale.")]},
    {"key": "ml-systems-architect", "role": "ML Systems Architect", "accent": "#0ea5e9",
     "tagline": "Design ML systems and the infrastructure they run on.",
     "who": "You're already a solid ML engineer and want to design systems, pass "
            "senior system-design loops, and operate them in production.",
     "outcome": "Architect data platforms, training/serving infra, and pass staff-level interviews.",
     "cap": "ml-system-design",
     "steps": [
        _step("ml-foundations", "Refresh the modeling fundamentals you'll design around."),
        _step("mlops", "Production mechanics: serving, monitoring, CI/CD."),
        _step("ml-system-design", "Frame any 'Design X' problem and defend the tradeoffs."),
        _step("data-engineering", "The data platform every ML system depends on.")]},
    {"key": "principal-ml-engineer", "role": "Principal ML Engineer", "accent": "#dc2626",
     "tagline": "Operate at org scale with the judgment of 15+ years.",
     "who": "You're a senior/staff ML engineer aiming for principal (L7+/E7+): strategy, "
            "platform architecture, economics, governance, and influence.",
     "outcome": "Write strategy, make defensible build-vs-buy calls, and pass principal loops.",
     "cap": "principal-ml-engineer",
     "steps": [
        _step("ml-system-design", "The system-design foundation this builds above."),
        _step("mlops", "Production and reliability at the level a principal owns."),
        _step("principal-ml-engineer", "The senior-to-principal delta, compressed.")]},
]
ROADMAP_BY_KEY = {r["key"]: r for r in ROADMAPS}

CATEGORY_OF = {slug: name for name, slugs in CATEGORIES for slug in slugs}

CONTENT_DIR = Path(__file__).parent / "content"


def _title_of(md_path):
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return md_path.stem.replace("_", " ").replace("-", " ").title()


def load_modules():
    mods = {}
    for c in ALL_COURSES:
        folder = CONTENT_DIR / c["slug"]
        items = []
        if folder.is_dir():
            for p in sorted(folder.glob("*.md")):
                items.append({"id": p.stem, "file": p.name, "title": _title_of(p)})
        mods[c["slug"]] = items
    return mods


MODULES = load_modules()

MD_EXT = ["fenced_code", "tables", "sane_lists", "toc", "attr_list"]


def render_markdown(slug, module_id):
    path = CONTENT_DIR / slug / f"{module_id}.md"
    if not path.is_file():
        return None
    return md.markdown(path.read_text(encoding="utf-8"), extensions=MD_EXT)


def course_view(slug):
    """Meta dict augmented with its module list and count."""
    c = COURSE_BY_SLUG.get(slug)
    if not c:
        return None
    v = dict(c)
    v["modules"] = MODULES.get(slug, [])
    v["certificate"] = c.get("certificate", True)
    v["foundation"] = c.get("foundation", False)
    v["syllabus"] = c.get("syllabus", [])
    v["category"] = CATEGORY_OF.get(slug, "Other")
    v["has_content"] = bool(v["modules"])
    v["module_count"] = len(v["modules"]) if v["modules"] else len(v["syllabus"])
    v["program"] = c.get("program")            # "Nanodegree" | "Specialization" | None
    v["role"] = c.get("role")
    v["prereqs"] = [_prereq_ref(s) for s in c.get("prereqs", [])]
    return v


def _prereq_ref(slug):
    """Light reference to a prerequisite course, for linking from a course/roadmap."""
    c = COURSE_BY_SLUG.get(slug)
    if not c:
        return {"slug": slug, "title": slug, "foundation": True, "program": None}
    return {"slug": slug, "title": c["title"],
            "foundation": c.get("foundation", False), "program": c.get("program")}


def roadmap_view(r):
    """Roadmap dict with each step resolved to a full course_view + kind badge."""
    steps = []
    for i, s in enumerate(r["steps"], 1):
        cv = course_view(s["slug"])
        if not cv:
            continue
        kind = "Free" if cv["foundation"] else (cv.get("program") or "Certificate")
        steps.append({"n": i, "why": s["why"], "kind": kind, "course": cv})
    v = dict(r)
    v["steps"] = steps
    v["n_steps"] = len(steps)
    v["hours"] = sum(int(s["course"].get("hours", 0) or 0) for s in steps)
    v["cap_course"] = course_view(r["cap"]) if r.get("cap") else None
    return v


# ---------------------------------------------------------------- gamification
# Milestone badges. Kept professional (skills/mastery), not childish. Criteria are
# recomputed from live state on each completion; ids are stored on the user doc.
BADGES = {
    "first-steps":     {"title": "First Steps",   "icon": "🌱", "desc": "Completed your first lesson"},
    "getting-serious": {"title": "Getting Serious","icon": "📚", "desc": "Completed 10 lessons"},
    "scholar":         {"title": "Scholar",        "icon": "🎓", "desc": "Completed 50 lessons"},
    "streak-7":        {"title": "On a Roll",      "icon": "🔥", "desc": "7-day learning streak"},
    "streak-30":       {"title": "Unstoppable",    "icon": "⚡", "desc": "30-day learning streak"},
    "xp-500":          {"title": "Rising",         "icon": "⭐", "desc": "Earned 500 XP"},
    "xp-2500":         {"title": "Elite",          "icon": "💎", "desc": "Earned 2,500 XP"},
    "explorer":        {"title": "Explorer",       "icon": "🧭", "desc": "Active in 3+ courses"},
    "finisher":        {"title": "Finisher",       "icon": "🏁", "desc": "Completed a full course"},
}
XP_PER_LESSON = 50
XP_COURSE_BONUS = 200


def _earned_badges(uid):
    """Recompute the full set of badge ids the user currently qualifies for."""
    stats = data.get_user_stats(uid)
    total_done, courses_touched, any_course_done = 0, 0, False
    for e in data.list_enrollments(uid):
        done = set(getattr(e, "completed", []) or [])
        total_done += len(done)
        if done:
            courses_touched += 1
        total = len(MODULES.get(e.course_slug, []))
        if total and len(done) >= total:
            any_course_done = True
    earned = set()
    if total_done >= 1:  earned.add("first-steps")
    if total_done >= 10: earned.add("getting-serious")
    if total_done >= 50: earned.add("scholar")
    if max(stats["streak"], stats["longest_streak"]) >= 7:  earned.add("streak-7")
    if stats["longest_streak"] >= 30: earned.add("streak-30")
    if stats["xp"] >= 500:  earned.add("xp-500")
    if stats["xp"] >= 2500: earned.add("xp-2500")
    if courses_touched >= 3: earned.add("explorer")
    if any_course_done: earned.add("finisher")
    return earned


# ---------------------------------------------------------------- auth
def current_user():
    uid = session.get("uid")
    return data.get_user(uid) if uid else None


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not current_user():
            return redirect(url_for("login", next=request.path))
        return f(*a, **kw)
    return wrapper


def enrollment_of(user, slug):
    if not user:
        return None
    return data.get_enrollment(user.id, slug)


def logged_in():
    return session.get("admin") is True


@app.context_processor
def inject_globals():
    return {"BRAND": BRAND, "PASS_MARK": PASS_MARK, "YEAR": datetime.utcnow().year,
            "user": current_user()}


# ---------------------------------------------------------------- helpers
BRAND_CODE = "AMLA"


def course_slug_code(title):
    s = "".join(c for c in title.upper() if c.isalnum())
    return s[:5] or "CERT"


def mint_code(course, year):
    for _ in range(30):
        rnd = secrets.token_hex(2).upper()
        code = f"{BRAND_CODE}-{course_slug_code(course)}-{year}-{rnd}"
        if not data.certificate_exists(code):
            return code
    return f"{BRAND_CODE}-{course_slug_code(course)}-{year}-{secrets.token_hex(3).upper()}"


# ---------------------------------------------------------------- public
@app.route("/")
def index():
    nanodegrees = sorted((course_view(c["slug"]) for c in COURSES),
                         key=lambda c: c["order"])
    specializations = sorted((course_view(c["slug"]) for c in CLOUD),
                             key=lambda c: c["order"])
    foundations = sorted((course_view(c["slug"]) for c in FOUNDATIONS),
                         key=lambda c: c["order"])
    grouped = [(name, [course_view(s) for s in slugs]) for name, slugs in CATEGORIES]
    roadmaps = [roadmap_view(r) for r in ROADMAPS]
    total_modules = sum(len(MODULES.get(c["slug"], [])) for c in ALL_COURSES)
    return render_template("index.html", courses=nanodegrees, nanodegrees=nanodegrees,
                           specializations=specializations, foundations=foundations,
                           roadmaps=roadmaps, grouped=grouped,
                           n_courses=len(ALL_COURSES), total_modules=total_modules)


@app.route("/roadmaps")
def roadmaps():
    views = [roadmap_view(r) for r in ROADMAPS]
    return render_template("roadmaps.html", roadmaps=views)


@app.route("/course/<slug>")
def course_detail(slug):
    c = course_view(slug)
    if not c:
        abort(404)
    user = current_user()
    enr = enrollment_of(user, slug)
    done = data.completed_modules(user.id, slug) if (user and enr) else set()
    total = len(c["modules"])
    progress = {"done": len(done), "total": total,
                "pct": int(len(done) * 100 / total) if total else 0}
    return render_template("course_detail.html", course=c, enrollment=enr,
                           completed=list(done), progress=progress)


@app.route("/course/<slug>/enroll", methods=["POST"])
@login_required
def enroll(slug):
    c = course_view(slug)
    if not c:
        abort(404)
    user = current_user()
    if not enrollment_of(user, slug):
        data.create_enrollment(user.id, slug)
        flash(f"You're enrolled in {c['title']}.")
    first = c["modules"][0]["id"] if c["modules"] else None
    if first:
        return redirect(url_for("module", slug=slug, module_id=first))
    flash("Lessons for this course are coming soon — you'll find them here when ready.")
    return redirect(url_for("course_detail", slug=slug))


@app.route("/course/<slug>/<module_id>")
@login_required
def module(slug, module_id):
    c = course_view(slug)
    if not c:
        abort(404)
    user = current_user()
    enr = enrollment_of(user, slug)
    if not enr:
        flash("Enroll in this course to read its modules.")
        return redirect(url_for("course_detail", slug=slug))
    ids = [m["id"] for m in c["modules"]]
    if module_id not in ids:
        abort(404)
    html = render_markdown(slug, module_id)
    idx = ids.index(module_id)
    prev_m = c["modules"][idx - 1] if idx > 0 else None
    next_m = c["modules"][idx + 1] if idx < len(ids) - 1 else None
    # remember last position for "continue"
    data.set_last_module(user.id, slug, module_id)
    note = data.get_note(user.id, slug, module_id)
    done = data.completed_modules(user.id, slug)
    return render_template("module.html", course=c, current=c["modules"][idx],
                           body=html, prev_m=prev_m, next_m=next_m, index=idx,
                           note_body=(note.body if note else ""),
                           note_highlights=json.dumps(note.highlights if note else []),
                           completed=list(done), is_done=(module_id in done),
                           done_count=len(done), total_count=len(c["modules"]))


def _module_guard(user, slug, module_id):
    if not enrollment_of(user, slug):
        abort(403)
    if slug not in COURSE_BY_SLUG or module_id not in [m["id"] for m in MODULES.get(slug, [])]:
        abort(404)


@app.route("/course/<slug>/<module_id>/note", methods=["POST"])
@login_required
def save_note(slug, module_id):
    user = current_user()
    _module_guard(user, slug, module_id)
    payload = request.get_json(silent=True) or {}
    data.save_note_body(user.id, slug, module_id, (payload.get("body") or "")[:20000])
    return {"ok": True}


@app.route("/course/<slug>/<module_id>/highlights", methods=["POST"])
@login_required
def save_highlights(slug, module_id):
    user = current_user()
    _module_guard(user, slug, module_id)
    payload = request.get_json(silent=True) or {}
    items = payload.get("highlights")
    if not isinstance(items, list):
        return {"ok": False, "error": "highlights must be a list"}, 400
    # keep only the fields we expect, cap size defensively
    clean = []
    for h in items[:500]:
        if not isinstance(h, dict):
            continue
        clean.append({
            "id": str(h.get("id", ""))[:40],
            "start": int(h.get("start", 0)),
            "end": int(h.get("end", 0)),
            "text": str(h.get("text", ""))[:2000],
            "color": (h.get("color") if h.get("color") in ("y", "g", "b", "p") else "y"),
            "note": str(h.get("note", ""))[:4000],
        })
    data.save_highlights(user.id, slug, module_id, clean)
    return {"ok": True, "count": len(clean)}


@app.route("/course/<slug>/<module_id>/complete", methods=["POST"])
@login_required
def complete_module(slug, module_id):
    user = current_user()
    _module_guard(user, slug, module_id)
    newly = data.mark_module_complete(user.id, slug, module_id)
    done = data.completed_modules(user.id, slug)
    total = len(MODULES.get(slug, []))
    course_completed = bool(total and len(done) >= total)
    resp = {"ok": True, "newly": newly, "course_done": len(done),
            "course_total": total, "course_completed": course_completed, "new_badges": []}
    if newly:
        gain = XP_PER_LESSON + (XP_COURSE_BONUS if course_completed else 0)
        stats = data.award_activity(user.id, gain)
        before = set(data.get_user_stats(user.id)["badges"])
        new = _earned_badges(user.id) - before
        data.add_badges(user.id, new)
        resp.update({"xp": stats["xp"], "streak": stats["streak"], "xp_gained": gain,
                     "new_badges": [dict(id=b, **BADGES[b]) for b in new if b in BADGES]})
    else:
        st = data.get_user_stats(user.id)
        resp.update({"xp": st["xp"], "streak": st["streak"], "xp_gained": 0})
    return resp


@app.route("/leaderboard")
@login_required
def leaderboard():
    user = current_user()
    rows = data.leaderboard(25)
    board = [{"rank": i + 1, "name": getattr(r, "name", "Learner"),
              "xp": getattr(r, "xp", 0) or 0, "is_me": r.id == user.id}
             for i, r in enumerate(rows)]
    my_xp = getattr(user, "xp", 0) or 0
    return render_template("leaderboard.html", board=board, my_xp=my_xp,
                           in_board=any(b["is_me"] for b in board))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    stats = data.get_user_stats(user.id)
    counts = data.completed_counts(user.id)
    enrolled = []
    for e in data.list_enrollments(user.id):
        c = course_view(e.course_slug)
        if not c:
            continue
        resume = e.last_module or (c["modules"][0]["id"] if c["modules"] else None)
        total = len(c["modules"])
        dn = counts.get(e.course_slug, 0)
        enrolled.append({"course": c, "enrollment": e, "resume": resume,
                         "done": dn, "total": total,
                         "pct": int(dn * 100 / total) if total else 0})
    badges = [dict(id=b, **BADGES[b]) for b in stats["badges"] if b in BADGES]
    return render_template("dashboard.html", enrolled=enrolled, stats=stats, badges=badges)


# ---------------------------------------------------------------- capstone
URL_RE = re.compile(r"^https?://[^\s]+\.[^\s]+$")


def _parse_artifact_urls(raw):
    """One URL per line; keep well-formed ones, cap count and length."""
    out = []
    for line in (raw or "").splitlines():
        u = line.strip()
        if u and URL_RE.match(u):
            out.append(u[:400])
        if len(out) >= 10:
            break
    return out


@app.route("/course/<slug>/capstone", methods=["GET", "POST"])
@login_required
def capstone(slug):
    c = course_view(slug)
    if not c:
        abort(404)
    if not c["certificate"]:
        flash("Open foundation courses don't have a capstone.")
        return redirect(url_for("course_detail", slug=slug))
    user = current_user()
    if not enrollment_of(user, slug):
        flash("Enroll in this course before submitting a capstone.")
        return redirect(url_for("course_detail", slug=slug))
    existing = data.get_capstone(user.id, slug)

    if request.method == "POST":
        if existing and existing.status in ("submitted", "under_review", "passed"):
            flash("Your capstone is already " + existing.status
                  + " — you can't resubmit right now.")
            return redirect(url_for("capstone", slug=slug))
        repo_url = request.form.get("repo_url", "").strip()
        summary = request.form.get("summary", "").strip()
        artifacts = _parse_artifact_urls(request.form.get("artifact_urls", ""))
        if not URL_RE.match(repo_url):
            flash("Enter a valid project URL (starting with http:// or https://).")
            return render_template("capstone.html", course=c, capstone=existing,
                                   repo_url=repo_url, summary=summary)
        if len(summary) < 50:
            flash("Write a summary of at least 50 characters describing what you built.")
            return render_template("capstone.html", course=c, capstone=existing,
                                   repo_url=repo_url, summary=summary)
        data.submit_capstone(user.id, slug, repo_url[:400], summary[:5000], artifacts)
        flash("Capstone submitted. It's now in review.")
        return redirect(url_for("capstone", slug=slug))

    return render_template("capstone.html", course=c, capstone=existing,
                           repo_url="", summary="")


# ---------------------------------------------------------------- student auth
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        if not name or not EMAIL_RE.match(email) or len(pw) < 8:
            flash("Enter a name, a valid email, and a password of at least 8 characters.")
            return render_template("register.html", name=name, email=email)
        if data.get_user_by_email(email):
            flash("That email is already registered. Try logging in.")
            return render_template("register.html", name=name, email=email)
        uid = data.create_user(email, name, generate_password_hash(pw))
        session["uid"] = uid
        nxt = request.args.get("next")
        return redirect(nxt or url_for("dashboard"))
    return render_template("register.html", name="", email="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        u = data.get_user_by_email(email)
        if u and check_password_hash(u.password_hash, pw):
            session["uid"] = u.id
            nxt = request.args.get("next")
            return redirect(nxt or url_for("dashboard"))
        flash("Wrong email or password.")
        return render_template("login.html", email=email)
    return render_template("login.html", email="")


@app.route("/logout")
def logout():
    session.pop("uid", None)
    return redirect(url_for("index"))


# ---------------------------------------------------------------- verify
@app.route("/verify", methods=["GET"])
def verify():
    return render_template("verify.html", code="", result=None)


@app.route("/verify/<code>")
@app.route("/c/<code>")
def verify_code(code):
    code = code.strip().upper()
    cert = data.get_certificate(code)
    if not cert:
        result = {"status": "invalid"}
    elif cert.revoked:
        result = {"status": "revoked", "cert": cert}
    else:
        result = {"status": "valid", "cert": cert}
    return render_template("verify.html", code=code, result=result)


@app.route("/certificate/<code>")
def certificate(code):
    cert = data.get_certificate(code.strip().upper())
    if not cert:
        abort(404)
    verify_url = request.host_url.rstrip("/") + url_for("verify_code", code=cert.code)
    return render_template(
        "certificate.html", cert=cert, verify_url=verify_url,
        brand=BRAND, monogram=MONOGRAM, est=EST_YEAR, program=PROGRAM_LINE,
        instructor=INSTRUCTOR, instructor_title=INSTRUCTOR_TITLE,
        cosigner=COSIGNER, cosigner_title=COSIGNER_TITLE,
        art_bg=art.bg_svg(), art_seal=art.seal_svg(BRAND),
        art_corner=art.corner_svg(), art_crest=art.crest_svg(MONOGRAM),
    )


# ---------------------------------------------------------------- admin
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        pw = request.form.get("password", "")
        if hmac.compare_digest(pw, ADMIN_PASSWORD):
            session["admin"] = True
            return redirect(url_for("admin"))
        flash("Wrong password. Try again.")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("index"))


@app.route("/admin")
def admin():
    if not logged_in():
        return redirect(url_for("admin_login"))
    certs = data.list_certificates()
    valid = sum(1 for c in certs if not c.revoked)
    default_pw = ADMIN_PASSWORD == "change-me"
    return render_template("admin.html", certs=certs, valid=valid,
                           courses=COURSES, today=date.today().isoformat(),
                           default_pw=default_pw,
                           students=data.count_users(),
                           enrollments=data.count_enrollments())


@app.route("/admin/issue", methods=["POST"])
def admin_issue():
    if not logged_in():
        abort(403)
    name = request.form.get("name", "").strip()
    course = request.form.get("course", "").strip()
    hours = request.form.get("hours", "").strip()
    issued_on = request.form.get("issued_on", "").strip() or date.today().isoformat()
    try:
        score = int(request.form.get("score", ""))
    except ValueError:
        score = -1

    if not name or not course or score < 0:
        flash("Fill in a name, a course, and a score.")
        return redirect(url_for("admin"))
    if score < PASS_MARK:
        flash(f"Score {score}% is below the {PASS_MARK}% pass mark. "
              f"No certificate issued.")
        return redirect(url_for("admin"))

    year = issued_on[:4]
    code = mint_code(course, year)
    data.create_certificate(code, name, course, score, hours, issued_on)
    flash(f"Issued {code} to {name}.")
    return redirect(url_for("admin"))


@app.route("/admin/toggle/<code>", methods=["POST"])
def admin_toggle(code):
    if not logged_in():
        abort(403)
    cert = data.toggle_certificate(code)
    if not cert:
        abort(404)
    flash(("Revoked " if cert.revoked else "Restored ") + code + ".")
    return redirect(url_for("admin"))


@app.route("/admin/capstones")
def admin_capstones():
    if not logged_in():
        return redirect(url_for("admin_login"))
    rows = []
    for cap in data.list_capstones():
        c = COURSE_BY_SLUG.get(cap.course_slug)
        author = data.get_user(cap.user_id)
        rows.append({"cap": cap,
                     "course_title": c["title"] if c else cap.course_slug,
                     "course_hours": (c or {}).get("hours", ""),
                     "author_name": author.name if author else "(unknown)",
                     "author_email": author.email if author else ""})
    return render_template("admin_capstones.html", rows=rows,
                           today=date.today().isoformat())


@app.route("/admin/capstones/<uid>/<slug>/decide", methods=["POST"])
def admin_capstone_decide(uid, slug):
    if not logged_in():
        abort(403)
    cap = data.get_capstone(uid, slug)
    if not cap:
        abort(404)
    action = request.form.get("action", "")
    c = COURSE_BY_SLUG.get(slug)
    if action == "pass":
        try:
            score = int(request.form.get("score", ""))
        except ValueError:
            score = -1
        if score < PASS_MARK:
            flash(f"Score {score}% is below the {PASS_MARK}% pass mark — not passing.")
            return redirect(url_for("admin_capstones"))
        author = data.get_user(uid)
        title = c["title"] if c else slug
        year = date.today().isoformat()[:4]
        code = mint_code(title, year)
        data.create_certificate(code, author.name if author else "(unknown)", title,
                                 score, (c or {}).get("hours", ""),
                                 date.today().isoformat(), kind="course")
        data.decide_capstone(uid, slug, "passed", score=score, verdict="pass")
        flash(f"Passed capstone and issued {code}.")
    elif action == "fail":
        data.decide_capstone(uid, slug, "failed", verdict="fail")
        flash("Capstone marked failed — the learner can revise and resubmit.")
    elif action == "review":
        data.decide_capstone(uid, slug, "under_review", verdict=None)
        flash("Marked under review.")
    return redirect(url_for("admin_capstones"))


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
