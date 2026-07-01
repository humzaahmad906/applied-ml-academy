# 10 — Deep Learning Track: CV and NLP Adjustments

The foundational chapters frame examples with tabular data — Bike Sharing, Adult Income, fraud transactions — because tabular is the cheapest path to learning MLOps primitives without GPUs. If your background is computer vision or NLP and you're working with PyTorch + Transformers + image data daily, this chapter remaps the curriculum so you don't waste cycles on irrelevant tooling and you study the *right* parts harder.

**Read this before you start any other chapter.** It changes what you build, what you skip, and what you study extra.

## Who This Chapter Is For

You're a CV or NLP engineer who already:

- Trains models in PyTorch (or JAX). You know `nn.Module`, autograd, dataloaders.
- Uses Hugging Face `transformers` / `datasets` / `tokenizers` and / or `torchvision`, `timm`, `mmdetection`, `mmsegmentation`.
- Has trained at least one Transformer or a CNN past ImageNet pretrain.
- Wants to do MLOps the way it's done at companies that ship DL into prod — not the way classical ML shops do it.

If that's you, the standard MLOps curriculum has three biases to correct for:

1. **Tabular-centric.** XGBoost-flavored "feature store + scikit-learn registry" is a small slice of what DL teams actually do.
2. **CPU-friendly.** A DL pipeline lives or dies on GPU economics; tabular ignores this.
3. **Under-weights certain topics.** Distributed training, KV cache, tokenizer parity, embedding versioning, image-pipeline determinism — barely covered in the base material.

This chapter fixes that.

---

## Mapping the Base Curriculum to a DL Worldview

### Tier 1 — Beginner Project Replacement

The beginner project in the foundational track is a tabular income classifier. For you, swap it for **one** of these:

#### Option A — Image classifier

- **Dataset:** Food-101 (10K-class manageable), or Stanford Cars, or a multi-class chest X-ray subset (MedMNIST), or a domain-specific image set from Hugging Face Datasets / Kaggle.
- **Model:** Start with a `timm` pretrained ResNet-50 or ConvNeXt-Tiny. Fine-tune the last block + classifier head. Move to a ViT-B/16 once the pipeline works.
- **Service:** FastAPI endpoint accepting a multipart image upload, returning class + probability + Grad-CAM heatmap link.

#### Option B — Text classifier / extractor

- **Dataset:** AG News, IMDB, Banking77 intent classification, or a small NER dataset (CoNLL-2003).
- **Model:** Fine-tune `distilbert-base-uncased` or `roberta-base` via Hugging Face Trainer. Move to a 1B-class model with LoRA once the pipeline works.
- **Service:** FastAPI endpoint accepting JSON text, returning predicted labels + token-level entity spans.

#### Option C — Embedding service

- Fine-tune a sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`) on a domain similarity task with contrastive loss.
- Service returns a vector; deploy alongside a vector DB (pgvector locally).
- Sets you up perfectly for the LLMOps modules later — RAG infra is your tier-2 extension.

**Why these instead of tabular:**

- Forces you to handle binary blob inputs (images, raw text) — different I/O contract than CSVs.
- Brings tokenizer / preprocessing parity into focus (the DL-flavored training-serving skew).
- Requires GPU at some point (even if just for a Colab training run).
- Produces a portfolio repo that signals "I do DL prod work" — not "I followed a generic MLOps tutorial."

### Things in the Tier 1 Material That Still Apply Verbatim

- `uv` + lock files + project structure (covered in Week 1 of the beginner track)
- Random seeds — *but* see "Determinism in DL is harder" below
- MLflow tracking, signatures, Model Registry — all still apply
- DVC for the dataset (yes, even a 30GB image dataset) — DVC handles it via S3
- Docker multi-stage builds, FastAPI + Pydantic, `/health` + `/ready` + Prometheus metrics
- GitHub Actions CI

### Things in Tier 1 to *De-emphasize*

- `sklearn` model packaging — you'll use `mlflow.pytorch` or `mlflow.transformers` instead of `mlflow.sklearn`
- Optuna sweep on `n_estimators` / `max_depth` — replace with sweeping LR / weight decay / dropout / warmup steps for a Transformer
- The "Bike Sharing" or "Adult Income" example datasets — pick a DL dataset

### Things in Tier 1 You Should Study *Harder*

#### Determinism in DL is harder than in sklearn

Set seeds and *also*:

```python
import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Determinism in CUDA — trades speed for repeatability
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Note `torch.use_deterministic_algorithms(True)` will *raise* on ops with no deterministic implementation. Use `warn_only=True` for training; flip to strict for unit tests that must be bit-reproducible.

You will *still* get tiny differences across GPU generations (A100 vs H100) for the same code + data + seed. Document this in your README.

#### Preprocessing parity (DL's training-serving skew)

The single biggest source of "the model works in eval, fails in prod" for DL is not Redis vs warehouse — it's preprocessing drift:

- **Vision:** the resize-and-normalize pipeline. `transforms.Resize(256)` followed by `CenterCrop(224)` and `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` at training time vs. a hand-rolled PIL resize at inference time.
- **NLP:** the tokenizer. A different `max_length`, `padding`, `truncation`, or `add_special_tokens` setting between training and serving silently kills accuracy.

The fix:

1. **Ship the preprocessing with the model.** `mlflow.transformers.log_model(...)` packages the tokenizer alongside the model — use that, not a hand-rolled tokenizer call at serve time.
2. **For vision, define a single `transforms.Compose(...)` and use it from both train and serve.** Don't write a separate "production" pipeline.
3. **Write a parity test.** Pick 10 fixed inputs; assert that the train-time preprocessor output and the serve-time preprocessor output are bit-identical. Put it in CI.

### Tier 2 — Medium Project Replacement

The intermediate track builds a feature store + orchestrated retraining + monitoring on top of the tabular model. For a DL engineer, the analogous tier 2 is:

#### CV variant — Production image classifier with drift

- Re-train on a sliding window of new labeled images weekly via Prefect / Airflow.
- "Feature pipeline" becomes a **deterministic preprocessing pipeline** versioned in Git and packaged with the model.
- Monitoring: input image statistics (brightness, blur, resolution distribution), prediction distribution per class, per-class precision/recall on a labeled audit subset.
- Drift: PSI on derived image statistics, plus *embedding drift* — average cosine distance between current-week embeddings and reference-week embeddings.

#### NLP variant — Production text classifier with drift

- Re-train weekly on newly-labeled customer support tickets.
- Tokenizer is the feature pipeline; pin its version explicitly in the registry alongside the model.
- Monitoring: token length distribution, OOV rate (or for BPE, fraction of `[UNK]` tokens), prediction distribution, per-class accuracy on a held-out audit.
- Drift: PSI on token length and prediction confidence; semantic drift via embedding-distance between current and reference samples.

#### What still applies from the intermediate track

- Prefect / Airflow / Dagster orchestration
- The four ML test categories (unit, data, model, integration) — same shape, DL examples instead
- Model registry with alias promotion (`@champion`, `@challenger`)
- CI/CD/CT pipelines
- Monitoring architecture diagram

#### What to swap

- **Feast** is overkill for most pure-DL projects. The model itself is the artifact; preprocessing is in code. You may still want Feast or an online store for tabular signals that *augment* a DL model (e.g., user history as side features for a vision recommender). When you don't have those, skip Feast — say so in the README.
- **Drift detection on individual scalar features** gets replaced by **embedding drift** and **input-statistic drift**. PSI is still the workhorse; you just compute it on different things.

### Tier 3 — Where the Curriculum Starts Matching Your World

The advanced track is already DL-heavy: DDP, FSDP, FlashAttention, KServe + Triton, vLLM, KV cache. Follow it directly. Pay extra attention to:

- **GPU memory math** — get the per-layer breakdown for *your* architecture (Transformer for NLP; CNN or ViT for CV) on a real GPU.
- **`torch.compile`** — DL teams ignore it and leave 1.5–3x throughput on the floor.
- **TensorRT for CV inference** — for any deployed image model serving > 10 RPS, exporting to ONNX → TensorRT is the standard production step. Convert one of your models; benchmark P95 latency before/after.

### Specialization Track — What's High-Leverage for You

The LLMOps phase is essential. The vector DB phase is essential. The cloud-platform phase matters less for DL — most DL teams build on raw K8s + open infra rather than SageMaker / Vertex managed offerings. Still study one cloud's ML platform enough to talk fluently about it in interviews.

### Capstone Projects — Pick DL-Native Ones

Of the seven capstone projects in the projects track:

- **Project 1 (real-time anomaly detection)** — Tabular-leaning; skip unless you want fraud experience.
- **Project 2 (LLM platform with RAG)** — **Do this one.** Directly maps to your skill set.
- **Project 3 (feature store)** — Skip; not DL-shaped.
- **Project 4 (cost crime scene)** — Do this one; cost optimization for GPU inference is your bread and butter.
- **Project 5 (distributed fine-tuning platform)** — **Do this one if your bias is NLP / LLMs.** This is the F50-LLM-team interview-anchor project.
- **Project 6 (online inference platform with SLOs)** — **Do this one if your bias is CV serving at scale or LLM serving.** Excellent project for a "frontend of the model" senior role.
- **Project 7 (federated platform with model contracts)** — Specialized; pick only if you target an enterprise ML architect role.

A solid two-project portfolio for a senior DL-MLOps role:

- **Path A (LLM-leaning):** Project 2 + Project 5
- **Path B (CV-serving-leaning):** Project 6 (with a CV model) + Project 4
- **Path C (general DL platform):** Project 2 + Project 6

---

## DL-Specific Topics to Study That the Base Curriculum Underweights

### 1. Tokenizer Hygiene (NLP)

- The same string can tokenize differently between the trainer's tokenizer and the server's tokenizer if anyone (a) loads the wrong revision from HF Hub, (b) calls the slow tokenizer on the trainer and the fast tokenizer on the server, (c) uses different `add_special_tokens` or `padding_side`.
- Always pin the tokenizer to a specific revision hash (not a tag). Always check `tokenizer.is_fast`.
- For long-context models, watch the truncation strategy. `truncation="only_second"` vs `True` will silently change what your model sees.

### 2. Image Preprocessing Hygiene (CV)

- Bilinear vs bicubic interpolation between training and inference changes pixel values enough to move predictions. Pin the interpolation mode.
- PIL vs OpenCV vs torchvision tensor pipelines often produce slightly different output for the "same" operation. Pick one and stay in it.
- Watch the `.convert("RGB")` step. Some upstream image pipelines emit RGBA, BGR, or grayscale; if your model expects RGB and you skip the convert, you get silent garbage.
- Augmentations (`RandAugment`, `AutoAugment`, MixUp, CutMix) belong in *training* only. Asserting "no augmentation transforms in the inference pipeline" should be a unit test.

### 3. Mixed-Precision Subtleties

- BF16 is now the right default on Ampere+ (A100, H100, RTX 30xx+). FP16 + GradScaler is legacy; you'll still see it in older codebases.
- For LayerNorm, RMSNorm, softmax, loss accumulation: keep these in FP32 even under BF16 autocast. PyTorch's `autocast` handles most of this automatically.
- Activation functions: GeLU, SiLU work fine in BF16. ReLU on huge magnitudes can lose precision; rare in practice.

### 4. FlashAttention and Modern Attention Kernels

- For any Transformer training or inference, enable FlashAttention. Hugging Face: `attn_implementation="flash_attention_2"`. PyTorch native: `torch.nn.functional.scaled_dot_product_attention` dispatches to FA2 when available.
- FlashAttention-3 (Hopper / Blackwell only) requires specific kernel builds; check your install.
- xFormers' `memory_efficient_attention` predates FA2 and is largely superseded; you may still see it in CV codebases (Stable Diffusion-derived).

### 5. Distributed Training You'll Actually Do

- **DDP** for any multi-GPU training where the model fits on a single GPU. Default.
- **FSDP** (`ShardingStrategy.FULL_SHARD`) for any model > ~3B parameters on consumer hardware, or > ~7B on A100-80GB. Standard for LLM fine-tuning.
- **DeepSpeed ZeRO-3** is FSDP's older cousin. You'll see it in HF Trainer configs; the two are interchangeable for most workflows.
- **Tensor parallel + pipeline parallel** show up only for > 70B models. For most DL engineers, FSDP + Accelerate is enough.

### 6. LoRA / QLoRA / Adapter Training

For NLP fine-tuning, the 2026 default is:

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()  # typically ~0.5% of base
```

You're training 0.1–1% of params; full fine-tuning of a 7B base on 1×A100-80GB becomes feasible. Combine with 4-bit quantization (QLoRA via `bitsandbytes`) to fit on 24 GB.

For CV, parameter-efficient fine-tuning isn't yet as dominant as in NLP; full fine-tuning of the last few blocks is still common. PEFT's adapter support for ViTs is improving — worth tracking.

### 7. Model Export Paths

| Source | Target | Tool | When |
|---|---|---|---|
| PyTorch CV model | ONNX | `torch.onnx.export` (legacy) or `torch.onnx.dynamo_export` (modern) | Cross-runtime serving |
| PyTorch | TensorRT | `torch-tensorrt` or ONNX → `trtexec` | NVIDIA GPU serving at lowest latency |
| PyTorch LLM | vLLM | Direct load via HF format | Default LLM serving |
| PyTorch LLM | TensorRT-LLM | NVIDIA's compiler | Highest-throughput LLM serving on NVIDIA |
| PyTorch | CoreML | `coremltools` | iOS / macOS edge |
| PyTorch | TFLite | ONNX → TFLite | Android edge |
| PyTorch LLM | GGUF | `llama.cpp` converters | CPU / consumer-GPU inference |
| PyTorch | Executorch | `executorch` | PyTorch's edge path |
| PyTorch | MLX | `mlx-lm` converters | Apple Silicon native |

Most production CV at scale flows through ONNX → TensorRT. Most production LLM at scale flows through vLLM or TensorRT-LLM. Know both paths.

### 8. Embedding Versioning

A subtle prod issue: when you update your sentence-embedding model, your vector DB's stored embeddings become incompatible. New embeddings live in a different geometry. Plans:

- **Rolling reindex.** Re-embed the entire corpus offline, swap the index atomically.
- **Dual-write.** During the migration, write both v1 and v2 embeddings; query both, prefer v2 once index is full.
- **Version tag on every embedding.** Required for any embedding-backed system that lives more than a few months.

### 9. Generation Hygiene (NLP / LLM)

When you ship an LLM endpoint:

- Pin generation parameters (`temperature`, `top_p`, `top_k`, `max_new_tokens`, `repetition_penalty`) as part of the model artifact. Two clients with different generation configs are effectively two different products.
- Stop sequences and EOS tokens — verify your tokenizer's EOS token matches the model's training convention. Mismatched EOS leads to models that generate forever.
- Structured output: prefer constrained decoding (Outlines, lm-format-enforcer, vLLM's guided decoding) over post-hoc regex on free-form generation.

### 10. Data Versioning for DL

DVC tracking a 200 GB image dataset works fine, but:

- For datasets that grow continuously (every week a new batch of labeled images), the **WebDataset** format (sharded tar files) is dramatically easier to stream from S3 than thousands of individual JPEGs.
- For large parquet / arrow datasets, **`mosaicml/streaming`** or **`webdataset`** beat raw `torch.utils.data.Dataset` for I/O throughput.
- For tokenized datasets, pre-tokenize once and store the tokens — never re-tokenize per epoch.

---

## DL-Specific F500 Interview Bank

Topic-organized questions you should be able to answer in 2–5 minutes each. The theory and interview bank chapter covers non-DL topics in depth.

### Training infrastructure

1. "Walk me through GPU memory during training of a Transformer. Where does it go and in what order does each component dominate as the model grows?"
2. "Your 13B fine-tune OOMs on 4×A100-80GB. Walk me through the lever sequence — what would you try first, second, third?"
3. "When would you reach for FSDP vs. DeepSpeed ZeRO-3? Same thing or meaningful difference?"
4. "Your training run achieves 30% GPU utilization. What are the top three suspects and how do you confirm?"
5. "Describe the difference between data parallel, tensor parallel, pipeline parallel, sequence parallel. When would you combine them and in what nesting?"
6. "FlashAttention — why is it faster, and what's the relationship between FA1, FA2, FA3?"
7. "Explain mixed-precision training. Why is BF16 preferred over FP16 on modern hardware?"
8. "Your loss is NaN-ing at step 1200. What's the debug protocol?"
9. "Compare full fine-tuning, LoRA, and QLoRA. When does each win? What's the memory math for QLoRA of a 7B model on a 24 GB GPU?"
10. "Why does activation recomputation save memory? What's its time cost roughly?"

### Inference / serving

1. "Walk me through what happens to memory and compute during LLM inference. Why is the KV cache often the bottleneck and not the parameters?"
2. "Explain PagedAttention. What problem was it designed to solve?"
3. "Continuous batching vs. static batching — why does continuous batching improve throughput so dramatically for LLM serving?"
4. "Speculative decoding — describe it. When does it win, when does it lose?"
5. "You need to serve a fine-tuned 7B model at 100 RPS with P95 < 200 ms TTFT. Walk me through the stack you'd choose."
6. "You have 50 small fine-tuned variants of one 7B base model, one per customer. How do you serve them efficiently?"
7. "Quantization: INT8 vs INT4 vs FP8. Where does each come from, when do you reach for each, what's the quality cost?"
8. "Describe the ONNX → TensorRT path for a CV model. What can go wrong?"
9. "Triton dynamic batching configuration — walk me through `preferred_batch_size` and `max_queue_delay_microseconds`. How do you tune them?"
10. "An image-classification model serves 200 RPS, P95 at 80 ms. Your traffic doubles to 400 RPS. What breaks first?"

### Computer Vision specifically

1. "How do you handle preprocessing parity between training and serving for an image model?"
2. "Your image model's accuracy drops 5% in production despite stable offline metrics. Walk me through the diagnostic tree."
3. "Describe an end-to-end pipeline for an object detection model going from labeled COCO-style data to a deployed TensorRT endpoint at 60 FPS."
4. "How do you handle class imbalance for an object detection model? Per-class, the underrepresented classes will be terrible — what do you do?"
5. "Test-time augmentation — when is it worth it? What's the production cost?"
6. "Your CV model needs to run on an iPhone. Walk me through your export and quantization choices."
7. "How does Grad-CAM / SHAP / Integrated Gradients work for a CNN? When would you use each for explainability?"
8. "Compare ResNet / EfficientNet / ConvNeXt / ViT for an image classification task. What drives the choice in production?"
9. "Your model is fine on natural images but fails on medical images even after fine-tuning. What's likely happening?"
10. "How do you monitor a deployed image model for drift? What signals would you watch?"

### NLP / LLM specifically

1. "Tokenizer hygiene — what are the most common ways a tokenizer mismatch ruins production?"
2. "Walk me through training a custom embedding model for a domain. What loss, what negatives, how do you build the training set?"
3. "Compare SFT, DPO, ORPO, KTO, GRPO. When do you reach for each?"
4. "Describe a RAG pipeline end-to-end. Where do most RAG systems fail in practice?"
5. "How do you evaluate an LLM application without ground truth? What's your eval harness shape?"
6. "Prompt injection — describe the attack and three defenses. Where do the defenses fail?"
7. "How would you fine-tune a 70B model on your data with a budget of $5K?"
8. "Multi-LoRA serving — describe how vLLM handles dozens of adapters sharing one base model. What's the throughput impact per adapter?"
9. "Your RAG quality is poor. Walk me through the diagnostic — chunking, embedding, retrieval, reranking, generation."
10. "Hallucination — define it precisely. How do you measure it, how do you reduce it, how do you detect it at inference time?"

### Production DL system design (10-minute answers)

1. "Design a system to serve a fine-tuned 7B LLM for a 200-team internal user base with per-team rate limits and cost attribution."
2. "Design a real-time image-moderation pipeline for a social network at 50K RPS."
3. "Design end-to-end training infrastructure for an org that fine-tunes 50 LLM variants per month."
4. "Design an embedding pipeline for a 500 M-document corpus with weekly refresh and embedding-model rolling upgrade."
5. "Design an on-device CV model deployment pipeline including OTA model updates and per-version analytics."
6. "Design a multi-tenant prompt + RAG platform with versioned prompts, eval gates on deployment, and rollback."

For each: spend the first 5–10 minutes on clarifying questions, then sketch the architecture, then drill into 2–3 components, then talk failure modes, cost, and operations. The system-design framework covered in the architect track maps directly.

---

## Where to Find the Datasets and Models for Your Practice

### CV

- Hugging Face Datasets — images
- `timm` model zoo for backbones
- COCO, ImageNet (the latter has license constraints — be careful)
- MedMNIST for medical imaging at low scale
- Open Images, Visual Genome for larger / more complex tasks
- Roboflow Universe for hundreds of pre-labeled domain datasets

### NLP

- Hugging Face Datasets — text (the canonical source)
- The Pile (large LLM training corpus; expensive)
- C4 (cleaner; smaller)
- Domain corpora: PubMed for medical NLP, Stack Exchange dumps for QA
- LM Eval Harness — academic benchmarks
- MTEB — text embedding benchmark

### LLMs (base models)

- Llama-3.1-{8B, 70B}, Llama-3.2-{1B, 3B} — Meta
- Qwen-2.5 family — Alibaba
- Mistral / Mixtral / Codestral — Mistral AI
- Phi-3.5, Phi-4 — Microsoft
- Gemma 2 — Google
- DeepSeek-V3, R1 — DeepSeek
- For 2026 freshness, check Hugging Face Hub trending weekly

---

## A Compact DL-MLOps Project Path

If you want a single recommended portfolio sequence:

### Month 1–2: tier-1 DL project

A vision or text classifier deployed end-to-end. Same MLOps scaffolding (uv, MLflow, DVC, FastAPI, Docker, GitHub Actions) but DL model and DL preprocessing. Adds a `pytest` parity test for the preprocessing pipeline.

### Month 3–5: tier-2 production DL

The same model with an orchestrated weekly retraining pipeline, drift monitoring (preprocessing-statistic and embedding-distance drift), alias-based promotion, CI/CD/CT, dashboards. Deployed on a K8s cluster (kind locally, or a small managed cluster).

### Month 6–9: tier-3 capstone

Pick one:

- **LLM platform with RAG + multi-LoRA serving** (Projects 2 + 5 elements from the capstone track)
- **High-throughput CV serving with TensorRT + KServe** (Project 6 from the capstone track, with CV)
- **Distributed fine-tuning factory** (Project 5 from the capstone track, in full)

### Month 10–12: polish, blog, apply

- Two-three blog posts on the most interesting decisions
- A 5-minute screencap demo per project
- A repo README that reads like a tech blog post
- Interview prep using the question bank above and the theory and interview bank chapter

This puts you, twelve months from a serious start, in a credible position for senior DL-MLOps / ML platform engineer roles at any F50 that does serious deep learning.
