# 01 — Beginner Guide: Foundations and Your First Production Model

**Topics:** ML refresher (scikit-learn / PyTorch quickstart), Docker for ML, experiment tracking (MLflow), data versioning (DVC), packaging models, simple deployment with FastAPI.

**Time:** 4–6 weeks at 8–10 hrs/week.
**Goal:** Build genuine confidence by training, tracking, packaging, and serving a real model end-to-end — from raw data on disk to a live HTTP endpoint with reproducible metrics.

## What You Will Be Able to Do After This Tier

- Train classical ML and small DL models locally without thrashing
- Reproducibly track every experiment with its code commit, data version, hyperparameters, and metrics
- Version data and models, not just code
- Containerize a model and serve it behind a FastAPI endpoint
- Run that container locally and on a small cloud VM
- Test the service like real software (unit, integration, contract)

By the end, you'll have a portfolio-worthy project even though you're only a quarter through the curriculum: a "model in a box" with reproducible training, tracked experiments, versioned artifacts, and a live API.

---

## Week 1 — Reproducible Local ML

### Why Reproducibility Comes First

The single most consistent pathology in industry ML is "the model worked on someone's laptop." A data scientist trained something, posted a notebook, and now no one can rebuild it. The numbers in the paper don't match the numbers in production. Nobody knows what version of the data was used.

MLOps starts here: before you optimize, before you scale, before you deploy — make the experiment reproducible. Same code + same data + same hyperparameters + same seed = same numbers, every time, anywhere.

### What to Learn

#### 1. Environment isolation with `uv`

`uv` is the 2026 standard for Python environments. Faster than `pip`, lock-file native, drops into `pyproject.toml`. Install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init my-mlops-project
cd my-mlops-project
uv add scikit-learn pandas mlflow fastapi pydantic typer
uv lock                       # creates uv.lock
uv sync                       # creates .venv/ and installs
uv run python my_script.py    # runs in the env
```

The lock file is the contract. Anyone with `uv.lock` and `pyproject.toml` gets the same dependency tree. Commit both.

If your shop uses `poetry` or `pipenv`, the principles are identical. The point is *lock files exist in your repo*.

#### 2. Random seeds — everywhere

Reproducibility requires deterministic randomness:

```python
import random, numpy as np, torch, os

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # The next two trade speed for determinism — flip when needed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Every entrypoint calls `set_seed()` first. Every train/test split passes `random_state=SEED`.

You will *not* always get fully deterministic GPU training even with this — some CUDA ops are nondeterministic by design — but you'll get close. Document the gap in your README.

#### 3. Project structure that scales

Don't ship a single 800-line notebook. Use this skeleton from day one:

```
my-mlops-project/
├── pyproject.toml
├── uv.lock
├── README.md
├── .env.example                  # never commit .env itself
├── .gitignore
├── Makefile                      # `make train`, `make serve`, `make test`
├── data/
│   ├── raw/                      # never modified; tracked by DVC, not Git
│   ├── interim/
│   └── processed/
├── notebooks/                    # exploration only; never the source of truth
├── src/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings; one source of truth
│   ├── data/
│   │   ├── load.py
│   │   └── features.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   ├── serving/
│   │   └── app.py                # FastAPI app
│   └── utils/
│       ├── logging.py
│       └── seed.py
├── tests/
│   ├── test_features.py
│   ├── test_train.py
│   └── test_serving.py
├── Dockerfile
└── .github/
    └── workflows/
        └── ci.yml
```

This structure follows the widely adopted convention for ML project layout, with a serving folder added. Standard enough that anyone reading your repo recognizes it instantly.

#### 4. Configuration with `pydantic-settings`

Never hardcode paths, hyperparameters, or credentials. Use `pydantic-settings`:

```python
# src/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    raw_data_path: Path = PROJECT_ROOT / "data/raw/train.csv"
    processed_data_path: Path = PROJECT_ROOT / "data/processed/train.parquet"
    model_artifact_path: Path = PROJECT_ROOT / "artifacts/model.joblib"

    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6
    seed: int = 42

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "baseline"

    # Registry — alias-based (MLflow 2.9+); see Week 2-3 of the medium guide
    model_name: str = "income_classifier"
    model_alias: str = "champion"
    model_version: str = "0.1.0"

settings = Settings()
```

`pydantic-settings` v2 uses `model_config = SettingsConfigDict(...)` — the old `class Config:` form still works but emits deprecation warnings. The module-level `PROJECT_ROOT` constant avoids confusion about referencing one field's default from another.

Now every value can be overridden by environment variable (`APP_N_ESTIMATORS=500`) — perfect for CI/CD and Kubernetes.

### Exercises

1. Set up the project skeleton above for the [Kaggle Bike Sharing](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) or [UCI Adult Income](https://archive.ics.uci.edu/dataset/2/adult) dataset. Just the skeleton — no training yet.
2. Write a `src/data/load.py` that loads the raw CSV and writes a cleaned Parquet to `data/processed/`. Make it idempotent.
3. Add `make data` to the Makefile that runs the loader. Confirm it works in a fresh `uv sync`.
4. Commit. Tag `v0.1.0-data`. You're done with week 1.

---

## Week 2 — Your First Tracked Experiments (MLflow)

### Why You Track Experiments

In a real ML team, you'll run hundreds of experiments per week across the team. Without tracking:

- You can't compare runs systematically
- You can't reproduce the winning run six months later
- You can't audit "which data did this model see during training?"
- You can't tell whether yesterday's improvement was the model change or a data shift

Experiment tracking is the audit log for your ML work. It's what makes MLOps possible.

### What MLflow Is

MLflow is the de facto OSS experiment tracker — and since **MLflow 3.0 (Jun 2025)** it's grown into a unified AI-engineering platform. The classic four components:

1. **Tracking** — log parameters, metrics, artifacts, code version, environment per run
2. **Models** — a standard format for packaging models with their environment
3. **Registry** — versioned, stage-promoted (Staging/Production) model store
4. **Projects** — a way to declare runnable projects with their dependencies (rarely used directly; use Docker instead)

Plus the 3.x additions you should know exist now (you'll use them in the LLMOps phase): **Tracing** (OpenTelemetry-compatible spans for LLM/agent calls via `@mlflow.trace`), **GenAI evaluation** (50+ built-in metrics), **prompt versioning**, and a built-in **AI Gateway** (one OpenAI-compatible endpoint fronting any provider). Learn the v3 mental model from day one — "MLflow = tracker" is the 2023 framing.

You'll use Tracking + Models + Registry heavily in this guide and ignore Projects.

### Setting Up

Local tracking server with a SQLite backend and local artifact store:

```bash
uv add mlflow
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

UI at `http://localhost:5000`. You'll keep this running in a terminal during dev.

In production, MLflow runs against a Postgres backend and an S3/GCS artifact store. Same code, different config — set `MLFLOW_TRACKING_URI`.

### Your First Tracked Training Run

```python
# src/models/train.py
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import settings
from src.utils.seed import set_seed


def main():
    set_seed(settings.seed)

    df = pd.read_parquet(settings.processed_data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=settings.seed, stratify=y
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": settings.n_estimators,
            "learning_rate": settings.learning_rate,
            "max_depth": settings.max_depth,
            "seed": settings.seed,
            "n_rows": len(df),
            "n_features": X.shape[1],
        })

        model = GradientBoostingClassifier(
            n_estimators=settings.n_estimators,
            learning_rate=settings.learning_rate,
            max_depth=settings.max_depth,
            random_state=settings.seed,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_proba),
        }
        mlflow.log_metrics(metrics)

        # Log the model with its signature and an input example.
        # Signatures protect you at deployment time — wrong-shaped inputs fail fast.
        signature = mlflow.models.infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(
            model,
            name="model",  # MLflow 2.9+ — `artifact_path` is deprecated
            signature=signature,
            input_example=X_val.head(3),
        )

        print(f"Run {run.info.run_id}: {metrics}")


if __name__ == "__main__":
    main()
```

Run it. Open the MLflow UI. Click into the run. You should see params, metrics, the model artifact, and the input/output signature.

### Things to Internalize From This Snippet

1. **`mlflow.log_params(...)`** captures *every* setting that could influence the result. Include data version, feature pipeline version, anything that varies.
2. **`mlflow.log_metrics(...)`** can be called multiple times — useful for logging metrics per epoch in DL training, with a `step=` argument.
3. **The signature** is the model's input/output schema. When you deploy the model later, MLflow uses it to validate request shapes — saves you from production payload mismatches.
4. **The input example** travels with the model artifact. New consumers can see "what does this model expect" without reading code.

### Hyperparameter Sweeps with Optuna

`Optuna` is the dominant HPO library. Integrates cleanly with MLflow:

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

mlflow_cb = MLflowCallback(
    tracking_uri=settings.mlflow_tracking_uri,
    metric_name="roc_auc",
)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
    }
    # ... train + evaluate ...
    return roc_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, callbacks=[mlflow_cb])
```

Each trial becomes an MLflow run. Sort the experiment by `roc_auc` descending. You have a winner.

### Compare-Runs Discipline

In the MLflow UI: select 5–10 runs, click "Compare." Look at:

- Parallel coordinates plot of params vs metric (best-trial trajectory)
- Scatter plot of any two params
- Aggregate metric table

A senior MLOps engineer can drive the MLflow comparison view fluently. You should be able to.

### Exercises

1. Add MLflow logging to your bike sharing / income classifier. Log at least 8 parameters, 5 metrics, and the model artifact.
2. Run 30 hyperparameter trials with Optuna. Pick the best.
3. Add a `make train` Makefile target that does a single run with current settings; `make sweep` that does 30 trials.
4. Re-train two weeks later with new data. Diff the metrics in the UI. Did anything regress?

---

## Week 3 — Versioning Data and Models with DVC

### Why Data Versioning Matters

Git tracks code. Git does not track data well — repos balloon, history is unreadable. But you need data versions because:

- Reproducing a model six months later requires its training data
- "Why did this model get worse?" often reduces to "the data changed"
- Compliance audits ask "which version of the training data was used?"

DVC (Data Version Control) is the dominant OSS tool. It puts pointers in Git and real data in S3/GCS/Azure/local-disk.

### Setting Up DVC

```bash
uv add dvc dvc-s3
dvc init
git add .dvc/.gitignore .dvc/config
git commit -m "Initialize DVC"

# Configure remote (S3 in this case)
dvc remote add -d storage s3://my-mlops-data-bucket/dvcstore
git add .dvc/config
git commit -m "Add DVC remote"
```

For local development without a cloud account, use a local remote — point it at a directory on your laptop.

### Tracking Data

```bash
dvc add data/raw/train.csv
# This creates data/raw/train.csv.dvc (a small pointer file) and adds train.csv to .gitignore.
git add data/raw/train.csv.dvc data/raw/.gitignore
git commit -m "Track raw training data with DVC"

dvc push  # uploads the data to S3
```

Now `data/raw/train.csv.dvc` is in Git; the actual CSV is in S3. Anyone who clones the repo runs `dvc pull` to retrieve it.

### DVC Pipelines (`dvc.yaml`)

DVC's killer feature is reproducible pipelines that re-run only when their inputs change:

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: uv run python -m src.data.load
    deps:
      - src/data/load.py
      - data/raw/train.csv
    outs:
      - data/processed/train.parquet

  train:
    cmd: uv run python -m src.models.train
    deps:
      - src/models/train.py
      - data/processed/train.parquet
    params:
      - src/config.py:Settings.n_estimators
      - src/config.py:Settings.learning_rate
      - src/config.py:Settings.max_depth
    outs:
      - artifacts/model.joblib
    metrics:
      - artifacts/metrics.json:
          cache: false
```

Run with `dvc repro`. DVC computes which stages need to run based on changed inputs. Skips the rest. This is your local equivalent of an orchestrator — and a fine choice for small projects.

### Comparing Experiments with `dvc exp`

```bash
dvc exp run -S 'Settings.n_estimators=500' -S 'Settings.learning_rate=0.01'
dvc exp run -S 'Settings.n_estimators=200' -S 'Settings.learning_rate=0.1'
dvc exp show
```

DVC ties hyperparameters, code, data, and metrics together at the commit level. A nice complement to MLflow — DVC is great for "the pipeline as a whole," MLflow is great for "this model's runs."

### Where DVC Fits in 2026

You will hear arguments about DVC vs LakeFS vs Delta Lake vs "just use S3 prefixes." Honest summary:

- **DVC** — best for small/medium teams, project-scoped data; integrates cleanly with Git workflows
- **LakeFS / Nessie / Pachyderm** — better for shared lakes with many teams; git-like branches on object storage
- **Iceberg/Delta time travel** — when your data is already in a lakehouse, time travel handles versioning naturally
- **Plain S3 prefixes** with date-stamped paths (`s3://bucket/data/2026-05-12/`) — works for many real teams; less elegant but trivial

Learn DVC first. It's the right primitive to internalize. Then know the alternatives exist.

### Exercises

1. Put your `data/raw/` directory under DVC. Push to a remote (S3 if you have it, local otherwise).
2. Write a `dvc.yaml` with `prepare` and `train` stages.
3. Run `dvc exp run` with three different hyperparameter sets. Compare with `dvc exp show`.
4. Modify the processed-data stage. Confirm DVC re-runs only that stage and its downstream.

---

## Week 3–4 — Docker for ML

### Why Docker Matters Even More for ML

Every problem you've had reproducing a notebook — Python version, CUDA version, system library version, OS-specific behavior — Docker solves. In ML, the matrix is worse because:

- **CUDA + cuDNN + driver versions** must match the framework version (PyTorch built for CUDA 12.1 won't run on a host with CUDA 11.8 drivers in the obvious way)
- **System libraries** like libgomp, MKL, OpenBLAS affect performance and sometimes correctness
- **Hardware-specific builds** — Apple Silicon vs x86-64 vs ARM64 servers

A `Dockerfile` pins all of this. Same container, same behavior. Production parity with your laptop.

### The Mental Model

- **Image:** a snapshot of an environment (read-only)
- **Container:** a running instance of an image
- **Layer:** each `RUN` / `COPY` / `ADD` produces a layer; layers are cached
- **Volume:** persistent storage attached to a container
- **Network:** containers reach each other by service name within a Docker network

### A Production-Quality Dockerfile for Python ML

```dockerfile
# Use a slim base + multi-stage to keep image small
FROM python:3.12-slim AS builder

# System deps required for some Python wheels (lightgbm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally so we can use it during build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first — these change rarely, maximizes cache hits
COPY pyproject.toml uv.lock ./

# Install deps into a virtualenv we'll copy across stages
RUN uv sync --frozen --no-install-project

# Now copy the source — this layer invalidates often, so it goes last
COPY src/ ./src/

# Install the project itself
RUN uv sync --frozen


# Stage 2: small runtime image with only what's needed
FROM python:3.12-slim AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run as non-root in production
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Things to internalize:

1. **Multi-stage builds** — final image contains only runtime artifacts, not build toolchain. Smaller images = faster pulls = lower cost.
2. **Layer order matters** — copy slowly-changing files (`pyproject.toml`, `uv.lock`) before fast-changing files (source). Maximizes cache reuse.
3. **`--frozen`** — never recompute the lock file inside a Docker build; you want the same versions you've been testing with locally.
4. **Non-root user** — Kubernetes Pod Security Standards require this. Bake it in.
5. **`PYTHONUNBUFFERED=1`** — without this, container logs appear in lumpy bursts.

### GPU Dockerfile

For CUDA workloads, start from NVIDIA's CUDA images:

```dockerfile
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ... install Python deps ...
```

Run with `docker run --gpus all` (with the NVIDIA Container Toolkit installed on the host). On Kubernetes, request `nvidia.com/gpu: 1` in your pod spec.

### docker-compose for Local Stacks

When your local dev stack is MLflow + Postgres + MinIO + your app, write a `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports: ["5432:5432"]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    ports: ["9000:9000", "9001:9001"]

  mlflow:
    build: ./mlflow
    depends_on: [postgres, minio]
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://mlflow:mlflow@postgres/mlflow
      --default-artifact-root s3://mlflow/
    ports: ["5000:5000"]

  api:
    build: .
    depends_on: [mlflow]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports: ["8000:8000"]

volumes:
  postgres_data:
  minio_data:
```

`docker compose up -d` — your entire MLOps dev stack comes up in one command. This is the standard you should target for every project.

### Exercises

1. Write a Dockerfile for your training script. Build it. Run training inside the container.
2. Write a `docker-compose.yml` like above. Bring up Postgres + MinIO + MLflow + your training container. Confirm tracking works end-to-end.
3. Use `docker stats` while training runs. Note CPU/memory usage. Tune container resource limits.
4. (If you have a GPU) Build a CUDA version. Confirm `torch.cuda.is_available()` is true inside the container.

---

## Week 4 — Serving Models with FastAPI

### The Goal

You have a trained, tracked, versioned model. Now wrap it in an HTTP service that other systems can call.

The serving layer is where MLOps starts to look like production software engineering. You'll think about: request validation, latency, throughput, observability, graceful shutdown, health checks. Welcome to the real job.

### Why FastAPI

- Built on Starlette + Pydantic — async-native, strict request validation, OpenAPI docs for free
- The dominant choice for ML serving in 2026, beating Flask in every dimension that matters
- Easy path from "Python function" to "production HTTP service"

### A Minimal Serving App

```python
# src/serving/app.py
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

# ---------- Request / response models ----------

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=10, max_length=10)

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

# ---------- App lifecycle ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s", settings.model_artifact_path)
    app.state.model = joblib.load(settings.model_artifact_path)
    app.state.model_version = settings.model_version
    yield
    # Cleanup on shutdown
    app.state.model = None

app = FastAPI(title="Income Classifier", version="0.1.0", lifespan=lifespan)

# ---------- Routes ----------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/ready")
async def ready() -> dict[str, str]:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    x = np.array(req.features).reshape(1, -1)
    proba = app.state.model.predict_proba(x)[0, 1]
    pred = int(proba >= 0.5)
    return PredictResponse(
        prediction=pred,
        probability=float(proba),
        model_version=app.state.model_version,
    )
```

Run with `uvicorn src.serving.app:app --reload`. Hit `http://localhost:8000/docs` — interactive OpenAPI docs, free.

### Things to Internalize

1. **Load the model once at startup** (`lifespan` context). Loading per request is a beginner's mistake — adds hundreds of ms.
2. **Separate `/health` and `/ready`.** Health = process is alive. Ready = process is alive *and* can serve traffic (model loaded, dependencies reachable). Kubernetes uses both differently — `/health` for liveness probes, `/ready` for readiness probes.
3. **Strict input validation via Pydantic.** Wrong-shaped input returns a 422 with a clear error message — saves debugging time downstream.
4. **`response_model`** turns FastAPI into a contract — clients know what they'll get.
5. **`app.state`** is per-process storage. Avoid module-level globals; they're confusing across reloads and tests.

### Loading Models from MLflow

In production you don't load `.joblib` from disk — you load a versioned model from the registry:

```python
import mlflow.sklearn

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
    app.state.model = mlflow.sklearn.load_model(model_uri)
    yield
    app.state.model = None
```

Now `MODEL_STAGE=Production` in your environment fetches the current-prod model; `MODEL_STAGE=Staging` fetches a candidate. You can canary-deploy by varying `MODEL_STAGE` per replica.

### Batch Endpoints

Single-row endpoints are easy and slow. Real systems batch:

```python
class BatchPredictRequest(BaseModel):
    items: list[list[float]]

class BatchPredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]

@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    X = np.array(req.items)
    proba = app.state.model.predict_proba(X)[:, 1]
    return BatchPredictResponse(
        predictions=[int(p >= 0.5) for p in proba],
        probabilities=proba.tolist(),
    )
```

For high-throughput services, you'll also implement **server-side micro-batching** (collect N single requests for X ms, then run them as a batch). We'll cover that in the advanced guide.

### Observability — The Three Pillars, From Day One

Even your beginner service should have:

1. **Logs** — structured (JSON), correlated by request ID
2. **Metrics** — Prometheus-format, exposed at `/metrics`
3. **Traces** — OpenTelemetry spans for end-to-end latency breakdown

Minimal Prometheus instrumentation:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

That single line gives you request counts, latencies (histograms), in-flight requests, all at `/metrics`. Grafana eats it natively.

### Testing the Service

Three layers:

```python
# tests/test_serving.py
from fastapi.testclient import TestClient
from src.serving.app import app

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

def test_predict_happy():
    with TestClient(app) as client:
        r = client.post("/predict", json={"features": [0.0] * 10})
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["probability"] <= 1.0

def test_predict_bad_input():
    with TestClient(app) as client:
        r = client.post("/predict", json={"features": [0.0] * 5})
        assert r.status_code == 422  # validation error
```

Run with `pytest -v`. Add to CI in week 5.

### Exercises

1. Wrap your trained model in a FastAPI service. Add `/health`, `/ready`, `/predict`, `/predict_batch`, `/metrics`.
2. Switch to loading from MLflow registry, parameterized by stage.
3. Write 5+ tests covering happy path, validation errors, batch endpoint, health/ready, and a "model fails to load" scenario.
4. Containerize the service. Run locally. Hit it from `curl`.
5. Add a `tests/test_load.py` that uses `locust` or `wrk` to send 100 RPS for 30 seconds. Note P50, P95, P99 latency.

---

## Week 5 — Putting It Together: CI + Quickstart Deployment

### A Basic GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - name: Lint
        run: uv run ruff check .
      - name: Format check
        run: uv run ruff format --check .
      - name: Type check
        run: uv run mypy src
      - name: Tests
        run: uv run pytest -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: coverage.xml

  build-image:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}/api:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

Every PR runs lint + type check + tests. Every merge to main builds and pushes a versioned image to GitHub Container Registry. This is the bare minimum for a real project.

### Deploying to a Cheap Cloud VM

For a beginner project, full Kubernetes is overkill. Two options:

1. **A single VM (DigitalOcean / Hetzner / AWS Lightsail / GCP Compute Engine).** Run your container with `docker run -d --restart unless-stopped`. Put nginx in front for TLS. Total cost: $5–10/month.
2. **Managed container platform (Cloud Run, AWS App Runner, Fly.io, Railway).** Push the image; the platform handles scaling, TLS, health checks. Total cost: pennies for low-traffic projects.

Both are fine for portfolio work. **Cloud Run is the standout** for ML serving on GCP — scales to zero, request-billed, easy to canary by traffic percentage. App Runner is the equivalent on AWS.

### Smoke-Testing the Deployed Service

```bash
ENDPOINT=https://your-service-url
curl -fsS "$ENDPOINT/health" | jq .
curl -fsS -X POST "$ENDPOINT/predict" \
  -H 'Content-Type: application/json' \
  -d '{"features": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}' | jq .
```

Add this as a smoke test step in your CI/CD pipeline, run against staging after every deploy.

---

## The Beginner-Tier Project

This is what you build before moving to Tier 2.

### Spec

An end-to-end "model in a box":

1. **A real dataset** — Kaggle, UCI, or HuggingFace Datasets. Tabular is fine for the first project.
2. **Project structure** matching the skeleton above.
3. **`uv` lock file** committed; `pyproject.toml` with explicit dependencies.
4. **DVC** tracking the raw data; data pulled from a real remote (S3 / GCS / Azure Blob).
5. **MLflow tracking server** (local or remote) with at least 30 logged runs across hyperparameter sweeps.
6. **A registered model in the MLflow Model Registry**, promoted to "Staging."
7. **A FastAPI service** loading the staging model, with `/health`, `/ready`, `/predict`, `/predict_batch`, `/metrics`.
8. **At least 10 tests** — unit (feature transforms), integration (end-to-end training), API (FastAPI TestClient).
9. **A `Dockerfile` and `docker-compose.yml`** that brings up MLflow + Postgres + MinIO + your API in one command.
10. **GitHub Actions CI** that lints, type-checks, tests, builds the image.
11. **Deployment to a real URL** (Cloud Run / App Runner / a $5 VM). Smoke-test passes from your laptop.
12. **A `README.md`** that includes:
    - 1-paragraph problem framing
    - Quickstart (5 commands to a local working stack)
    - Architecture diagram (Mermaid is fine)
    - Sample request/response
    - Known limitations and what you'd build next

### Acceptance Criteria

- A reviewer can clone your repo, follow your README, and have the service running locally in under 15 minutes.
- The deployed service returns sensible predictions for valid input and 422 for invalid input.
- Re-running training from `dvc repro` reproduces the metrics in the README within 1% (random seed acceptable).
- Cost is tracked — your README says what running this costs per month.

### What This Project Proves

- You can structure a Python project for production from day one
- You can track experiments with the audit trail real teams require
- You can version data alongside code
- You can package and serve a model as software
- You can deploy and operate it at minimum-viable level

These are the floor competencies. You're now on the floor.

---

## Confidence Checks Before Tier 2

Don't move on until you can answer these without googling:

1. Why is `requirements.txt` not enough — what does a lock file give you?
2. What's the difference between MLflow's tracking, models, and registry components?
3. Why do we copy `pyproject.toml` before `src/` in a Dockerfile?
4. What's the difference between Kubernetes liveness and readiness probes, and why do they map to `/health` and `/ready`?
5. Given a versioned model in the MLflow registry, what happens when you promote it from Staging to Production?
6. Why is loading the model in `lifespan` better than loading it per request?
7. What's the difference between `dvc repro` and `dvc exp run`?
8. What's `app.state.model` for in FastAPI, and why is it preferred over a module global?

If any is shaky, go back. You're not behind; you're being thorough.

When all eight feel solid, move on to the Medium Guide.

---

## You can now

- Scaffold a production-shaped Python ML project from day one — `uv` lock file, deterministic seeds, `pydantic-settings` config, and a layout that a reviewer recognizes instantly.
- Track every experiment in MLflow with its params, metrics, model signature, and input example, and drive the compare-runs view to pick a winner.
- Version data and reproducible pipelines with DVC — `dvc add`, a `dvc.yaml` with `prepare`/`train` stages, and `dvc exp` for hyperparameter comparison.
- Containerize a model with a multi-stage, non-root Dockerfile and bring up a full local stack (MLflow + Postgres + MinIO + your API) with one `docker compose up`.
- Serve a model behind FastAPI with `/health`, `/ready`, `/predict`, `/predict_batch`, and `/metrics`, loading the model once at startup and validating inputs strictly.
- Ship a GitHub Actions CI pipeline that lints, type-checks, tests, and builds a versioned image, then deploy the service to a real URL and smoke-test it.
