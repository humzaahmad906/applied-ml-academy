# 07 — Serving an ML Model Behind an Endpoint

This is the lesson everything so far has been building toward. You have an HTTP server (Lesson 04), typed request and response models (Lesson 05), and the tools to handle errors and load resources once (Lesson 06). Now you put a real trained model behind an endpoint so that any system — a web app, a mobile client, another service — can send features and get a prediction back. This is the concrete bridge from "I trained a model in a notebook" to "my model is a service other people can call," which is the heart of MLOps.

## The core pattern: load once, predict many

The single most important rule of model serving is: **load the model once at startup, never per request.** A route runs on every request; loading a model there would reload it thousands of times and make every call slow. Instead, load it in the `lifespan` handler (Lesson 06) and stash it on `app.state`, which is FastAPI's place for shared, app-wide objects. The predict route then reads it from there.

First, train and save a tiny model so we have something concrete to serve. We use scikit-learn and `joblib`, the standard way to persist a fitted estimator:

```python
# train.py — run once to produce model.joblib
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=1000).fit(X, y)
joblib.dump({"model": clf, "version": "1.0.0"}, "model.joblib")
```

We save a dict bundling the estimator with a version string, so the artifact carries its own metadata. Now the app loads that artifact at startup and puts it on `app.state`:

```python
from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load("model.joblib")
    app.state.model = bundle["model"]
    app.state.model_version = bundle["version"]
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)
```

The model is now in memory for the entire life of the process. Every request reuses the same object.

## Validating the input before inference

A model has strict expectations: the right number of features, of the right types, in a sensible range. If you feed it garbage, you get either a crash or — worse — a confident, meaningless prediction. Validate the input with a Pydantic model (Lesson 05) *before* it reaches the model, so bad input is rejected with a clean `422`:

```python
from pydantic import BaseModel, Field

class IrisRequest(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)
```

The iris model expects exactly four features, so `min_length=4, max_length=4` enforces the feature count for free. For richer inputs you would name each feature and constrain its range with `Field(ge=..., le=...)`. The point is that the model never sees a malformed row.

## The predict route and output shaping

The route reads the model off `app.state`, runs inference, and returns a **shaped** response — not a bare number, but a small object that says what was predicted, how confident the model is, and which model version produced it:

```python
from fastapi import FastAPI, HTTPException, Request

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
def predict(req: IrisRequest, request: Request) -> PredictionResponse:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    proba = model.predict_proba([req.features])[0]
    return PredictionResponse(
        prediction=int(proba.argmax()),
        confidence=float(proba.max()),
        model_version=request.app.state.model_version,
    )
```

Including `model_version` in every response is a small habit with a big payoff: when you later diagnose a bad prediction in production, you know exactly which model made it. Returning `confidence` lets callers set their own thresholds — for example, routing low-confidence cases to a human.

## The health check endpoint

Deployment systems — Kubernetes, load balancers, cloud autoscalers — need a cheap way to ask "is this instance ready to serve traffic?" That is what a **health check** is. It returns quickly and reports whether the model actually loaded:

```python
@app.get("/health")
def health(request: Request) -> dict:
    ready = request.app.state.model is not None
    return {"status": "ok" if ready else "loading", "model_loaded": ready}
```

Deployment platforms poll this endpoint and only send real requests to instances that report healthy; if `/health` fails, the platform restarts the instance or holds traffic back. Without it, a broken instance whose model failed to load would silently return errors to users. You will configure exactly these health checks when you deploy in the MLOps Engineer Nanodegree.

## Handling model-specific errors

Beyond validation, inference has its own failure modes: an input with the wrong dimensions slipping past a loose schema, a `NaN` in the features, or the model not being loaded at all (the `503` above). Catch these and return meaningful errors rather than a raw `500`:

```python
import math

@app.post("/predict-safe", response_model=PredictionResponse)
def predict_safe(req: IrisRequest, request: Request) -> PredictionResponse:
    if any(math.isnan(x) for x in req.features):
        raise HTTPException(status_code=422, detail="features contain NaN")
    model = request.app.state.model
    proba = model.predict_proba([req.features])[0]
    return PredictionResponse(
        prediction=int(proba.argmax()),
        confidence=float(proba.max()),
        model_version=request.app.state.model_version,
    )
```

Each check turns a would-be crash into a clear message the caller can understand and act on.

## A performance note

Model inference is usually **CPU-bound** — it burns processor time rather than waiting on the network. That matters because FastAPI runs `async def` routes on a single event loop, and heavy CPU work inside an `async def` route *blocks* that loop, freezing every other request. The safe default is to write your predict route as a **plain `def`** (as above): FastAPI automatically runs `def` routes in a thread pool, keeping the event loop free. If a route must be `async` for other reasons, offload the inference with `run_in_executor`. Lesson 09 is the full async deep-dive; for now, remember the rule: **CPU-bound inference belongs in a plain `def` route.**

## From endpoint to deployed service

You now have a complete, self-contained prediction API. The natural next step is to package it so it runs the same way everywhere. You will wrap this API in a container in the **Docker & Containers course** — its Lesson 06 shows the container side: writing the Dockerfile, copying in `model.joblib`, and exposing the port. Together, this lesson and that one are the full path from a trained model to a running service.

## Key takeaways

- Load the model once in `lifespan` and store it on `app.state`; never load a model inside a route.
- Validate inputs with a Pydantic model (feature count, types, ranges) so the model only ever sees well-formed data.
- Shape the output as `{"prediction": ..., "confidence": ..., "model_version": ...}` — versioning every response pays off in debugging.
- Expose a `GET /health` endpoint; deployment systems poll it and only route traffic to instances that report ready.
- Handle model-specific failures — not loaded (`503`), NaN or wrong dimensions (`422`) — with clear errors instead of a raw `500`.
- CPU-bound inference blocks the event loop, so use a plain `def` route (auto-threaded) or `run_in_executor`; the async details are in Lesson 09.
- This API becomes a deployable service once containerized — the Docker & Containers course, Lesson 06, shows the container side.

## Try it

Run the `train.py` above to produce `model.joblib`, then build the serving app: load the bundle in `lifespan`, store the model and version on `app.state`, and add the `/predict`, `/health`, and validation pieces. Start it with `uvicorn main:app --reload`, open `/docs`, and POST a valid iris row like `{"features": [5.1, 3.5, 1.4, 0.2]}` to see a shaped prediction with a confidence and version. Then hit `/health` and confirm it reports the model loaded. Finally, send a body with only three features and with a `NaN`, and verify you get clean `422`s rather than a crash.
