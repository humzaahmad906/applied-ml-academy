# 10 — Testing, Documentation, and API Versioning

An API you cannot test with confidence is an API you are afraid to change. This final lesson closes the loop: you will write automated tests that call your routes and assert on the responses, learn to enrich the auto-generated docs so your API explains itself, and see how to version an API so you can evolve it without breaking existing callers. These are the habits that turn a working prototype into a service a team can maintain for years.

## Testing with TestClient

FastAPI ships a `TestClient` (built on `httpx`) that lets you call your app *in the same process* — no running server, no network. You import your `app`, wrap it, and make requests just like a real client, then assert on the status code and JSON body:

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_happy_path():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {0, 1, 2}
    assert 0.0 <= body["confidence"] <= 1.0
```

Each test sends a request and checks two things: the **status code** (did the API respond the way it should?) and the **JSON body** (is the payload correct?). Because it runs in-process, the whole suite executes in milliseconds, so you can run it on every change. These tests use `pytest` (functions named `test_*`), the standard Python test runner.

## Fixtures: a client and a mock dependency

`pytest` fixtures provide reusable setup. A simple one hands every test a ready `client` so you do not repeat the wiring:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)
```

The more powerful trick is **overriding a dependency** (Lesson 06). In tests you rarely want to load the real model — it is slow and its predictions drift. FastAPI lets you swap any dependency for a fake via `app.dependency_overrides`, so the route runs against a predictable stub:

```python
from main import app, get_model

def fake_model():
    class Stub:
        def predict_proba(self, X):
            return [[0.1, 0.7, 0.2]]   # always class 1
    return Stub()

@pytest.fixture
def client_with_stub():
    app.dependency_overrides[get_model] = fake_model
    yield TestClient(app)
    app.dependency_overrides.clear()   # undo after the test
```

Now the predict route uses `fake_model` instead of the real one, so the test is fast and deterministic. Clearing the overrides afterward keeps tests isolated. (This is exactly why Lesson 06 injected the model via `Depends` — so it could be swapped here.)

## What to test — the useful 80%

You cannot test everything, and chasing 100% coverage produces brittle tests that assert on trivia. Aim instead for the handful of cases that catch real bugs:

- **Happy path** — a valid request returns `200` and the expected body.
- **Validation errors** — malformed input returns `422` (too few features, wrong types).
- **Auth failures** — missing or wrong credentials return `401`/`403` (Lesson 08).
- **Not found** — an unknown resource returns `404` (Lesson 06).

```python
def test_validation_error(client):
    response = client.post("/predict", json={"features": [1.0, 2.0]})  # need 4
    assert response.status_code == 422

def test_missing_api_key(client):
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 401
```

That set — happy path plus the main failure modes — is the useful 80% that gives real confidence, not coverage theater. Measure coverage with `pytest-cov` if you like, but treat the number as a hint, not a target.

## Enriching the auto-docs

FastAPI already generates OpenAPI (the machine-readable API spec) and renders it as Swagger UI at `/docs` (Lesson 04). You make those docs genuinely useful by adding a little metadata to each route:

```python
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run a prediction",
    description="Accepts four iris measurements and returns the predicted class.",
    tags=["inference"],
)
def predict(...): ...
```

`summary` and `description` become human-readable text in the docs; `response_model` (Lesson 05) documents the exact response shape *and* filters the output; `tags` group related routes into sections. Well-tagged, well-described docs mean a new teammate — or you in six months — can understand and call the API without reading its source.

## Versioning an API

Once other systems depend on your API, you cannot freely change its shape — a renamed field or a new required parameter breaks every caller. **Versioning** lets you ship changes while keeping old callers working. The simplest and most common approach is a **URL prefix**:

```python
from fastapi import APIRouter, FastAPI

app = FastAPI()
v1 = APIRouter(prefix="/v1")

@v1.post("/predict")
def predict_v1(...): ...

app.include_router(v1)
```

Callers hit `/v1/predict`. When you need a breaking change, you add a `/v2` router and run both for a while, giving clients time to migrate before you retire `/v1`. Header-based versioning (a custom `API-Version` header) also exists but is rare in ML serving; the URL prefix is clear, visible in logs, and easy to test, which is why it dominates.

## A note on gRPC and GraphQL

REST-over-JSON, which these lessons teach, is the default for serving ML models, but you will encounter two alternatives. **gRPC** uses binary Protocol Buffers over HTTP/2; it is faster and strongly typed, and you see it in high-throughput internal microservices and some model servers (TensorFlow Serving, Triton). **GraphQL** lets clients request exactly the fields they want in one query; it shines for complex front-ends with many related resources. Both are worth recognizing, but REST+JSON remains the ML-serving default because it is universally supported, trivially debuggable with `curl`, and self-documenting through OpenAPI — the reasons this whole course is built on it.

## Testing async routes

Most of your tests can use the synchronous `TestClient` even for `async` routes — it runs the event loop for you. When you specifically need to test *async* behavior (concurrent calls, streaming), use `httpx.AsyncClient` with an `ASGITransport` pointed at your app, inside an async test (with `pytest-asyncio`):

```python
import httpx
from httpx import ASGITransport
from main import app

async def test_predict_async():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
```

For the vast majority of route tests, though, the plain `TestClient` is simpler and enough.

## Key takeaways

- `TestClient` (from `fastapi.testclient`, built on `httpx`) calls your app in-process; tests assert on status code and JSON body and run in milliseconds.
- Use `pytest` fixtures for a shared `client`, and `app.dependency_overrides` to swap a real model for a fast, deterministic stub.
- Test the useful 80% — happy path, `422` validation, `401`/`403` auth, `404` not found — not coverage theater.
- FastAPI auto-generates OpenAPI/Swagger docs; enrich them with `summary`, `description`, `response_model`, and `tags`.
- Version with a URL prefix (`/v1/...`) via an `APIRouter`; run old and new versions in parallel during migrations. Header-based versioning is rare in ML.
- gRPC (fast, binary, HTTP/2) and GraphQL (client-chosen fields) exist, but REST+JSON is the ML-serving default for its ubiquity and debuggability.
- For genuinely async tests, use `httpx.AsyncClient` with `ASGITransport`; otherwise `TestClient` is enough.

## Try it

Write a full test suite for the predict API from Lesson 07. Create a `conftest.py` with a `client` fixture and a second fixture that overrides the model dependency with a stub returning a fixed class. Then write tests covering: the happy path (`200` with a valid four-feature body and a confidence in `[0, 1]`), a validation error (`422` for the wrong number of features), an auth failure if you added the Lesson 08 API key (`401` with no key), and the health check (`GET /health` returns the model-loaded status). Run it with `pytest -v`, then add `summary`, `description`, and `tags` to your routes and confirm at `/docs` that the descriptions and grouping appear.
