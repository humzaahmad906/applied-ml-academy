# 04 — Your First FastAPI Application

So far you have been a *client*, calling other people's APIs. Now you become the *server*. When you deploy a model, you wrap it in an API so other systems can send inputs and get predictions back — and the tool of choice for that in modern Python is **FastAPI**. This lesson builds your first working API from an empty file, introduces the auto-generated documentation that makes FastAPI a joy to work with, and shows the small project layout you will reuse for every service from here on.

## Why FastAPI

Python has several web frameworks. **Flask** is the classic minimal one and is still common in older codebases and simple apps; it is worth knowing it exists. But for serving ML models, FastAPI has become the default for concrete reasons:

- It is **async-native**, so it handles many concurrent requests efficiently — important when a model call is slow and requests pile up.
- It does **automatic request validation** through Pydantic (Lesson 05): you declare the shape of your data with Python type hints, and FastAPI rejects malformed input with a clear `422` before your code ever runs.
- It generates **interactive documentation** for free, so your API is self-describing from the first line.

That combination — speed, validation, and docs with almost no extra work — is why FastAPI dominates ML serving today.

## Installing and the minimal app

Install FastAPI with its standard extras, which pull in the `uvicorn` server and other conveniences. Using `uv` (the fast packaging tool we use throughout these courses):

```bash
uv add "fastapi[standard]"
```

The smallest possible app is three lines. Create a file called `main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root() -> dict:
    return {"status": "ok"}
```

`FastAPI()` creates the application object, conventionally named `app`. The `@app.get("/")` decorator registers the function below it to handle GET requests at the path `/`. Whatever the function returns — here a dict — FastAPI serializes to JSON automatically. There is no `json.dumps`, no manual `Content-Type` header; returning a dict is enough.

## Running it

FastAPI apps are run by an ASGI server; `uvicorn` is the standard one. From the folder containing `main.py`:

```bash
uv run uvicorn main:app --reload
```

`main:app` means "the object called `app` in the module `main`." The `--reload` flag restarts the server whenever you edit a file, which is exactly what you want during development. Visit `http://127.0.0.1:8000/` in a browser or `curl` it, and you get `{"status":"ok"}` back.

## Path parameters

To capture part of the URL as an argument, put a name in braces in the path and declare a matching function parameter. The **type annotation** is doing real work here:

```python
@app.get("/models/{model_id}")
def get_model(model_id: int) -> dict:
    return {"model_id": model_id, "name": "resnet50"}
```

Because `model_id` is annotated as `int`, FastAPI converts the URL segment to an integer for you and returns a clean `422` if someone passes `/models/abc`. You get validation from a single type hint. This is the path-param-as-identity idea from Lesson 02, now in code.

## Query parameters

Any function parameter that is *not* in the path becomes a **query parameter**. Give it a default value to make it optional:

```python
@app.get("/models")
def list_models(framework: str | None = None, limit: int = 20) -> dict:
    return {"framework": framework, "limit": limit}
```

A request to `/models?framework=pytorch&limit=5` fills those in; `/models` alone uses the defaults. This mirrors the query-params-for-filtering-and-pagination pattern from Lesson 02. As with path params, the type annotations drive automatic conversion and validation.

## Route ordering

FastAPI matches routes in the order you declare them, so put **specific paths before variable ones**. If a fixed route can be swallowed by a parameterized route above it, it will never be reached:

```python
@app.get("/models/featured")   # specific — declare first
def featured_models() -> dict:
    return {"featured": ["resnet50"]}

@app.get("/models/{model_id}") # variable — declare after
def get_model(model_id: int) -> dict:
    return {"model_id": model_id}
```

If these were reversed, `/models/featured` would try to match `{model_id}` as an int, fail with 422, and never hit the featured route.

## Setting the status code

By default a successful route returns `200`. When a route *creates* something, REST convention says return `201 Created` (Lesson 01). Declare it on the decorator:

```python
@app.post("/predictions", status_code=201)
def create_prediction() -> dict:
    return {"id": 1, "result": "cat"}
```

Now a successful POST responds with `201` instead of `200`, matching the semantics of creation.

## The automatic docs

Here is FastAPI's standout feature. With the server running, open `http://127.0.0.1:8000/docs`. You get **Swagger UI**: an interactive page listing every route, its parameters, and its expected responses — generated entirely from your type hints and decorators, with nothing extra to write. You can expand any endpoint and click "Try it out" to send a live request from the browser. Your API documents and lets you test itself from the very first line of code. (There is a second style at `/redoc` if you prefer it.)

## A small project layout

A simple service does not need much structure. For a single-file API using `uv`, the whole project is:

```
my-api/
├── main.py           # the FastAPI app and routes
├── pyproject.toml    # dependencies, managed by uv
└── README.md
```

`uv init` creates the `pyproject.toml` for you and `uv add "fastapi[standard]"` records the dependency there, so anyone who clones the project can run `uv sync` to reproduce your environment. As the API grows you will split routes into multiple files, but starting with a single `main.py` is exactly right. In the Docker & Containers course you will take an app like this and package it into a container for deployment.

## Key takeaways

- FastAPI is the modern default for ML serving because it is async-native, validates requests automatically, and generates interactive docs for free; Flask still exists for simpler or legacy cases.
- Install `fastapi[standard]` and run with `uvicorn main:app --reload` during development.
- Create routes by decorating functions with `@app.get`, `@app.post`, etc.; a returned dict is serialized to JSON automatically.
- Path parameters in `{braces}` with type annotations give you automatic conversion and validation.
- Any non-path parameter is a query parameter; a default value makes it optional.
- Declare specific routes before parameterized ones, and set `status_code=201` on creation routes.
- `/docs` gives you a live Swagger UI generated from your code — the API documents itself.

## Try it

Build a small "model registry" API in a single `main.py`. Add a root route returning a status message, a `GET /models` route that accepts optional `framework` and `limit` query parameters, a `GET /models/{model_id}` route that echoes the id, and a `POST /predictions` route that returns `201`. Run it with `uvicorn ... --reload`, then open `/docs` and exercise every endpoint from the Swagger UI. Deliberately request `/models/abc` and observe the automatic `422` — then read the error body to see exactly what FastAPI told the caller.
