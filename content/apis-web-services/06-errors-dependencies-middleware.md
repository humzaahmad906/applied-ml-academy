# 06 — Errors, Dependencies, and Middleware

Your API so far returns happy results, but real services spend a surprising amount of their time saying "no": the requested model does not exist, the input failed a check, the caller is not allowed. How you say no matters — a clean JSON error is something another program can act on, while an HTML stack trace is a leak and a dead end. This lesson covers three tools that make an API robust and maintainable: raising proper HTTP errors, sharing setup logic through dependency injection, and running code across every request with lifespan events and middleware.

## Raising HTTP errors with HTTPException

When a route cannot fulfill a request, it should return the right status code (Lesson 01) with a helpful message — not crash. FastAPI gives you `HTTPException` for exactly this. You `raise` it, and FastAPI turns it into a proper JSON response:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

models = {1: "resnet50", 2: "bert-base"}

@app.get("/models/{model_id}")
def get_model(model_id: int) -> dict:
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "name": models[model_id]}
```

Requesting `/models/99` returns HTTP `404` with the body `{"detail": "Model not found"}`. That `{"detail": ...}` shape is FastAPI's convention for every error, so callers can always look in the same place. Raising the exception also stops the function immediately — the `return` below never runs — which keeps your error handling as simple guard clauses at the top of a route.

## Custom exception handlers

Sometimes an error is raised deep inside your code, far from the route, and you do not want to sprinkle `HTTPException` everywhere. You can register a **handler** that catches a given exception type anywhere in your app and turns it into a consistent response:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
```

Now any `ValueError` that bubbles up out of a route — from your own helper functions, a library, anywhere — becomes a clean `400` with a JSON body instead of an ugly `500` and a stack trace. This is how you translate your domain's own exceptions into HTTP responses in one place. Define your own exception classes for real projects and give each a handler that maps it to the correct status code.

## Dependency injection with Depends

Many routes need the same thing before they run: a database session, the current user, a shared config object. Copying that setup into every route is repetitive and error-prone. FastAPI's **dependency injection** lets you write the setup once as a function and *declare* that a route needs it. FastAPI calls the dependency and passes the result in:

```python
from typing import Annotated
from fastapi import Depends, FastAPI

app = FastAPI()

def get_settings() -> dict:
    # In real code this reads from env vars / pydantic-settings.
    return {"model_dir": "/models", "max_batch": 32}

@app.get("/config")
def show_config(settings: Annotated[dict, Depends(get_settings)]) -> dict:
    return {"max_batch": settings["max_batch"]}
```

The `Annotated[dict, Depends(get_settings)]` says "before running this route, call `get_settings` and hand me the result as `settings`." A dependency can itself depend on other dependencies, so you can build a small chain — `get_current_user` depending on `get_db`, for example. Dependencies are also the natural place for an **auth check**: a dependency that inspects the request and raises `HTTPException(status_code=401)` if the caller is not authorized (Lesson 08 builds this out). The payoff is that shared logic lives in one testable function, and Lesson 10 shows how you can *override* a dependency in tests to swap the real database for a fake one.

## Lifespan events — load a model once

Loading an ML model can take seconds and megabytes. You must not do it inside a route, or you would reload it on every request. Instead, load it **once at startup** and reuse it. FastAPI's `lifespan` handler runs setup code before the app starts serving and cleanup code after it stops:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

ml_models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["clf"] = load_model()   # runs once, at startup
    yield
    ml_models.clear()                 # runs once, at shutdown

app = FastAPI(lifespan=lifespan)
```

Everything before the `yield` runs at startup; everything after runs at shutdown, so it is the right place to release resources — close database pools, flush logs, free GPU memory. This `lifespan` approach **replaces the older `@app.on_event("startup")` and `@app.on_event("shutdown")` decorators**, which are deprecated; if you see `@app.on_event` in a tutorial, it is out of date. Lesson 07 uses this pattern in earnest to serve a real model.

## Middleware — code around every request

Middleware wraps every request and response, letting you run logic that is not tied to any single route. Two everyday uses are timing/logging and enabling cross-origin browser requests. A timing middleware measures how long each request took and adds it as a header:

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response
```

`call_next(request)` runs the rest of the pipeline (other middleware and the actual route) and hands back the response, so you can act before *and* after. This is where request logging belongs — one place that records every call.

FastAPI also ships ready-made middleware. The most important is **CORS**, which controls whether browsers on other domains may call your API (the full story is in Lesson 08):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.example.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Always return JSON errors

The thread running through this lesson is a single rule: an API's errors should be **structured JSON, never HTML stack traces**. A stack trace shown to a caller leaks your file paths and internal logic, and no client program can parse it. FastAPI already does the right thing — validation failures, `HTTPException`, and your custom handlers all produce `{"detail": ...}` — so the job is mostly to *not undo it*: catch your own exceptions, map them to status codes, and never let a raw `500` with a traceback reach the client. In production you also disable debug mode so tracebacks stay in your logs where they belong.

## Key takeaways

- Raise `HTTPException(status_code=..., detail=...)` to return a proper error; FastAPI renders it as `{"detail": ...}` and stops the route.
- Register `@app.exception_handler(SomeError)` to translate an exception type into a consistent JSON response from one place.
- `Depends()` injects shared setup — config, a db session, an auth check — so routes declare what they need instead of duplicating it.
- Use an `@asynccontextmanager` `lifespan` to load a model once at startup and clean up at shutdown; this replaces the deprecated `@app.on_event`.
- Middleware wraps every request/response — ideal for timing and logging; `call_next` runs the rest of the pipeline.
- Add `CORSMiddleware` to control which browser origins may call your API.
- Always return structured JSON errors, never HTML stack traces — they leak internals and no client can parse them.

## Try it

Extend the model-registry API from earlier lessons. Add a `GET /models/{model_id}` route that raises `HTTPException(status_code=404, detail="Model not found")` for unknown ids, and confirm the JSON body. Write a `get_settings` dependency returning a small config dict and inject it into two different routes with `Depends()`. Add a timing middleware that sets an `X-Process-Time` header, then call any endpoint with `curl -i` and read that header off the response. Finally, register an `@app.exception_handler(ValueError)` that returns a `400`, raise a `ValueError` from inside a route, and verify you get a clean `{"detail": ...}` instead of a stack trace.
