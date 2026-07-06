# 09 — Async Basics and Concurrency for APIs

FastAPI lets you write routes two ways — with `def` or with `async def` — and choosing correctly is the difference between an API that handles hundreds of simultaneous callers smoothly and one that stalls under load. Lesson 07 gave you the one-line rule for model serving; this lesson explains the machinery behind it and shows when reaching for `async` genuinely helps. The goal is a practical mental model, not an asyncio deep-dive: you will finish knowing which keyword to type and why.

## Two kinds of routes

FastAPI handles both `def` and `async def` routes, but it runs them differently, and this is the whole game:

- A plain **`def` route runs in a thread pool.** FastAPI hands it to a separate worker thread so that if it does something slow or CPU-heavy, it does not freeze the rest of the app.
- An **`async def` route runs on the event loop** — a single thread that juggles many requests by switching between them whenever one is *waiting*.

The event loop is fast precisely because it never sits idle: while one request waits for a database or another API to respond, the loop serves other requests. But that only works if the waiting is done with `await`. If an `async` route does slow work *without* awaiting — a CPU-heavy loop, a blocking call — it hogs the single loop thread and every other request waits behind it.

## What "await" means

`await` marks a point where your code is *waiting for something external* and is happy to let the event loop go do other work in the meantime. You use it with libraries that support it. `httpx` (Lesson 03) offers an async client for exactly this:

```python
import httpx

async def fetch_model_card(model_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://registry.example.com/models/{model_id}")
        return response.json()
```

`await client.get(...)` says "send the request, and while the network round-trip is in flight, let the loop serve other requests; wake me when the response arrives." During that wait — which might be tens of milliseconds — the loop is not blocked. Multiply that across many concurrent callers and you see why async shines for I/O.

## When async helps — and when it does not

The distinction that decides everything is **I/O-bound versus CPU-bound work**:

- **I/O-bound** work spends its time *waiting* — for a network call to another API, a database query, reading a file. Here `async` is a big win, because the loop can serve other requests during the wait.
- **CPU-bound** work spends its time *computing* — running model inference, crunching a large array, parsing a huge payload. Here `async` does **not** help; the CPU is busy the whole time, there is nothing to await, and putting it in an `async def` route actively hurts by blocking the loop.

This is why Lesson 07 said model inference belongs in a plain `def` route: inference is CPU-bound, and a `def` route runs in a thread pool where its busyness does not freeze the event loop.

## Running I/O concurrently with gather

The real superpower of async is doing several independent I/O operations **at the same time**. Suppose a route needs data from three downstream services. Done sequentially, you wait for the sum of all three. With `asyncio.gather`, you fire them together and wait only for the slowest:

```python
import asyncio
import httpx

async def gather_features(user_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        profile, history, prefs = await asyncio.gather(
            client.get(f"https://svc/profile/{user_id}"),
            client.get(f"https://svc/history/{user_id}"),
            client.get(f"https://svc/prefs/{user_id}"),
        )
    return {
        "profile": profile.json(),
        "history": history.json(),
        "prefs": prefs.json(),
    }
```

If each call takes 100 ms, the sequential version takes ~300 ms and the `gather` version takes ~100 ms — the calls overlap. This pattern is common in ML serving when you assemble features from several sources before predicting.

## The classic mistakes

Two mistakes account for most async performance bugs, and both come down to **blocking the event loop**:

```python
import time
import asyncio

@app.get("/bad")
async def bad():
    time.sleep(5)          # WRONG: blocks the whole loop for 5 seconds

@app.get("/good")
async def good():
    await asyncio.sleep(5) # right: yields the loop while waiting
```

`time.sleep()` blocks the thread; in an `async` route that thread *is* the event loop, so every other request stalls for the full five seconds. The async-aware `asyncio.sleep()` yields control instead. The same trap applies to any blocking call inside an `async def`: a synchronous database driver, a `requests.get()` (Lesson 03's sync client), or CPU-heavy work. If you must do such work from an async route, hand it to a thread with `run_in_executor`:

```python
import asyncio

@app.post("/predict")
async def predict(req: IrisRequest, request: Request):
    model = request.app.state.model
    loop = asyncio.get_running_loop()
    # Offload the CPU-bound inference to a thread so the loop stays free.
    proba = await loop.run_in_executor(None, model.predict_proba, [req.features])
    return {"prediction": int(proba[0].argmax())}
```

## The practical rule

You do not need to reason about the event loop on every route. Two guidelines cover almost every case:

- If a route **only does I/O** — calling another API, awaiting a database — write it `async def` and `await` those calls.
- If a route does **CPU work** like model inference — write it as a plain `def` (FastAPI threads it automatically), or if you are already in an `async def`, offload it with `run_in_executor`.

When in doubt, a plain `def` route is the safe default: FastAPI runs it in the thread pool, so even slow or CPU-bound work will not freeze your whole service. Reach for `async def` when you have genuine concurrent I/O to overlap.

## Key takeaways

- FastAPI supports both `def` (run in a thread pool) and `async def` (run on the single-threaded event loop) routes.
- `await` marks a point where your code waits on external I/O and lets the event loop serve other requests in the meantime.
- Async helps I/O-bound work (network, DB, files); it does not help CPU-bound work like model inference — and blocking the loop with CPU work hurts.
- `asyncio.gather` runs independent I/O calls concurrently, so total time is the slowest call, not the sum.
- Never call `time.sleep()`, a sync HTTP client, or heavy CPU work directly inside an `async def` — it blocks the loop for everyone.
- Offload unavoidable blocking/CPU work from an async route with `run_in_executor`.
- Practical rule: I/O-only route → `async def`; CPU work → plain `def` or `run_in_executor`; when unsure, plain `def` is safe.

## Try it

Write two small `async` routes. The first, `/slow-bad`, calls `time.sleep(3)`; the second, `/slow-good`, calls `await asyncio.sleep(3)`. Start the server, then open two terminals and hit each route twice at once (e.g. with `curl` in the background). Observe that the two `/slow-bad` calls total about six seconds because the loop is blocked, while the two `/slow-good` calls finish in about three seconds because they overlap. Then write an `async` route that uses `httpx.AsyncClient` and `asyncio.gather` to fetch three public URLs concurrently, and confirm it is roughly as fast as the single slowest request rather than their sum.
