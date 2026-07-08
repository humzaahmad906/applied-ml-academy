# 09b — WebSockets, File Uploads, and Background Tasks

Lesson 07 served a model behind a request-response endpoint, and Lesson 07b added streaming for the case where the answer arrives token by token. Both are still fundamentally *one-directional*: the client asks, the server answers. This lesson rounds out the ML-serving toolkit with three patterns that show up constantly in real systems and do not fit the plain request-response mold: **WebSockets** for two-way live conversations, **file uploads** for the image-in / prediction-out shape at the heart of computer-vision APIs, and **background tasks** for work you want to happen *after* you have already replied. We finish with the batch endpoint, the single easiest throughput win you can add to any model server.

## WebSockets: two-way, not one-way

It is worth being precise about how a WebSocket differs from the streaming you saw in Lesson 07b. Server-Sent Events (SSE) and `StreamingResponse` are **one-way**: the server pushes a sequence of chunks down to the client, but the client cannot say anything back over that same connection. That is exactly right for streaming an LLM's answer — the tokens only flow one direction.

A **WebSocket** is **bidirectional**. After an initial handshake, both sides hold one long-lived connection open and either side can send a message at any time. That two-way freedom is the whole reason to reach for it. If your data only flows server-to-client, SSE is simpler and you should prefer it. Choose a WebSocket when the client also needs to send messages *during* the exchange — a live chat, a collaborative editor, or a voice/vision assistant where the user streams audio up while the model streams text down.

FastAPI exposes WebSockets through the `@app.websocket` decorator. Here is a complete echo endpoint — the "hello world" of WebSockets:

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/echo")
async def echo(websocket: WebSocket):
    await websocket.accept()                 # complete the handshake
    try:
        while True:
            message = await websocket.receive_text()
            await websocket.send_text(f"you said: {message}")
    except WebSocketDisconnect:
        print("client disconnected")         # loop ends cleanly
```

Three things make this tick. `await websocket.accept()` completes the handshake — until you call it, no messages flow. The `while True` loop is the norm, not a smell: a WebSocket handler stays alive for the life of the connection, blocking on `receive_text()` until the client sends something. And when the client closes the tab, `receive_text()` raises `WebSocketDisconnect`, which is how you break out and clean up. The handler is naturally `async def` because every send and receive is awaited I/O.

For a chat that broadcasts to *everyone*, you keep a list of the active connections and fan each message out:

```python
active: list[WebSocket] = []

@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    active.append(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            for conn in active:
                await conn.send_text(text)   # broadcast to all
    except WebSocketDisconnect:
        active.remove(websocket)
```

That in-memory list works for one process. The moment you scale to multiple workers, each has its own list and they cannot see each other's clients — at that point you route messages through a shared broker like Redis pub/sub. That is a scaling detail, not a beginner concern, but knowing the ceiling exists saves you a confusing afternoon later.

## File uploads: the image-in / prediction-out shape

The canonical computer-vision API takes an uploaded image and returns a prediction. Unlike JSON bodies (Lesson 05), file uploads arrive as `multipart/form-data`, and FastAPI models them with `UploadFile`:

```python
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

app = FastAPI()

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="expected an image")

    raw = await file.read()                  # bytes of the whole file
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    model = app.state.model                  # loaded once at startup (Lesson 07)
    label, score = model.predict(image)
    return {"label": label, "confidence": round(score, 4)}

# output:
# {"label": "tabby_cat", "confidence": 0.9723}
```

A few points earn their keep here. `UploadFile = File(...)` tells FastAPI this is a multipart file part, not a JSON field; `File(...)` marks it required. `UploadFile` is a *spooled* file — small uploads stay in memory, large ones spill to a temp file on disk automatically, so you do not blow up RAM on a big upload. `await file.read()` reads it asynchronously and gives you the raw bytes, which you wrap in `io.BytesIO` so PIL can decode it without ever touching the filesystem. We also check `content_type` and reject non-images with `415 Unsupported Media Type` — clients lie, so validate.

Note the route is `async def` even though inference is CPU-bound. That is deliberate: the `await file.read()` is genuine I/O and belongs on the event loop, but per Lesson 09 the *inference* is CPU work. For a heavy model, offload `model.predict` with `run_in_executor` so it does not block the loop, or make the whole route a plain `def`.

**Size limits matter.** Nothing above stops a client from uploading a 2 GB file and exhausting your disk. FastAPI does not cap upload size by default. The clean fix is to enforce a limit at your reverse proxy (nginx's `client_max_body_size`), but you can also guard in-app by reading in chunks and bailing once you cross a threshold:

```python
MAX_BYTES = 10 * 1024 * 1024                 # 10 MB

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    size, chunks = 0, []
    while chunk := await file.read(1024 * 1024):   # 1 MB at a time
        size += len(chunk)
        if size > MAX_BYTES:
            raise HTTPException(status_code=413, detail="file too large")
        chunks.append(chunk)
    return {"received_bytes": size}
```

Reading in chunks keeps memory flat regardless of file size and lets you reject an oversized upload before you have buffered the whole thing.

## Background tasks: fire-and-forget, and its honest limit

Sometimes you want to do something *after* sending the response — write an audit log, send a confirmation email, warm a cache. The caller should not wait for it. FastAPI's `BackgroundTasks` handles exactly this: you declare it as a parameter, register callbacks, and FastAPI runs them after the response is delivered.

```python
from fastapi import BackgroundTasks

def write_log(label: str, confidence: float):
    with open("predictions.log", "a") as f:
        f.write(f"{label}\t{confidence}\n")

@app.post("/predict-logged")
async def predict_logged(file: UploadFile, tasks: BackgroundTasks):
    raw = await file.read()
    label, score = app.state.model.predict(Image.open(io.BytesIO(raw)))
    tasks.add_task(write_log, label, score)   # runs AFTER the response
    return {"label": label, "confidence": score}
```

The client gets its prediction immediately; the log write happens afterward on the same process.

Two honest limits keep you out of trouble. First, a gotcha specific to uploads: since FastAPI 0.106, the `UploadFile` is **closed before background tasks run**, so you cannot pass `file` into `add_task` and read it later — read the bytes *inside* the route (as above) and pass the bytes, not the file. Second, and more important: `BackgroundTasks` runs **in your web process**. It is perfect for quick, cheap side effects. It is the wrong tool for **slow inference** — a 30-second diffusion job in a background task ties up your worker, and if the process restarts the task is simply lost. There is no retry, no persistence, no visibility.

For real slow work you need a **task queue** — Celery or RQ backed by Redis. The shape is always the same:

```python
# 1. The endpoint enqueues the job and returns a ticket immediately.
@app.post("/jobs")
async def submit(file: UploadFile):
    raw = await file.read()
    job = queue.enqueue(run_heavy_inference, raw)   # RQ/Celery hands to a worker
    return {"job_id": job.id, "status": "queued"}

# 2. The client polls for the result with the ticket.
@app.get("/jobs/{job_id}")
async def status(job_id: str):
    job = queue.fetch_job(job_id)
    return {"job_id": job_id, "status": job.get_status(), "result": job.result}
```

The endpoint does no inference at all — it drops the job on the queue and returns a `job_id`. Separate **worker** processes (which you scale independently of your web servers) pick up jobs and run them. The client either polls `GET /jobs/{id}` until it is done or, in fancier setups, gets a callback/webhook. This *enqueue → return id → poll* pattern is how every serious "submit a long ML job" API works, and it is the natural next step once `BackgroundTasks` stops being enough.

## The batch endpoint: a list in, a list out

The cheapest throughput win in model serving is **batching**. Modern models — especially on a GPU — process a batch of N inputs in barely more time than a single one, because the fixed overhead (moving data to the device, kernel launch) is amortized across the batch. An endpoint that accepts one input at a time throws that away.

So offer a batch route: accept a list, run one batched forward pass, return a list in the same order.

```python
from pydantic import BaseModel

class BatchRequest(BaseModel):
    inputs: list[list[float]]                # N feature vectors

@app.post("/predict-batch")
def predict_batch(req: BatchRequest):
    preds = app.state.model.predict(req.inputs)   # ONE call, whole batch
    return {"predictions": [int(p) for p in preds]}

# output:
# {"predictions": [0, 2, 1]}
```

The key is that `model.predict(req.inputs)` is a *single* call over the whole batch, not a Python loop of one-at-a-time predictions. Two guardrails: cap the batch size (reject a 100,000-item request with `422` before it OOMs your GPU), and always return predictions in input order so callers can line results up with their inputs. A common production refinement is *dynamic batching* — a server-side buffer that collects individual requests for a few milliseconds and runs them together — but an explicit batch endpoint gives you most of the benefit with none of the complexity.

## Key takeaways

- WebSockets are **bidirectional** (both sides send anytime over one open connection); SSE/`StreamingResponse` from Lesson 07b is **one-way**. Prefer SSE unless the client must send during the exchange.
- A WebSocket handler calls `await websocket.accept()`, loops on `receive_text()`/`send_text()`, and catches `WebSocketDisconnect` to clean up. In-memory connection lists do not survive multiple workers — use Redis pub/sub to scale.
- File uploads arrive as multipart; model them with `UploadFile = File(...)`, read bytes with `await file.read()`, validate `content_type`, and enforce a size limit (proxy config or chunked reads) since FastAPI sets no default cap.
- `BackgroundTasks` is for quick fire-and-forget side effects (logging, email) that run after the response — but in-process, no retries, and the `UploadFile` is closed by then (pass bytes, not the file).
- Slow inference needs a real task queue (Celery/RQ + Redis): enqueue → return a `job_id` → poll or callback. The web process never runs the heavy work.
- A batch endpoint (list in, list out, one batched call) is the cheapest throughput win; cap the batch size and preserve input order.

## Try it

Build a small image classifier API (any pretrained model, or a stub that returns a fixed label). Add three routes: a `POST /predict-image` that reads an uploaded image with `await file.read()` and returns a prediction, rejecting non-images with `415` and files over 5 MB with `413`; a `POST /predict-batch` that accepts a list of inputs and returns a list of predictions in order; and a `BackgroundTasks` log write on the single-image route. Confirm the log line appears *after* the response returns. Then add a `@app.websocket("/ws")` echo endpoint and connect to it with a tiny client (`websockets` library or the browser console) — send three messages and watch each echo come straight back, then close the connection and confirm your `WebSocketDisconnect` handler fires. See Lesson 07 for loading the model once at startup, Lesson 07b for one-way streaming, and Lesson 09 for keeping CPU-bound inference off the event loop.
