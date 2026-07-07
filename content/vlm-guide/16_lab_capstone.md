# Lab 7 — Capstone: Serve a Local RAG Agent Behind an API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/vlm-guide/notebooks/16_lab_capstone.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup. The full write-up and stack alternatives are below.

You wire Lab 5's FAISS retrieval index and Lab 6's ReAct agent into a single FastAPI service: a `POST /chat` endpoint that runs the agent (with a `retrieve_context` tool backed by the vector index) and a `GET /health` check. Models load once at startup via FastAPI's `lifespan` hook; the API surface stays stable across backend swaps. Makes the RAG and Agents theory chapters concrete as a deployed system.

## Setup

```bash
pip install transformers accelerate torch sentence-transformers faiss-cpu \
            fastapi uvicorn httpx pydantic
```

**Models:** `all-MiniLM-L6-v2` (~90 MB) for embeddings + `Qwen/Qwen2.5-1.5B-Instruct` (~3.1 GB bf16) for generation. Swap to `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) if VRAM is tight. On M-series Mac a 3-step query takes ~5–10 s; on a 4090 it is interactive.

## The RAG index

Recap from Lab 5 — `IndexFlatIP` over L2-normalized embeddings = cosine search. In production swap to `faiss.read_index` on a pre-built file; the index here builds in under a second.

```python
# rag_agent_service.py
import re, math, datetime, logging
from contextlib import asynccontextmanager

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

_DOCUMENTS = [
    "The Eiffel Tower stands 330 m (1,083 ft) tall, located in Paris, France.",
    "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "The Transformer architecture uses self-attention and feed-forward blocks (Vaswani 2017).",
    "FAISS is Facebook AI's library for efficient similarity search on dense vectors.",
    "RAG (Retrieval-Augmented Generation) grounds LLM outputs with retrieved external evidence.",
    "FastAPI is a modern Python web framework built on Starlette and Pydantic with OpenAPI docs.",
]


def build_faiss_index(docs: list[str], embed_model: SentenceTransformer):
    embs = embed_model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    return index, docs

def _retrieve(query: str, index, chunks: list[str], embed_model, top_k: int = 3) -> str:
    q = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, ids = index.search(q, top_k)
    return "\n".join(f"[{i+1}] {chunks[idx]}" for i, idx in enumerate(ids[0]) if idx >= 0)
```

## Tools

```python
def make_retrieve_context(index, chunks: list[str], embed_model) -> callable:
    """Closure — binds the index once; keeps the tool signature as (query: str) -> str."""
    return lambda query: _retrieve(query, index, chunks, embed_model)


def calculator(expression: str) -> str:
    safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe["__builtins__"] = {}
    try:
        return str(eval(expression, safe))  # noqa: S307
    except Exception as exc:
        return f"Error: {exc}"

def get_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## The agent

Lab 6 ReAct loop refactored to receive models + registry from app state. System prompt delta: `retrieve_context` is listed first and agents are told to use it before answering factual questions.

```python
_SYSTEM = """\
You answer questions using tools. At each turn emit EXACTLY ONE of:

  Thought: <reasoning>
  Action: tool_name(argument)
  Answer: <final answer>

Available tools:
  retrieve_context(query)  – search the local knowledge base for relevant facts
  calculator(expression)   – evaluate a Python math expression
  get_datetime()           – return the current date and time

Use retrieve_context for any factual question before answering.
"""

_ACTION_RE = re.compile(r"Action:\s*(\w+)\(([^)]*)\)")
_ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.DOTALL)


def _run_agent(
    question: str, registry: dict, tokenizer, model, device: str, max_steps: int = 8
) -> str:
    def _chat(msgs: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": question}]
    for _ in range(max_steps):
        resp = _chat(messages)
        m = _ANSWER_RE.search(resp)
        if m:
            return m.group(1).strip()
        m = _ACTION_RE.search(resp)
        if m:
            name, raw = m.group(1), m.group(2).strip()
            args = [a.strip().strip("\"'") for a in raw.split(",")] if raw else []
            if name not in registry:
                obs = f"Unknown tool '{name}'. Available: {list(registry)}"
            else:
                try:
                    obs = registry[name](*args)
                except Exception as exc:
                    obs = f"Tool error: {exc}"
            messages += [{"role": "assistant", "content": resp},
                         {"role": "user",      "content": f"Observation: {obs}"}]
        else:
            messages += [{"role": "assistant", "content": resp},
                         {"role": "user", "content": "Continue. Emit Action: or Answer:."}]
    return "Agent budget exhausted without a final answer."
```

## FastAPI application

`lifespan` (FastAPI 0.93+) populates `_state` once at startup — no per-request model loading, clean shutdown path.

```python
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("startup: device=%s", device)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    index, chunks = build_faiss_index(_DOCUMENTS, embed_model)
    logger.info("FAISS index: %d docs", len(chunks))

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    gen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16, device_map=device
    )
    gen.eval()
    logger.info("generation model ready")

    _state.update({
        "tokenizer": tok, "gen_model": gen, "device": device,
        "registry": {
            "retrieve_context": make_retrieve_context(index, chunks, embed_model),
            "calculator":       calculator,
            "get_datetime":     get_datetime,
        },
    })
    logger.info("service ready")
    yield
    _state.clear()


app = FastAPI(title="RAG Agent API", version="1.0.0", lifespan=lifespan)
```

## Pydantic schemas and endpoints

```python
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    max_steps: int = Field(default=8, ge=1, le=20)

class ChatResponse(BaseModel):
    question: str
    answer: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=bool(_state))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not _state:
        raise HTTPException(status_code=503, detail="Models still loading.")
    try:
        answer = _run_agent(
            req.question, _state["registry"],
            _state["tokenizer"], _state["gen_model"], _state["device"],
            req.max_steps,
        )
    except Exception as exc:
        logger.exception("agent error: %s", req.question)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(question=req.question, answer=answer)
```

## Testing

`TestClient` as a context manager triggers the full `lifespan` — real inference, not schema stubs.

```python
# test_service.py
from fastapi.testclient import TestClient
from rag_agent_service import app

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200 and r.json()["models_loaded"] is True

def test_chat_math():
    with TestClient(app) as client:
        r = client.post("/chat", json={"question": "What is 6 * 7?"})
    assert r.status_code == 200 and "42" in r.json()["answer"]

def test_validation():
    with TestClient(app) as client:
        assert client.post("/chat", json={"question": ""}).status_code == 422      # min_length=1
        assert client.post("/chat", json={"question": "hi", "max_steps": 0}).status_code == 422  # ge=1
```

Start the server and exercise with curl:

```bash
uvicorn rag_agent_service:app --host 0.0.0.0 --port 8000

curl http://localhost:8000/health

curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "How tall is the Eiffel Tower in feet?"}' | python3 -m json.tool
```

## Stacks & alternatives

The FastAPI routes, Pydantic schemas, and tool registry are unchanged. The generation backend is swappable — replace `_chat()` only. Both alternatives expose the same OpenAI-compatible `/v1/chat/completions` endpoint; switching is one URL string and one model name.

### vLLM or Ollama as the generation backend

```bash
# vLLM (high-throughput, PagedAttention, production):
pip install vllm && vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 11434 --dtype bfloat16

# Ollama (easiest local, GGUF quant, macOS Metal, ~800 MB 4-bit vs 3.1 GB bf16):
ollama pull qwen2.5:1.5b && ollama serve
```

```python
from openai import OpenAI

# vLLM config
_client = OpenAI(base_url="http://localhost:11434/v1", api_key="none")
_model  = "Qwen/Qwen2.5-1.5B-Instruct"

# Ollama: swap these two lines
# _client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# _model  = "qwen2.5:1.5b"

def _chat(messages: list[dict], **_) -> str:
    resp = _client.chat.completions.create(
        model=_model, messages=messages, max_tokens=256, temperature=0.0,
    )
    return resp.choices[0].message.content.strip()
```

vLLM: reach for production, multi-user load, or models over 7B. PagedAttention prevents KV-cache fragmentation; throughput is 5–10× a naïve HF `generate` loop. Ollama: reach for local dev, macOS Metal, or when the quantized model size matters.

## Build it further

**1. Streaming responses.** `StreamingResponse` + vLLM/Ollama's `stream=True` via the OpenAI client streams the agent's final answer token-by-token over SSE. Add `POST /chat/stream`. Deliver a working endpoint + a `curl --no-buffer` command that shows tokens arriving live.

**2. Multi-turn memory.** Add `session_id: str | None` to `ChatRequest`. Persist message history keyed by session ID (in-memory dict for dev, Redis in prod). Prepend prior turns to `messages`, capped at N tokens. Deliver a curl session that references a prior answer without restating the question.

**3. Eval harness.** `eval_harness.py`: 20 fixed QA pairs, POST each to `/chat`, measure exact-match and ROUGE-L, log to wandb, print — `backend | accuracy | avg_latency_s | avg_steps`. Run against all three backends. This converts "it feels right" into a number you can defend.

**4. Containerize.** `FROM python:3.11-slim`, copy service + requirements, `CMD ["uvicorn", "rag_agent_service:app", "--host", "0.0.0.0", "--port", "8000"]`. Mount model weights as a volume — never bake them into the image. Test with `docker build` + `docker run -p 8000:8000` + the curl commands from § Testing.

## What you built

- A production-shape FastAPI service: `lifespan` model/index loading, Pydantic-validated endpoints, structured error handling, clean shutdown.
- A RAG retriever wired as a first-class agent tool — `retrieve_context` grounds factual answers in real vector search, not model weights.
- The Lab 6 ReAct agent behind HTTP: the model never touches the web layer; the routes never touch the model directly.
- Three interchangeable generation backends (HF / vLLM / Ollama) behind the same API surface — the canonical local-to-production migration pattern.
