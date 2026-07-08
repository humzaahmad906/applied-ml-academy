# 07b — Streaming Responses and SSE

In Lesson 07 an endpoint took a request, ran inference, and returned one response. That works when the answer is ready in milliseconds. But the defining ML-API task of 2026 is different: you call a large language model, and the answer takes several seconds to generate — token by token. Making a user stare at a spinner for ten seconds while the whole reply is assembled is a bad experience when you could show the words *as they appear*, exactly like every chat interface you have used. This lesson is about that: how to stream a response out of a FastAPI endpoint, the specific case of **Server-Sent Events (SSE)**, and the canonical worked example — a FastAPI route that calls an LLM and forwards tokens to the browser as they generate.

## Why stream at all

A normal `return` builds the entire response in memory and sends it once, complete. Streaming instead sends the body in pieces over one open connection, so the client can start using data before the server is done producing it. Two situations make this essential:

- **LLM responses.** The model emits tokens sequentially over seconds. Buffering them into one final blob throws away the single biggest UX win available — showing text as it is written. Streaming turns a 10-second wait into a response that starts in under a second.
- **Large file or CSV downloads.** A million-row export would exhaust memory if you built the whole string first. Streaming yields it row by row, so memory stays flat regardless of size.

The mechanism is the same in both cases: hand FastAPI a **generator** that yields chunks, and it flushes each chunk to the client as it is produced.

## StreamingResponse with an async generator

FastAPI's `StreamingResponse` wraps a generator and turns each yielded piece into a chunk on the wire. Here is the simplest possible example — a route that yields three chunks with a pause between them, so you can watch them arrive one at a time:

```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def word_generator():
    for word in ["Streaming", " is", " easy"]:
        yield word
        await asyncio.sleep(0.5)  # simulate work between chunks

@app.get("/stream")
def stream():
    return StreamingResponse(word_generator(), media_type="text/plain")
```

The route itself returns *immediately* — it just hands the generator to `StreamingResponse`. FastAPI then drives the generator, sending each yielded string as it comes. The `await asyncio.sleep(0.5)` matters for a subtle reason: an async generator only yields control to the event loop at an `await`. If your generator does blocking CPU work with no `await`, it can starve every other request — so either put real `await` points in (as an LLM call naturally does), or write the generator with plain `def` for blocking work and let FastAPI run it in a thread pool.

That is raw byte streaming. For token-by-token LLM output to a browser, we want a slightly more structured wire format: SSE.

## Server-Sent Events: the format and when to use it

**Server-Sent Events** is a tiny standard for a server to push a stream of text events to a client over one long-lived HTTP connection. It is one-directional: server → client only. The wire format is deliberately simple — each event is a line beginning with `data:`, terminated by a **blank line**:

```
data: Hello

data: world

```

The two things that make it SSE rather than plain streaming are that framing (`data: <payload>\n\n` per event) and the media type **`text/event-stream`**, which tells browsers and any proxies in between "this is a live event stream — do not buffer it." Optional `event:` and `id:` fields can accompany the `data:` field for named event types and reconnection support.

Choosing between the three real-time transports comes down to direction:

- **Plain `StreamingResponse`** — a raw byte stream (a file download, an NDJSON feed). No event framing, no browser `EventSource` support.
- **SSE** — one-way, server → client, over ordinary HTTP. Perfect for streaming LLM tokens, progress updates, or a live log. Works through normal HTTP infrastructure and auto-reconnects in the browser.
- **WebSockets** — full-duplex, both directions at once. Reach for these only when the client must also send continuously, like a collaborative editor or a multiplayer game. They are more complex to run and secure; do not use them for a problem SSE already solves.

For "stream the LLM's answer to the user," the answer is almost always SSE.

You *can* format SSE by hand with `StreamingResponse` (yield `f"data: {chunk}\n\n"` with `media_type="text/event-stream"`), but the small details — blank-line framing, keep-alive pings, graceful shutdown — are easy to get wrong. The standard library **`sse-starlette`** handles them for you. Install it with `pip install sse-starlette`, then yield plain dicts and let `EventSourceResponse` do the framing:

```python
from sse_starlette import EventSourceResponse

@app.get("/sse")
async def sse():
    async def event_generator():
        for word in ["Hello", "world"]:
            yield {"data": word}       # becomes: data: Hello\n\n
    return EventSourceResponse(event_generator())
```

Each yielded dict becomes one SSE event; the media type is set to `text/event-stream` for you. You can add `"event"` and `"id"` keys for named events. That is the whole API you need.

## The real ML example: streaming LLM tokens as SSE

Now the canonical 2026 task. We call Claude with streaming enabled and forward each token to the client as an SSE event. The bridge is an async generator: the Anthropic SDK's streaming helper produces tokens, and our generator re-yields each one as an SSE dict.

The Anthropic Python SDK exposes streaming through `client.messages.stream(...)`, used as an async context manager. Its `text_stream` attribute is an async iterator over just the text deltas — exactly what we want to forward:

```python
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

app = FastAPI()
client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    async def token_stream():
        async with client.messages.stream(
            model="claude-opus-4-8",
            max_tokens=1024,
            messages=[{"role": "user", "content": req.prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield {"data": text}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(token_stream())
```

Read the `token_stream` generator carefully — it is the whole idea. The `async with client.messages.stream(...)` opens the streaming request to Claude. `async for text in stream.text_stream` gives us each token as it arrives from the model, and we immediately `yield {"data": text}`, which `EventSourceResponse` turns into a `data:` event on the wire. Tokens flow model → SDK → our generator → HTTP response → browser, one at a time, with no buffering. When the model finishes, we emit a final `done` event so the client knows the stream is complete.

Note we use `AsyncAnthropic` (the async client) because the route is `async` and the work is I/O-bound — waiting on the network, not burning CPU — so it belongs on the event loop, not in a thread pool. This is the opposite of the CPU-bound rule from Lesson 07: network waits are exactly what `async` is for.

## Consuming the stream on the client

A browser consumes SSE natively with `EventSource`:

```javascript
const source = new EventSource("/chat");   // GET; see note below
source.onmessage = (e) => {
    if (e.data === "[DONE]") { source.close(); return; }
    document.getElementById("out").textContent += e.data;
};
```

`EventSource` only issues GET requests, so for a POST body (as above) you either switch the route to GET with a query parameter, or use the `fetch`-based streaming APIs on the front end. From Python — say, a test or another service — stream it with `httpx`:

```python
import httpx

with httpx.stream("POST", "http://localhost:8000/chat",
                  json={"prompt": "Explain streaming in one sentence."}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            print(line.removeprefix("data: "), end="", flush=True)
```

And if the *client* is your own Python code talking directly to Claude (no FastAPI in between), you do not parse SSE by hand at all — the SDK's own helper does it: `async with client.messages.stream(...) as stream: async for text in stream.text_stream: ...`, and `await stream.get_final_message()` gives you the complete assembled message at the end.

## Gotchas

Streaming has a handful of traps that do not exist for normal responses:

- **Proxy buffering.** An `nginx` or load balancer in front of your app may buffer the whole response before forwarding it, silently defeating streaming. The `text/event-stream` media type helps; for `nginx` specifically you may also need `X-Accel-Buffering: no`. If your local dev works but production delivers everything at once, suspect the proxy.
- **Errors mid-stream.** Once the first chunk is sent, the HTTP status is already `200` — you cannot switch to a `500`. If the LLM call fails halfway, catch it inside the generator and yield an error *event* (e.g. `{"event": "error", "data": "..."}`) that the client checks for, rather than letting the connection die silently.
- **Keeping the connection alive.** Long gaps between tokens can cause intermediaries to drop an idle connection. `sse-starlette` sends periodic keep-alive pings automatically; this is one more reason to prefer it over hand-rolled SSE.
- **Client disconnects.** If the user closes the tab mid-stream, keep generating and you waste tokens (and money). The generator is cancelled when the client goes away, but only at an `await` point — another reason your generator must actually `await`.

## Key takeaways

- Streaming sends a response in pieces over one open connection, so clients use data before the server finishes — essential for LLM token output and large downloads.
- `StreamingResponse(generator, media_type=...)` is the base tool; the route returns immediately and FastAPI drives the generator, flushing each yielded chunk.
- SSE is one-way server→client streaming over HTTP: events framed as `data: ...\n\n` with media type `text/event-stream`. Use SSE for LLM tokens/progress; use WebSockets only when the client must also stream back.
- Prefer `sse-starlette`'s `EventSourceResponse` over hand-formatting SSE — yield plain dicts (`{"data": ...}`) and it handles framing, the media type, and keep-alive pings.
- The canonical ML pattern: an async generator bridges `client.messages.stream(...)` (from `AsyncAnthropic`) to the HTTP response — `async for text in stream.text_stream: yield {"data": text}`.
- Use the async client for the LLM call: it is I/O-bound, so it belongs on the event loop, not a thread pool (the opposite of CPU-bound inference in Lesson 07).
- Watch the gotchas: proxy buffering, errors after the first chunk (status is already `200` — yield an error event), idle-connection drops, and wasted tokens on client disconnect.

## Try it

Install `pip install fastapi uvicorn sse-starlette anthropic` and set `ANTHROPIC_API_KEY`. Build the `/chat` route above, start it with `uvicorn main:app --reload`, and hit it with the `httpx` client snippet — watch the answer print token by token instead of all at once. Then break something on purpose: wrap the `async with` block in a `try/except` and, in the `except`, `yield {"event": "error", "data": str(e)}` — pass a bad model name and confirm the client receives a clean error event mid-stream rather than a dropped connection. Finally, compare the two response styles: add a second non-streaming route that returns `await stream.get_final_message()` and time how long until the first byte reaches the client versus the streaming route. The difference is the whole point of this lesson.
