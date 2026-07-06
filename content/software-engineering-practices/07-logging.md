# 07 — Logging Done Right

When something goes wrong in a program that is running on your laptop, you can watch it happen. When something goes wrong in a program running on a server at three in the morning, all you have is what the program wrote down. Logging is how a program keeps a running account of what it did, so that later — often much later, and often on a machine you cannot see — you can reconstruct what happened. This lesson teaches you to log well: the standard-library tool first, and a production-grade upgrade second.

## Rule one: stop using print for debugging

The first thing to internalise is that `print()` is not a logging tool. A `print()` call always writes to standard output, always at the same volume, with no timestamp, no severity, and no way to switch it off without editing code. When you sprinkle `print("here")` through a program to see what it is doing, you are building something you will have to tear out again, and that tells you nothing about *when* or *how serious* an event was.

Logging solves all of that. A single line configures where messages go and how much detail you want. Each message carries a severity level, so you can ask for only the important ones in production and all of them while debugging — without touching the code.

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Model loaded", extra={"path": "model.pkl"})
logger.warning("Prediction latency high: %d ms", 450)
```

`logging.getLogger(__name__)` is the idiom you will use in every module. Passing `__name__` names the logger after the module it lives in, so log output tells you which file a message came from, and you can later tune verbosity per module.

## The five levels

Every log message has a level that says how serious it is. There are five, from least to most severe:

- **DEBUG** — fine-grained detail useful only when diagnosing a problem. Shapes of arrays, intermediate values, "entering function X."
- **INFO** — normal, expected events worth recording. "Server started", "Loaded 1,240 rows", "Request completed in 90 ms."
- **WARNING** — something unexpected happened, but the program carried on. "Cache miss", "Retrying after timeout", "Config value missing, using default."
- **ERROR** — an operation failed. A request could not be served, a file could not be written. The program survives but something did not work.
- **CRITICAL** — the program itself cannot continue. Out of memory, database unreachable at startup.

The skill is choosing honestly. If you log everything at ERROR, real errors drown in noise. A good rule: INFO for the story of what the program is doing, DEBUG for the details you would only want while investigating, WARNING and above for things a human might need to act on.

## Configuring where logs go

By default a fresh logger sends nothing anywhere useful. The quickest way to get output during development is `basicConfig`, called once at program start:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("Ready")
# 2026-07-06 14:20:01,123 INFO __main__ Ready
```

`level=logging.INFO` means "show INFO and above, hide DEBUG." Change one word to `logging.DEBUG` and every debug message appears — no code deletion required. That switch is the whole point.

Two concepts sit underneath `basicConfig`. A **handler** decides *where* messages go — the console, a file, a network service. A **formatter** decides *what each line looks like*. For real applications you set these up explicitly:

```python
import logging

logger = logging.getLogger("myapp")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console)
```

In development you want a human-readable console handler like this one. In production you usually want a different format entirely — which brings us to structure.

## Freeform strings vs. structured logging

Look at a typical human-friendly log line:

```
ERROR: prediction failed for user 8842 on model v3 after 450ms
```

A person reads that easily. A machine does not. In production your logs do not go to a terminal a human is watching — they go to an aggregator such as Datadog, CloudWatch, or Grafana Loki, where you search millions of lines. To ask "show me every failure for model v3 where latency exceeded 400 ms," the aggregator needs the values as separate, named fields, not buried in prose.

**Structured logging** means emitting log events as key–value pairs rather than sentences:

```json
{"level": "error", "event": "prediction_failed", "user_id": 8842, "model": "v3", "latency_ms": 450}
```

Now `model` and `latency_ms` are fields you can filter and aggregate. This is why production systems log JSON: it is machine-parseable and queryable. The stdlib can produce JSON with a custom formatter, but there is a cleaner tool built for exactly this.

## structlog — the production upgrade

`structlog` (version 26.x) is the standard structured-logging library, and it is worth learning once you have the stdlib basics down. You install it with `uv add structlog`. Its two big ideas are a **processor pipeline** and **bound context**.

```python
import structlog

log = structlog.get_logger()

log.info("prediction_completed", model="v3", latency_ms=88, user_id=8842)
```

Instead of formatting a sentence, you pass an event name and keyword fields. A pipeline of *processors* then transforms that event — adding a timestamp, adding the level, and finally rendering it (colourful key–value pairs for the console in dev, compact JSON in prod). You configure which renderer to use per environment, so the same log calls produce friendly output locally and JSON on the server.

The second idea, **binding context**, removes enormous repetition. In a web request you want every log line to carry the request id and user id. Rather than passing them into every call, you bind them once:

```python
log = structlog.get_logger()
request_log = log.bind(request_id="abc-123", user_id=8842)

request_log.info("request_received", path="/predict")
request_log.info("prediction_completed", latency_ms=88)
# both lines automatically include request_id and user_id
```

`bind()` returns a new logger carrying that context, and every message from it inherits those fields. This is how you trace a single request across dozens of log lines in an aggregator.

structlog integrates with stdlib `logging` — it can route through the same handlers, so libraries that use stdlib logging and your own structlog calls end up in one stream. Teach yourself the stdlib first, because it is always present and underlies everything; reach for structlog when you ship to production and want queryable events.

## Key takeaways

- Never use `print()` for debugging or observability — use `logging`, which carries severity, timestamps, and an off switch.
- Create a logger per module with `logging.getLogger(__name__)`.
- Choose levels honestly: DEBUG for diagnostics, INFO for the normal story, WARNING for surprises, ERROR for failed operations, CRITICAL for "cannot continue."
- `basicConfig` gets you logging in one line; handlers decide *where* messages go and formatters decide *what they look like*.
- Human-readable strings suit the dev console; production logs to aggregators as structured JSON so fields are searchable.
- `structlog` (26.x) is the production upgrade: a processor pipeline and `bind()` for attaching request/user context to every line.

## Try it

Take a small script you have written — anything with a few functions. Replace every `print()` with logging: add `logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(message)s")` at the top and a module logger via `getLogger(__name__)`. Assign each old print a level: DEBUG for internal detail, INFO for normal progress, WARNING or ERROR for anything that went wrong. Run it once at `level=logging.DEBUG` and watch every line appear, then change one word to `level=logging.INFO` and watch the debug lines vanish without touching anything else. Finally, `uv add structlog`, rewrite two of your INFO lines as `structlog` events with key–value fields, and compare how the two styles read.
