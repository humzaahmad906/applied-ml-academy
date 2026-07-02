# 07 — Serverless: Cloud Functions

Not every piece of an ML system deserves a container, a cluster, or an always-on server. A lot of the glue — react to a new file, transform an incoming event, kick off a pipeline, call a model and post the result — is small, event-driven, and stateless. That is exactly what serverless **functions** are for: you write a single function, Google Cloud runs it on demand, scales it automatically, and you pay only while it executes. In the ML world, functions are the reflexes of your system: fast, cheap, event-triggered reactions that stitch bigger services together.

## Naming: Cloud Functions is now Cloud Run functions

Google has consolidated its function-as-a-service offering under **Cloud Run functions** (the product page reads "Cloud Run functions, formerly known as Cloud Functions"). Functions are built into containers and run on the Cloud Run infrastructure, which gives them Cloud Run's scaling, concurrency, and networking while keeping the simple "just deploy my function" developer experience. The modern deploy path is `gcloud run deploy --function=<entry-point>` (which builds and deploys your function as a Cloud Run service); the legacy `gcloud functions deploy` command still works for backward compatibility, and older **1st-gen** functions still exist. For anything new, target the current generation. The mental model: *write a function, get a scalable containerized service without managing the container.*

Because they run on Cloud Run, the current-generation functions inherit generous limits: up to 60-minute timeouts for HTTP (9 minutes for event-driven), up to 16 GiB memory and 4 vCPU, and up to 1,000 concurrent requests per instance — a big step up from the one-request-per-instance, 9-minute-max 1st gen.

## The programming model

You write a function in a supported runtime (Python, Node.js, Go, Java, and others) that is invoked in one of two shapes:

- **HTTP functions** — invoked by an HTTPS request; they receive a request and return a response. Good for lightweight webhooks and simple inference endpoints.
- **Event-driven functions** — invoked by an event via **Eventarc**, the routing layer that delivers events from across Google Cloud. Common ML triggers:
  - **Cloud Storage** — a new object appears (a file is uploaded), firing a function to validate, register, or preprocess it.
  - **Pub/Sub** — a message is published, firing a function to handle a streaming event.
  - **Firestore/other** — data changes.
  - Direct HTTP.

A minimal Python HTTP function using the Functions Framework:

```python
import functions_framework

@functions_framework.http
def classify(request):
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    # ... call a lightweight model or a Vertex/Gemini endpoint ...
    return {"label": "spam" if "win $$$" in text.lower() else "ham"}
```

A Cloud Storage-triggered function that reacts to new data:

```python
import functions_framework

@functions_framework.cloud_event
def on_new_object(cloud_event):
    data = cloud_event.data
    bucket, name = data["bucket"], data["name"]
    print(f"New object gs://{bucket}/{name} — enqueueing for preprocessing")
    # e.g. publish to Pub/Sub, or trigger a Vertex AI pipeline run
```

Deploying:

```bash
gcloud functions deploy on-new-object \
  --gen2 \
  --runtime=python312 \
  --region=us-central1 \
  --source=. \
  --entry-point=on_new_object \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=myco-fraud-data" \
  --service-account=fn-sa@myco-fraud-dev.iam.gserviceaccount.com \
  --no-allow-unauthenticated
```

Note the dedicated **service account** and `--no-allow-unauthenticated`: functions run as an identity and should require authentication by default, exactly the least-privilege posture from the security module.

The deploy command carries the full set of runtime and trigger flags, and this is where you tune behavior in practice. The **trigger** flags determine how the function is invoked — pick exactly one:

- `--trigger-http` — an HTTP function reachable at an HTTPS URL.
- `--trigger-topic=NAME` — fire on every message published to a Pub/Sub topic (shorthand for the equivalent Eventarc filter).
- `--trigger-bucket=NAME` — fire on object finalize/delete in a Cloud Storage bucket (shorthand).
- `--trigger-event-filters="type=...,..."` — the general **Eventarc** form; repeat the flag to add filters (an event `type` plus resource filters like `bucket=` or `serviceName=`). Use this for the full range of Eventarc sources, Audit-Log-based triggers, and multi-attribute matching.

The **scaling and resource** flags map straight onto the underlying Cloud Run service. `--min-instances` keeps N instances warm (kills cold starts for latency-sensitive glue; costs money at idle), `--max-instances` caps fan-out so a burst of events cannot blow your budget or hammer a downstream database, `--concurrency` sets how many requests one instance handles at once (>1 amortizes cold starts and cost for I/O-bound work; keep it at 1 for CPU-bound inference), and `--memory` / `--cpu` / `--timeout` size each instance (up to 16 GiB / 4 vCPU, 60-minute HTTP / 9-minute event-driven). Config and secrets come in via `--set-env-vars`, `--set-secrets` (mount a Secret Manager version as an env var or file — never bake keys into source), and `--run-service-account` (the identity the running container uses, as distinct from the build identity). A production-shaped deploy:

```bash
gcloud functions deploy fraud-score \
  --gen2 --region=us-central1 --runtime=python312 \
  --source=. --entry-point=score \
  --trigger-http --no-allow-unauthenticated \
  --min-instances=1 --max-instances=50 --concurrency=20 \
  --memory=1Gi --cpu=1 --timeout=120s \
  --set-env-vars=MODEL_BUCKET=myco-fraud-models,LOG_LEVEL=info \
  --set-secrets=API_KEY=projects/myco-fraud-prod/secrets/scoring-api-key:latest \
  --run-service-account=serving-sa@myco-fraud-prod.iam.gserviceaccount.com
```

### The modern deploy path: `gcloud run deploy --function`

Because current-generation functions *are* Cloud Run services, you can deploy them straight through the Cloud Run surface with `gcloud run deploy --function=<entry-point>`. This gives you the function build experience (no Dockerfile — buildpacks detect your runtime from the source) while exposing the full Cloud Run flag set. It is the forward-looking path; `gcloud functions deploy --gen2` remains fully supported and is what most existing tooling uses.

```bash
gcloud run deploy fraud-score \
  --function=score \
  --source=. \
  --base-image=python312 \
  --region=us-central1 \
  --no-allow-unauthenticated
```

### Day-two operations

Deploying is the easy part; you spend far more time inspecting, invoking, and cleaning up. These are the commands you run constantly:

```bash
# List and inspect
gcloud functions list --regions=us-central1
gcloud functions describe fraud-score --region=us-central1 --gen2

# Read logs (last 5 min of executions, most recent first)
gcloud functions logs read fraud-score --region=us-central1 --gen2 --limit=50

# Invoke directly for a smoke test (bypasses the network path)
gcloud functions call fraud-score --region=us-central1 --gen2 \
  --data='{"amount": 940.0, "merchant_category": "electronics"}'

# Delete
gcloud functions delete fraud-score --region=us-central1 --gen2
```

Grant callers the right to invoke an authenticated function without opening it to the world with `add-invoker-policy-binding` — the least-privilege alternative to `--allow-unauthenticated`:

```bash
gcloud functions add-invoker-policy-binding fraud-score \
  --region=us-central1 \
  --member=serviceAccount:caller-sa@myco-fraud-prod.iam.gserviceaccount.com
```

To see the Eventarc plumbing behind an event-driven function — useful when a trigger silently stops firing — list the triggers directly:

```bash
gcloud eventarc triggers list --location=us-central1
gcloud eventarc triggers describe TRIGGER_NAME --location=us-central1
```

### gen1 vs gen2 — and why it matters

You will still encounter **1st-gen** functions in older projects, and the difference is not cosmetic. Gen1 caps out at one request per instance, ~8 GiB memory, and a 9-minute timeout, uses a different (legacy) event format, and can only target a fixed set of trigger types. Gen2 (Cloud Run functions) gives you concurrency, up to 16 GiB / 4 vCPU, 60-minute HTTP timeouts, and the full Eventarc trigger surface. Deploy new work with `--gen2` (or the `gcloud run deploy --function` path); migrate gen1 functions deliberately, since the event payload shape and the required Functions Framework decorators differ between generations.

## Triggers and event flow

Eventarc is what makes functions the connective tissue of an ML system. A typical event chain: a client uploads a batch-prediction input to Cloud Storage → a Storage-triggered function validates it and publishes to Pub/Sub → that message triggers a function (or a Cloud Run job) that submits a Vertex AI batch prediction → completion publishes another event that triggers a function to notify the caller. Each hop is a small, independently scalable, pay-per-use function. Because Pub/Sub (and Eventarc generally) delivers **at-least-once**, your event functions should be **idempotent** — processing the same event twice must be safe. The practical pattern is to derive a stable key from the event (the Cloud Storage object generation, the Pub/Sub message ID) and skip work you have already done, so a redelivery is a no-op rather than a double-charge or a duplicate row. A related gotcha: event-driven functions are capped at a **9-minute** timeout even on gen2 (only HTTP functions get the full 60 minutes), so an event handler that might run long should hand off to a Cloud Run job rather than doing the work inline.

## Limits — and when a function is the wrong tool

Functions are deliberately constrained, and knowing the limits keeps you from misusing them:

- **Execution time.** There is a maximum timeout per invocation (generous for the current generation, but still bounded). A multi-minute reaction is fine; a multi-hour training job is not — that belongs on Vertex AI or a Cloud Run job.
- **Memory and CPU.** You can allocate a meaningful but capped amount of memory/CPU per instance. Enough for lightweight inference, feature transforms, and orchestration; **not** enough to load and serve a large GPU model — functions do not give you GPUs.
- **Request/response size.** Payloads are capped, so functions suit small JSON events, not streaming gigabytes.
- **Cold starts.** An idle function scales to zero; the next call pays a startup cost. Set **min instances** to keep some warm if latency matters, or accept cold starts for infrequent glue work.
- **Statelessness.** No local state survives between invocations; persist to Cloud Storage, a database, or Pub/Sub.

The decision rule: **use a function for short, event-driven, stateless glue and lightweight inference.** For sustained or GPU-backed serving, use **Cloud Run** (which gives you GPUs, longer requests, and fine concurrency control) or a **Vertex AI endpoint**. For long batch work, use a **Cloud Run job** or Vertex AI batch prediction. For heavy orchestration, use **Vertex AI Pipelines** or **Cloud Composer**.

## Serving lightweight models

Functions *can* serve a small model — a scikit-learn classifier, a compact embedding model, or a proxy that calls a Gemini model on Vertex AI. Load the model once in the global scope (so it is reused across invocations on a warm instance), read any API keys from Secret Manager, and return predictions. This is a perfectly good pattern for low-traffic, CPU-only inference where standing up an endpoint would be overkill. The moment you need GPUs, high concurrency, or large models, graduate to Cloud Run or Vertex.

## How this fits the whole solution

In the end-to-end architecture, functions are the low-cost reflexes between the big services. They fire when data lands, route events through Pub/Sub, trigger pipelines and batch jobs, and expose thin HTTP shims in front of models. They scale to zero when nothing is happening, so this reactive layer costs almost nothing at idle. Used for glue and light inference — and not stretched into roles that belong to Cloud Run, GKE, or Vertex AI — functions keep the system loosely coupled, event-driven, and cheap.

## Key takeaways

- The current offering is **Cloud Run functions** (the evolution of Cloud Functions 2nd gen), built on Cloud Run infrastructure; deploy new work with `--gen2`.
- Functions are **HTTP** or **event-driven** (via **Eventarc**), with common ML triggers on **Cloud Storage** and **Pub/Sub**; make event handlers **idempotent**.
- Respect the limits — **bounded timeout, capped memory/CPU, no GPUs, size limits, cold starts, statelessness** — and use functions only for **short, stateless glue and lightweight inference**.
- Run functions under a **dedicated service account**, require authentication by default, and read secrets from **Secret Manager**.

## CLI cheat-sheet

```bash
# --- Deploy (gen2 / Cloud Run functions) ---
# HTTP function, authenticated, warm + capped, with secrets and env
gcloud functions deploy fraud-score --gen2 --region=us-central1 \
  --runtime=python312 --source=. --entry-point=score \
  --trigger-http --no-allow-unauthenticated \
  --min-instances=1 --max-instances=50 --concurrency=20 \
  --memory=1Gi --cpu=1 --timeout=120s \
  --set-env-vars=MODEL_BUCKET=myco-fraud-models \
  --set-secrets=API_KEY=projects/myco-fraud-prod/secrets/scoring-api-key:latest \
  --run-service-account=serving-sa@myco-fraud-prod.iam.gserviceaccount.com

# Event triggers (pick one)
gcloud functions deploy on-new-object --gen2 --region=us-central1 \
  --runtime=python312 --source=. --entry-point=on_new_object \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=myco-fraud-data"
gcloud functions deploy on-msg --gen2 ... --trigger-topic=transactions
gcloud functions deploy on-file --gen2 ... --trigger-bucket=myco-fraud-data

# Modern Cloud Run path (buildpacks, no Dockerfile)
gcloud run deploy fraud-score --function=score --source=. \
  --base-image=python312 --region=us-central1 --no-allow-unauthenticated

# --- Day-two ops ---
gcloud functions list --regions=us-central1
gcloud functions describe fraud-score --region=us-central1 --gen2
gcloud functions logs read fraud-score --region=us-central1 --gen2 --limit=50
gcloud functions call fraud-score --region=us-central1 --gen2 --data='{"amount":940}'
gcloud functions delete fraud-score --region=us-central1 --gen2

# Grant invoke to a specific caller (least privilege vs --allow-unauthenticated)
gcloud functions add-invoker-policy-binding fraud-score --region=us-central1 \
  --member=serviceAccount:caller-sa@myco-fraud-prod.iam.gserviceaccount.com

# Inspect the Eventarc plumbing behind an event-driven function
gcloud eventarc triggers list --location=us-central1
gcloud eventarc triggers describe TRIGGER_NAME --location=us-central1
```

## Try it

Wire an event-driven reaction into a data pipeline:

1. Deploy a Cloud Storage-triggered `--gen2` function that logs the name of every new object in a data bucket and publishes a message to a Pub/Sub topic.
2. Upload a file to the bucket and confirm in Cloud Logging that the function fired and published.
3. Deploy a second, Pub/Sub-triggered function that consumes that topic and (for now) just logs the payload — you have now built a two-hop event chain.
4. Give both functions a dedicated service account with only the roles they need (`pubsub.publisher` for the first, `pubsub.subscriber`/invoker plumbing for the second), and confirm they still work — then try removing a role and observe the failure, seeing least privilege enforced end to end.
