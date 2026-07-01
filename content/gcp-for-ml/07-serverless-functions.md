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

## Triggers and event flow

Eventarc is what makes functions the connective tissue of an ML system. A typical event chain: a client uploads a batch-prediction input to Cloud Storage → a Storage-triggered function validates it and publishes to Pub/Sub → that message triggers a function (or a Cloud Run job) that submits a Vertex AI batch prediction → completion publishes another event that triggers a function to notify the caller. Each hop is a small, independently scalable, pay-per-use function. Because Pub/Sub delivers at-least-once, your event functions should be **idempotent** — processing the same event twice must be safe.

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

## Try it

Wire an event-driven reaction into a data pipeline:

1. Deploy a Cloud Storage-triggered `--gen2` function that logs the name of every new object in a data bucket and publishes a message to a Pub/Sub topic.
2. Upload a file to the bucket and confirm in Cloud Logging that the function fired and published.
3. Deploy a second, Pub/Sub-triggered function that consumes that topic and (for now) just logs the payload — you have now built a two-hop event chain.
4. Give both functions a dedicated service account with only the roles they need (`pubsub.publisher` for the first, `pubsub.subscriber`/invoker plumbing for the second), and confirm they still work — then try removing a role and observe the failure, seeing least privilege enforced end to end.
