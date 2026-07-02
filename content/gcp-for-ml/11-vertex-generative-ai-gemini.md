# 11 — Vertex AI Generative AI and Gemini

Generative AI is now a first-class part of the ML engineer's toolkit, and on Google Cloud the home for it is Vertex AI's generative offering, built around the **Gemini** family of models plus **Model Garden**, grounding, retrieval-augmented generation, and tuning. Where earlier modules trained and served your own models, this one is about building on foundation models you call as a managed service — the fastest path to shipping intelligent features, and increasingly the backbone of production ML systems.

## The Gemini model family

**Gemini** is Google's family of multimodal foundation models, served on Vertex AI with enterprise controls (data governance, VPC, IAM). As of 2026 the lineup spans generations:

- **Gemini 2.5 Pro / 2.5 Flash / 2.5 Flash-Lite** — generally available, production-ready. Pro is the high-reasoning tier; Flash balances quality, latency, and cost (the default workhorse); Flash-Lite is the most cost-efficient, low-latency option.
- **Gemini 3 / 3.1 family** — the newest generation, with the top-tier **Gemini 3.1 Pro** (advanced reasoning, 1M-token context) and **Gemini 3 Flash** available in preview at the time of writing, plus image-generation variants.

Model versions have lifecycles and retirement dates, so pin an explicit model ID in production and track the release notes. The practical rule: default to a **Flash** model for most tasks (fast, cheap), reach for a **Pro** model when a task genuinely needs deeper reasoning or the largest context, and use **Flash-Lite** for high-volume, latency-critical calls. Gemini models are multimodal — they accept text, images, audio, video, and documents — and support long context, function calling, and structured output.

## The SDK: use google-genai

This is the most important practical point in the module. The generative-AI modules of the older Vertex AI SDK (`vertexai.generative_models` and related) are **deprecated and scheduled for removal on June 24, 2026**. The current, recommended SDK is **google-genai** (`pip install google-genai`), a single unified library that talks to both the Gemini API and Vertex AI. It uses a **client** pattern rather than the old module-level `vertexai.init()`:

```python
from google import genai
from google.genai import types

# Point the client at Vertex AI (uses your ADC credentials)
client = genai.Client(vertexai=True, project="myco-fraud-dev", location="us-central1")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Summarize this transaction dispute in one sentence: ...",
    config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=256),
)
print(response.text)
```

(Note the non-generative parts of the Vertex AI SDK — training, pipelines, prediction, model registry, covered in other modules via `from google.cloud import aiplatform` — are **not** deprecated. Only the generative submodules moved to google-genai.)

Under the hood the SDK is calling the Vertex `generateContent` REST endpoint, and it is worth seeing that raw call — it is what a non-Python service, a shell script, or a quick auth test hits. You authenticate with a short-lived OAuth token from `gcloud auth print-access-token` (no API key needed on Vertex):

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/myco-fraud-dev/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent" \
  -d '{"contents": {"role": "user", "parts": {"text": "Summarize this dispute in one sentence: ..."}}}'
```

The region prefix on the host (`us-central1-aiplatform.googleapis.com`) must match the `location` in the path. The newest models are often region-limited at launch — a **gotcha** worth checking the model card for, since a model available in `us-central1` may not yet be in your preferred region, and requests to a region that lacks it return `404`.

## Model Garden

**Model Garden** is the catalog of models available on Vertex AI. It includes Google's first-party models (Gemini, embeddings, image and video generation), a curated set of **partner models** (such as Anthropic's Claude and Meta's Llama, callable through the same Vertex surface), and hundreds of **open models** you can deploy to your own endpoints. It is where you discover a model, read its card, and either call it as a managed API or deploy it to an endpoint you control. For an ML engineer, Model Garden turns "which model should I use?" into a browse-and-try exercise rather than a procurement project, and lets you keep every model — first-party, partner, and open — behind one consistent Vertex AI interface with unified auth, logging, and billing.

You can browse and deploy from the command line. `gcloud ai model-garden models list` catalogs what is available (filter with `--model-filter=gemma`), and `gcloud ai model-garden models deploy` stands up an **open model** on a Vertex endpoint you own — self-hosted weights on your GPUs rather than a shared managed API:

```bash
gcloud ai model-garden models list --model-filter=gemma
gcloud ai model-garden models list --can-deploy-hugging-face-models

# Deploy a Google open model (Gemma) to your own endpoint
gcloud ai model-garden models deploy \
  --model=google/gemma3@gemma-3-9b \
  --region=us-central1 --accept-eula

# Deploy a gated Hugging Face model (Llama) — needs a token and EULA acceptance
gcloud ai model-garden models deploy \
  --model=meta-llama/Meta-Llama-3-8B \
  --hugging-face-access-token=$HF_TOKEN \
  --region=us-central1 --accept-eula
```

Deploying an open model gives you a standard Vertex endpoint (module 10) — you pay for the running replicas and manage autoscaling yourself — whereas calling Gemini or a partner model as a managed API is pay-per-token with no infrastructure to run. Choose self-hosting when you need a specific open model, data-residency control, or predictable high-volume cost; choose the managed API for the frontier models and zero ops.

## Embeddings

Not every generative task is text generation. **Embeddings** — dense vectors that place semantically similar text near each other — are the backbone of retrieval, RAG, clustering, and semantic search. Vertex serves dedicated embedding models (the `text-embedding-005` / `gemini-embedding` family), and the same google-genai client produces them with `embed_content`. You pass a `task_type` so the model optimizes the vector for how it will be used (`RETRIEVAL_DOCUMENT` when embedding your corpus, `RETRIEVAL_QUERY` when embedding a user question), which measurably improves retrieval quality:

```python
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="myco-fraud-dev", location="us-central1")

resp = client.models.embed_content(
    model="text-embedding-005",
    contents=["chargeback policy for disputed card-not-present transactions"],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",
                                     output_dimensionality=768),
)
vector = resp.embeddings[0].values
```

For embedding a whole corpus, do it as a **batch** job rather than a loop of online calls — cheaper and higher-throughput (see the batch section below).

## Grounding and RAG

Foundation models hallucinate and have a knowledge cutoff. Two features close that gap:

- **Grounding with Google Search** — let the model consult Google Search at inference time so answers reflect current, real-world information, with citations. You enable it as a tool on the request:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What are the current spot GPU prices for the fraud-scoring region?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    ),
)
```

- **Retrieval-augmented generation (RAG)** — ground the model in *your* private data. The **Vertex AI RAG Engine** manages the retrieval pipeline (ingest documents, chunk, embed, index, retrieve relevant context, and inject it into the prompt) so the model answers from your corpus. For search-heavy applications, **Vertex AI Search** (part of the broader Agent Builder toolset) provides a managed, Google-quality retrieval layer over your data that you can wire into a generative app. RAG is how you build a support assistant that answers from your documentation, or a fraud analyst tool grounded in your policy corpus — without retraining a model.

## Tuning Gemini

When prompting and RAG are not enough — you need the model to adopt a specific style, format, or domain behavior consistently — you can **tune** Gemini. Vertex AI supports **supervised fine-tuning (SFT)** of Gemini models: you provide a dataset of input/output examples (typically as JSONL in Cloud Storage), launch a managed tuning job, and get a tuned model version you call like any other. You launch it through the same google-genai client:

```python
tuning_job = client.tunings.tune(
    base_model="gemini-2.5-flash",
    training_dataset=types.TuningDataset(
        gcs_uri="gs://myco-fraud-data/tuning/disputes.jsonl"),
    config=types.CreateTuningJobConfig(
        tuned_model_display_name="fraud-dispute-summarizer",
        epoch_count=3,
    ),
)
# When done, call the tuned endpoint by its resource name:
resp = client.models.generate_content(model=tuning_job.tuned_model.endpoint, contents="...")
```

SFT is the right tool when few-shot prompting cannot reliably produce the behavior you need and you have a few hundred to a few thousand high-quality labeled examples. It is parameter-efficient and managed — no GPU wrangling — and the tuned model integrates with the same endpoints and monitoring as the base model. Reserve it for when cheaper options (better prompts, grounding, RAG) have been exhausted, because a good RAG setup often beats fine-tuning for knowledge-injection tasks.

## Building generative applications and agents

Beyond single calls, Vertex AI supports **function calling** (the model requests that your code run a tool and feeds the result back), structured JSON output, and agent frameworks for multi-step, tool-using workflows. A production generative feature typically combines several: a Flash model for latency, grounding or RAG for accuracy, function calling to take actions, and structured output so downstream code can consume the result reliably. Four production concerns round this out:

**Safety settings.** Every request can carry `safety_settings` that set the block threshold per harm category. Tune these to your domain — a fraud analyst tool discussing financial crime needs different thresholds than a consumer chatbot — but never disable them blindly:

```python
config=types.GenerateContentConfig(
    safety_settings=[types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH")],
)
```

**Context caching.** When many requests share a large common prefix — a long system prompt, a policy document, a fixed set of examples — cache that prefix once with `client.caches.create(...)` and reference the cache handle on each call. You are billed a reduced rate for cached tokens instead of re-sending (and re-paying for) the same context on every request, which is a large saving for RAG and document-Q&A workloads.

**Batch prediction for Gemini.** For non-interactive bulk work — summarizing a day's disputes, classifying a backlog, embedding a corpus — submit a **batch job** (`client.batches.create(...)`) reading JSONL from Cloud Storage or a BigQuery table and writing results back. It runs asynchronously at a lower per-token price than online calls and does not count against your online QPM quota.

**Provisioned Throughput.** Pay-per-token (the default, "on-demand") is subject to shared-capacity quotas and can be rate-limited under load. For latency- and availability-critical production traffic you can purchase **Provisioned Throughput** — reserved, guaranteed generation capacity billed at a fixed rate — so a spike in your fraud pipeline is not throttled behind other tenants.

A recurring **gotcha** across all of these: on-demand generative calls are governed by **queries-per-minute (QPM) and tokens-per-minute quotas** that vary by model and region. A feature that works in testing can hit `429 RESOURCE_EXHAUSTED` under production load — plan for it with retries and backoff, request a quota increase, or move critical paths to Provisioned Throughput.

## How this fits the whole solution

Generative AI is a serving surface alongside your custom models. In the end-to-end system, a Gemini call — grounded in RAG over your BigQuery/Cloud Storage data, secured with IAM and Secret Manager, deployed behind Cloud Run or called directly, and observed through Cloud Monitoring — is often the user-facing layer, while classic ML models handle scoring and ranking underneath. Because it is a managed API, it slots into the same pipelines, cost controls, and monitoring as everything else. Many modern systems are hybrids: BigQuery ML and Vertex custom models for structured prediction, Gemini for language understanding and generation, all composed into one product.

## Key takeaways

- Default to **Gemini Flash** for most tasks, **Pro** for deep reasoning / largest context, **Flash-Lite** for high-volume low-latency; 2.5 models are GA and the 3.x generation is the newest. Pin explicit model IDs and watch retirement dates.
- **Use the `google-genai` SDK** with the `genai.Client(vertexai=True, ...)` pattern — the old `vertexai.generative_models` modules are removed as of June 24, 2026. The non-generative `aiplatform` SDK is unaffected.
- **Model Garden** unifies first-party, partner (Claude, Llama), and open models behind one Vertex interface; **grounding with Google Search** and the **RAG Engine / Vertex AI Search** fix hallucination and knowledge-cutoff by injecting current or private data.
- **Supervised fine-tuning** of Gemini is the managed option for consistent style/format/domain behavior — reach for it only after prompting, grounding, and RAG fall short.

## CLI cheat-sheet

```bash
# --- Model Garden: discover and deploy open models ---
gcloud ai model-garden models list --model-filter=gemma
gcloud ai model-garden models list --can-deploy-hugging-face-models
gcloud ai model-garden models deploy --model=google/gemma3@gemma-3-9b \
  --region=us-central1 --accept-eula
gcloud ai model-garden models deploy --model=meta-llama/Meta-Llama-3-8B \
  --hugging-face-access-token=$HF_TOKEN --region=us-central1 --accept-eula

# --- Raw generateContent REST call (auth via short-lived token, no API key on Vertex) ---
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJ/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent" \
  -d '{"contents": {"role": "user", "parts": {"text": "Hello"}}}'

# --- Everything else is the google-genai SDK (pip install google-genai) ---
#   client = genai.Client(vertexai=True, project=..., location="us-central1")
#   client.models.generate_content(model="gemini-2.5-flash", contents=..., config=...)
#   client.models.embed_content(model="text-embedding-005", contents=[...])
#   client.caches.create(...)          # context caching
#   client.batches.create(...)         # batch generation / embeddings
#   client.tunings.tune(base_model="gemini-2.5-flash", training_dataset=...)   # SFT

# Region prefix on the host must match the location in the path.
# Watch model retirement dates, QPM/TPM quotas, and per-region model availability.
```

## Try it

Build a grounded generative feature end to end:

1. `pip install google-genai`, then call `gemini-2.5-flash` via `genai.Client(vertexai=True, ...)` and print a response.
2. Add **grounding with Google Search** as a tool and compare the answer (and its citations) on a question about recent events.
3. Stand up a small **RAG** flow: put a few of your own documents in Cloud Storage, ingest them with the Vertex AI RAG Engine (or a simple embed-and-retrieve loop), and have the model answer strictly from that corpus.
4. Wrap the whole thing in a **Cloud Run** service with the API key in **Secret Manager** and authentication required, then hit it — you now have a deployed, grounded, secured generative endpoint that fits the rest of your system.
