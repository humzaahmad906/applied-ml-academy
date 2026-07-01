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

## Model Garden

**Model Garden** is the catalog of models available on Vertex AI. It includes Google's first-party models (Gemini, embeddings, image and video generation), a curated set of **partner models** (such as Anthropic's Claude and Meta's Llama, callable through the same Vertex surface), and hundreds of **open models** you can deploy to your own endpoints. It is where you discover a model, read its card, and either call it as a managed API or deploy it to an endpoint you control. For an ML engineer, Model Garden turns "which model should I use?" into a browse-and-try exercise rather than a procurement project, and lets you keep every model — first-party, partner, and open — behind one consistent Vertex AI interface with unified auth, logging, and billing.

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

When prompting and RAG are not enough — you need the model to adopt a specific style, format, or domain behavior consistently — you can **tune** Gemini. Vertex AI supports **supervised fine-tuning (SFT)** of Gemini models: you provide a dataset of input/output examples (typically as JSONL in Cloud Storage), launch a managed tuning job, and get a tuned model version you call like any other. SFT is the right tool when few-shot prompting cannot reliably produce the behavior you need and you have a few hundred to a few thousand high-quality labeled examples. It is parameter-efficient and managed — no GPU wrangling — and the tuned model integrates with the same endpoints and monitoring as the base model. Reserve it for when cheaper options (better prompts, grounding, RAG) have been exhausted, because a good RAG setup often beats fine-tuning for knowledge-injection tasks.

## Building generative applications and agents

Beyond single calls, Vertex AI supports **function calling** (the model requests that your code run a tool and feeds the result back), structured JSON output, context caching for cost savings on repeated large prompts, and agent frameworks for multi-step, tool-using workflows. A production generative feature typically combines several: a Flash model for latency, grounding or RAG for accuracy, function calling to take actions, and structured output so downstream code can consume the result reliably.

## How this fits the whole solution

Generative AI is a serving surface alongside your custom models. In the end-to-end system, a Gemini call — grounded in RAG over your BigQuery/Cloud Storage data, secured with IAM and Secret Manager, deployed behind Cloud Run or called directly, and observed through Cloud Monitoring — is often the user-facing layer, while classic ML models handle scoring and ranking underneath. Because it is a managed API, it slots into the same pipelines, cost controls, and monitoring as everything else. Many modern systems are hybrids: BigQuery ML and Vertex custom models for structured prediction, Gemini for language understanding and generation, all composed into one product.

## Key takeaways

- Default to **Gemini Flash** for most tasks, **Pro** for deep reasoning / largest context, **Flash-Lite** for high-volume low-latency; 2.5 models are GA and the 3.x generation is the newest. Pin explicit model IDs and watch retirement dates.
- **Use the `google-genai` SDK** with the `genai.Client(vertexai=True, ...)` pattern — the old `vertexai.generative_models` modules are removed as of June 24, 2026. The non-generative `aiplatform` SDK is unaffected.
- **Model Garden** unifies first-party, partner (Claude, Llama), and open models behind one Vertex interface; **grounding with Google Search** and the **RAG Engine / Vertex AI Search** fix hallucination and knowledge-cutoff by injecting current or private data.
- **Supervised fine-tuning** of Gemini is the managed option for consistent style/format/domain behavior — reach for it only after prompting, grounding, and RAG fall short.

## Try it

Build a grounded generative feature end to end:

1. `pip install google-genai`, then call `gemini-2.5-flash` via `genai.Client(vertexai=True, ...)` and print a response.
2. Add **grounding with Google Search** as a tool and compare the answer (and its citations) on a question about recent events.
3. Stand up a small **RAG** flow: put a few of your own documents in Cloud Storage, ingest them with the Vertex AI RAG Engine (or a simple embed-and-retrieve loop), and have the model answer strictly from that corpus.
4. Wrap the whole thing in a **Cloud Run** service with the API key in **Secret Manager** and authentication required, then hit it — you now have a deployed, grounded, secured generative endpoint that fits the rest of your system.
