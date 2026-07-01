# 11 — Azure AI Foundry, OpenAI, and AI Services

Not every ML problem is worth training a model for. When the task is language, vision, speech, or reasoning over documents, Azure offers **pre-built and foundation models** you deploy and call rather than train from scratch. The umbrella for this is **Microsoft Foundry** (the platform recently rebranded from Azure AI Foundry) — the place to discover, deploy, ground, evaluate, and serve foundation models, including the **Azure OpenAI** family, alongside a broader **AI Services** suite for vision, speech, language, and document intelligence. In the end-to-end solution, this is the GenAI branch: the alternative (or complement) to the custom-training path, wired into the same identity, networking, data, and monitoring backbone.

## Microsoft Foundry: the GenAI platform

**Microsoft Foundry** is the unified platform for building AI applications and agents. Its structure:

- A **Foundry resource** (the Azure resource, formerly the Azure OpenAI / Azure AI Services resource) and, within it, **projects** that organize your work, connections, and deployments.
- A **model catalog** — thousands of models from Microsoft, OpenAI, Meta, Mistral, DeepSeek, xAI, and others. Some are **sold directly by Azure** (billed and supported by Microsoft, including the Azure OpenAI models); others are open-weight models you deploy to your own managed compute.
- **Deployments** — you deploy a catalog model to get a callable endpoint. This is the GenAI analog of an Azure ML online endpoint, but for foundation models.

The current models include the **GPT-5.x** generation of OpenAI models (earlier GPT-4o versions were retired and auto-upgraded), the **o-series reasoning models** for hard multi-step problems, **embedding models** (`text-embedding-3-large` and successors) for vectorization, plus vision, image-generation, and audio models. Because Microsoft renames and retires models on a regular cadence, always confirm the exact current model name and its regional availability in the catalog before hard-coding a deployment name — treat the model catalog and the model-retirements schedule as the source of truth.

## Deployment types: matching capacity to workload

When you deploy a model, you choose a **deployment type** that governs capacity and billing:

- **Standard (pay-as-you-go)** — billed per token, shared capacity, no commitment. The default for development and variable traffic. **Global Standard** routes to available capacity worldwide for the best throughput; **regional/data-zone** variants keep processing within a geography for data-residency needs.
- **Provisioned (PTU — Provisioned Throughput Units)** — reserved, dedicated capacity with predictable latency and throughput, billed by reservation. For production workloads with steady, high volume and strict latency SLAs.
- **Batch** — asynchronous, high-throughput processing at a large discount, for non-latency-sensitive bulk jobs (mass summarization, offline classification).

```bash
# Deploy a chat model on Global Standard capacity
az cognitiveservices account deployment create \
  --name foundry-mlx --resource-group rg-mlx-dev \
  --deployment-name gpt-5.1-chat \
  --model-name gpt-5.1 --model-version "latest" --model-format OpenAI \
  --sku-name GlobalStandard --sku-capacity 50
```

## Calling models: the SDKs

The **Azure AI Projects** SDK (`azure-ai-projects`) is the unified entry point — you connect to a single project endpoint and get access to model inference, agents, evaluation, and connected resources. It exposes an **OpenAI-compatible client**, so code written against the OpenAI API works with a change of base URL and Entra authentication:

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project = AIProjectClient(
    endpoint="https://foundry-mlx.services.ai.azure.com/api/projects/mlx",
    credential=DefaultAzureCredential(),      # managed identity, no API keys
)

client = project.get_openai_client(api_version="2025-01-01-preview")
resp = client.chat.completions.create(
    model="gpt-5.1-chat",                     # your deployment name
    messages=[{"role": "user", "content": "Summarize this transaction as fraud risk."}],
)
print(resp.choices[0].message.content)
```

Authenticate with **managed identity** and grant callers the `Cognitive Services OpenAI User` role — the same keyless pattern as everywhere else. Avoid API keys; they are a secret to leak.

## Grounding models: RAG with Azure AI Search

Foundation models do not know your private data. **Retrieval-Augmented Generation (RAG)** grounds them by retrieving relevant snippets from your corpus and injecting them into the prompt. The retrieval layer on Azure is **Azure AI Search** — a managed search service supporting **vector, keyword, and hybrid (semantic) search**, with native indexers that ingest from Blob Storage, Data Lake, Cosmos DB, SharePoint, and more.

The classic RAG loop: chunk your documents, embed each chunk with an embedding model, index the vectors in Azure AI Search; at query time, embed the user's question, retrieve the top-k nearest chunks, and pass them as context to the chat model.

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

search = SearchClient("https://srch-mlx.search.windows.net", "docs",
                      DefaultAzureCredential())

q_vec = client.embeddings.create(model="text-embedding-3-large",
                                 input="What is our refund policy?").data[0].embedding
hits = search.search(vector_queries=[VectorizedQuery(
    vector=q_vec, k_nearest_neighbors=5, fields="contentVector")])
context = "\n\n".join(h["content"] for h in hits)

answer = client.chat.completions.create(
    model="gpt-5.1-chat",
    messages=[
        {"role": "system", "content": f"Answer only from this context:\n{context}"},
        {"role": "user", "content": "What is our refund policy?"},
    ],
)
```

Foundry increasingly abstracts this: **Foundry IQ** provides a managed knowledge layer that unifies retrieval across sources (Azure AI Search, Fabric, Azure SQL, file search, and MCP-connected tools) behind a single retrieval endpoint, so agents ground themselves without you hand-building the pipeline. For sensitive data, private connectivity keeps the AI Search ↔ Foundry traffic inside your network boundary — the same private-endpoint discipline from the networking section.

## Agents and the broader AI Services

Beyond chat, Foundry provides an **Agent Service** for building tool-using agents — models that call functions, retrieve knowledge, and take multi-step actions, with the orchestration, threading, and tool invocation managed for you. This is how you go from a single completion to an assistant that queries your database, calls your ML endpoint, and grounds its answers.

The wider **AI Services** suite gives task-specific pre-built models you call as APIs, no training required:

- **Document Intelligence** — extract structured data (tables, key-value pairs, entities) from PDFs, invoices, and forms; the workhorse for document-heavy pipelines.
- **AI Vision** — image analysis, OCR, object detection.
- **AI Speech** — speech-to-text, text-to-speech, translation.
- **AI Language** — entity recognition, sentiment, summarization, PII detection, custom classification.
- **Content Safety** — moderation and guardrails for both inputs and model outputs.

These compose with the foundation models: Document Intelligence extracts fields from a scanned form, an embedding model vectorizes the text, AI Search indexes it, and a chat model answers questions over it — a full document-AI pipeline from managed pieces.

## Build vs. buy: choosing this path

Reach for Foundry/AI Services when a foundation or pre-built model already solves the task well (language, summarization, extraction, chat, vision, speech), when time-to-value matters more than a custom model, or when you lack the labeled data to train one. Reach for **custom training** (the Azure ML path) when you have proprietary data and a task where a purpose-built model beats a general one, when you need a small/cheap model at high volume, or when you must own the weights. Most mature systems do **both** — a fine-tuned custom model for the core prediction, plus foundation models for the language and document surfaces around it. Foundry also supports **fine-tuning** the hosted models, bridging the two paths.

## How the GenAI layer fits the whole solution

In the reference architecture, the GenAI branch mirrors the custom-ML branch and shares its backbone. Documents land in the **lake**; **Document Intelligence** and embedding models turn them into vectors indexed in **Azure AI Search**; a **Foundry deployment** (or Foundry IQ knowledge layer) serves grounded answers through the same thin **HTTP Function / API gateway** that fronts the custom endpoints. Calls authenticate with **managed identity** (`Cognitive Services OpenAI User`), traffic stays on **private endpoints**, prompts and responses are logged to **Azure Monitor**, and token spend is tracked in **Cost Management**. **Content Safety** guards the boundary. The result: whether a request needs a custom fraud score or a natural-language answer over policy documents, it flows through one governed, observable, secure platform.

## Key takeaways

- **Microsoft Foundry** (formerly Azure AI Foundry) is the platform to deploy and call foundation models — including **Azure OpenAI** (GPT-5.x, o-series, embeddings) — via a **model catalog** and **deployments**; always verify current model names and regions.
- Pick a **deployment type** by workload: **Standard/Global Standard** (pay-per-token, variable), **Provisioned (PTU)** (reserved, low-latency, high-volume), **Batch** (bulk, discounted).
- Call models with the **Azure AI Projects SDK** (OpenAI-compatible client) using **managed identity** and the `Cognitive Services OpenAI User` role — no API keys.
- Ground models with **RAG over Azure AI Search** (vector + hybrid), or let **Foundry IQ** manage retrieval; use the **Agent Service** for tool-using agents.
- The broader **AI Services** (Document Intelligence, Vision, Speech, Language, Content Safety) are call-an-API models that compose into document-AI and multimodal pipelines — choose them over custom training when they already solve the task.

## Try it

Create a Foundry resource and deploy both a chat model and an embedding model. Using the Azure AI Projects SDK with `DefaultAzureCredential`, send a chat completion and confirm it works with zero API keys. Then build a minimal RAG loop: create an Azure AI Search index, embed and index three short documents, and at query time retrieve the top matches and pass them as context to the chat model — observing how grounding changes the answer. Finally, decide for one task in your domain whether you would use a foundation model here or train a custom model on the Azure ML path, and justify the choice.
