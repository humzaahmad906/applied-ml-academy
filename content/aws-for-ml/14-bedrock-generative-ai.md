# 14 — Bedrock: Generative AI and Foundation Models

Everything before this module served *your* models — models you trained, registered, and deployed. Generative AI flips that: the model is a managed service you call, not an artifact you host. **Amazon Bedrock** is AWS's fully managed way to use foundation models (FMs) — large language models, embedding models, and image models — from Anthropic, Amazon, Meta, Mistral, Cohere, AI21, and others behind a single API, with no GPUs to provision and no weights to manage. It is the AWS analog of Azure OpenAI Service and Google Vertex AI's Gemini. This module covers how to call models, the RAG and safety and agent layers built on top, and the CLI you use to drive all of it.

## What Bedrock is (and what it is not)

Bedrock is serverless model *access*: you send a prompt over HTTPS and pay per input and output token, with nothing running when you are idle. It is split into two API surfaces that map to two CLI namespaces. The **control plane** (`aws bedrock`) manages models, guardrails, logging, and inference profiles — the setup. The **runtime plane** (`aws bedrock-runtime`) is where you actually invoke a model for a completion. Keep the two straight: you `list-foundation-models` on the control plane but `converse` on the runtime plane.

Before you can call a model you must **request access** to it in the console (Model access page) — most models are opt-in per account and per Region, and a call to a model you have not enabled fails with `AccessDeniedException`. Model availability also varies by Region exactly like GPU capacity does, so confirm your target model exists where you are working.

```bash
# Control plane: what models can I use in this Region?
aws bedrock list-foundation-models --region us-east-1 --output table

# Filter to a provider (text models from Anthropic)
aws bedrock list-foundation-models \
  --by-provider anthropic --by-output-modality TEXT
```

## Invoking a model: Converse vs InvokeModel

There are two ways to call a model, and the choice matters. **`invoke-model`** sends a raw request body whose JSON schema is *specific to each provider* — Anthropic's Claude, Amazon Nova, and Meta Llama each expect a different shape — so your code is coupled to the model you picked. **`converse`** (the Converse API) is the unified interface: one request shape works across every model that supports messages, so you can swap models without rewriting your code, and it natively handles multi-turn conversations, system prompts, and tool use. **Prefer Converse** for almost everything; reach for `invoke-model` only for a provider-specific feature Converse does not expose yet, or for embeddings and image models.

```bash
# Converse — the unified, model-agnostic call (recommended)
aws bedrock-runtime converse \
  --model-id amazon.nova-lite-v1:0 \
  --messages '[{"role":"user","content":[{"text":"Summarize what a feature store solves in one sentence."}]}]' \
  --inference-config '{"maxTokens":512,"temperature":0.2,"topP":0.9}'
```

`invoke-model` needs two things people forget: `--cli-binary-format raw-in-base64-out` (so the CLI treats `--body` as raw JSON, not base64), and a trailing output file for the response bytes:

```bash
# InvokeModel — provider-specific body shape, response written to a file
aws bedrock-runtime invoke-model \
  --model-id amazon.nova-lite-v1:0 \
  --body '{"messages":[{"role":"user","content":[{"text":"Say hello in one line."}]}],"inferenceConfig":{"maxTokens":128,"temperature":0.5}}' \
  --cli-binary-format raw-in-base64-out \
  out.json
```

For interactive experiences use **`converse-stream`**, which returns tokens incrementally so a UI can render as the model writes — the same streaming you expect from any chat product. In boto3 the calls are `client("bedrock-runtime").converse(...)` and `.converse_stream(...)`.

## Model IDs and inference profiles

A **model ID** like `anthropic.claude-3-5-sonnet-20241022-v2:0` or `amazon.nova-pro-v1:0` names a specific model version in a single Region. The complication: many current models are only callable through a **cross-region inference profile**, an ID prefixed with a geography (for example `us.anthropic.claude-3-5-sonnet-...`) that lets Bedrock route your request to whichever Region in that geography has capacity, improving throughput and availability. If a direct model ID returns an error saying on-demand throughput is not supported, switch to the corresponding inference-profile ID. You can also create **application inference profiles** to tag and track cost and usage per team or workload — the generative-AI equivalent of cost-allocation tags on an endpoint.

```bash
# System-defined cross-region profiles you can invoke
aws bedrock list-inference-profiles --type-equals SYSTEM_DEFINED

# Create an application inference profile to attribute cost to a team
aws bedrock create-inference-profile \
  --inference-profile-name team-fraud \
  --model-source '{"copyFrom":"arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"}' \
  --tags Key=team,Value=fraud
```

## Knowledge Bases: managed RAG

A raw FM only knows what was in its training data. **Retrieval-Augmented Generation (RAG)** grounds it in *your* documents: you retrieve the relevant passages and pass them to the model as context. **Bedrock Knowledge Bases** is the managed version of this whole pipeline — you point it at documents in S3, it chunks them, embeds them with an embedding model, stores the vectors (in OpenSearch Serverless, Aurora pgvector, S3 Vectors, and others), and exposes two runtime calls: `retrieve` (get the relevant chunks) and `retrieve-and-generate` (retrieve *and* have an FM answer using them, with citations). You avoid operating an embedding pipeline and a vector database yourself.

```bash
# Ask a grounded question against a knowledge base, model answers with citations
aws bedrock-agent-runtime retrieve-and-generate \
  --input '{"text":"What is our refund policy for enterprise plans?"}' \
  --retrieve-and-generate-configuration '{
    "type":"KNOWLEDGE_BASE",
    "knowledgeBaseConfiguration":{
      "knowledgeBaseId":"KB123ABC",
      "modelArn":"arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"}}'
```

The control plane for building these lives under `aws bedrock-agent` (`create-knowledge-base`, `create-data-source`, `start-ingestion-job`), and the query-time calls live under `aws bedrock-agent-runtime`. Chunking strategy and the embedding model are the two levers that most affect retrieval quality.

## Guardrails: safety as a policy you attach

**Bedrock Guardrails** apply configurable safety policies to prompts and responses independently of which model you use: content filters (hate, violence, sexual, misconduct, prompt-attack), denied topics, word filters, sensitive-information redaction (PII), and **contextual grounding** checks that flag hallucinated or off-source answers. You define a guardrail once and reference it by ID and version on any `converse`/`invoke-model` call — so the same safety policy protects every model behind your application.

```bash
# Create a guardrail that blocks a topic and sets the refusal messages
aws bedrock create-guardrail \
  --name no-medical-advice \
  --topic-policy-config '{"topicsConfig":[{"name":"MedicalAdvice","definition":"Requests for personalized medical diagnosis or treatment.","type":"DENY"}]}' \
  --content-policy-config '{"filtersConfig":[{"type":"HATE","inputStrength":"HIGH","outputStrength":"HIGH"}]}' \
  --blocked-input-messaging "I cannot help with that request." \
  --blocked-outputs-messaging "I cannot provide that response."

# Apply it on a runtime call by ID + version
aws bedrock-runtime converse \
  --model-id amazon.nova-lite-v1:0 \
  --messages '[{"role":"user","content":[{"text":"..."}]}]' \
  --guardrail-config '{"guardrailIdentifier":"gr-abc123","guardrailVersion":"1"}'
```

## Agents and AgentCore

Beyond single completions, Bedrock supports **agents** that plan multi-step tasks and call tools (your APIs, Lambda functions, or a knowledge base) to accomplish a goal. The original **Bedrock Agents** (`aws bedrock-agent`) let you define an agent, its action groups, and its attached knowledge bases declaratively. Since October 2025 there is also **Amazon Bedrock AgentCore** (`aws bedrock-agentcore` / `bedrock-agentcore-control`), a serverless, purpose-built runtime for hosting agents built with *any* framework (Strands, LangGraph, CrewAI, and others) and any model — with sessions up to eight hours, plus managed memory, gateway, identity, and observability. Use Bedrock Agents for the fully-managed AWS-native path; use AgentCore when you bring your own agent framework and want production-grade hosting for it.

## Cost, throughput, and logging

Bedrock bills per token by default (**on-demand**), which is ideal for spiky or exploratory use. For steady high volume, **Provisioned Throughput** reserves dedicated model capacity (model units) at a lower per-token effective rate — the generative analog of Savings Plans for inference. **Batch inference** processes a large set of prompts from S3 asynchronously at roughly half the on-demand price, for offline jobs like bulk classification or dataset labeling. Because token spend is invisible until the bill arrives, enable **model invocation logging** early — it captures every request and response to CloudWatch Logs and/or S3, which is how you audit usage, debug prompts, and attribute cost.

```bash
# Turn on model invocation logging to CloudWatch + S3
aws bedrock put-model-invocation-logging-configuration \
  --logging-config '{
    "cloudWatchConfig":{"logGroupName":"/bedrock/invocations","roleArn":"arn:aws:iam::<acct>:role/bedrock-logging"},
    "s3Config":{"bucketName":"my-ml-data","keyPrefix":"bedrock-logs/"},
    "textDataDeliveryEnabled":true}'
```

## How this fits the whole ML solution

Bedrock is the second model backend of the end-to-end system. In the reference architecture (module 12), the same API Gateway and Lambda front door that calls a SageMaker endpoint for a classical prediction instead calls Bedrock's Converse API for a generative response — grounded by a Knowledge Base, filtered by a Guardrail, and logged to the same CloudWatch that watches your endpoints. It sits inside your VPC via PrivateLink, uses IAM roles for access exactly like every other service, and its token spend shows up in the same Cost Explorer. Generative features are not a separate system; they are one more backend behind the front door you already built.

## Key takeaways

- Bedrock is serverless, pay-per-token access to foundation models from many providers; request model access per Region before calling.
- Two planes: `aws bedrock` (control — models, guardrails, logging, profiles) and `aws bedrock-runtime` (invoke — converse, invoke-model, converse-stream).
- Prefer the **Converse API** for a model-agnostic, multi-turn interface; use `invoke-model` only for provider-specific bodies, embeddings, or image models.
- Many current models require a **cross-region inference profile** ID; use application inference profiles to attribute cost.
- Knowledge Bases give managed RAG (`retrieve-and-generate` with citations); Guardrails attach reusable safety policies; Bedrock Agents and AgentCore handle multi-step tool-using agents.
- Control cost with Provisioned Throughput (steady volume) and Batch inference (offline), and turn on model invocation logging to audit and attribute token spend.

## CLI cheat-sheet

```bash
# --- Discover models (control plane) ---
aws bedrock list-foundation-models --by-provider anthropic --by-output-modality TEXT
aws bedrock get-foundation-model --model-identifier amazon.nova-pro-v1:0
aws bedrock list-inference-profiles --type-equals SYSTEM_DEFINED

# --- Invoke (runtime plane) ---
# Unified, model-agnostic (recommended)
aws bedrock-runtime converse \
  --model-id amazon.nova-lite-v1:0 \
  --messages '[{"role":"user","content":[{"text":"Hello"}]}]' \
  --inference-config '{"maxTokens":512,"temperature":0.2}'
# Streaming for interactive UIs
aws bedrock-runtime converse-stream --model-id amazon.nova-lite-v1:0 \
  --messages '[{"role":"user","content":[{"text":"Hello"}]}]'
# Provider-specific raw body (note the two required extras)
aws bedrock-runtime invoke-model --model-id amazon.nova-lite-v1:0 \
  --body '{"messages":[{"role":"user","content":[{"text":"Hi"}]}]}' \
  --cli-binary-format raw-in-base64-out out.json

# --- Guardrails ---
aws bedrock create-guardrail --name my-guard \
  --blocked-input-messaging "Blocked." --blocked-outputs-messaging "Blocked." \
  --content-policy-config '{"filtersConfig":[{"type":"HATE","inputStrength":"HIGH","outputStrength":"HIGH"}]}'
aws bedrock list-guardrails
aws bedrock create-guardrail-version --guardrail-identifier gr-abc123

# --- Knowledge Bases (RAG) ---
aws bedrock-agent create-knowledge-base --name docs-kb --role-arn <arn> \
  --knowledge-base-configuration <json> --storage-configuration <json>
aws bedrock-agent create-data-source --knowledge-base-id KB123 --name s3-docs --data-source-configuration <json>
aws bedrock-agent start-ingestion-job --knowledge-base-id KB123 --data-source-id DS123
aws bedrock-agent-runtime retrieve --knowledge-base-id KB123 \
  --retrieval-query '{"text":"question"}'
aws bedrock-agent-runtime retrieve-and-generate --input '{"text":"question"}' \
  --retrieve-and-generate-configuration <json>

# --- Agents ---
aws bedrock-agent create-agent --agent-name planner --foundation-model amazon.nova-pro-v1:0 --agent-resource-role-arn <arn>
aws bedrock-agent-runtime invoke-agent --agent-id A123 --agent-alias-id AL1 --session-id s1 --input-text "..."

# --- Cost & logging ---
aws bedrock put-model-invocation-logging-configuration --logging-config <json>
aws bedrock get-model-invocation-logging-configuration
aws bedrock create-inference-profile --inference-profile-name team-x --model-source <json> --tags Key=team,Value=x
```

## Try it

Request access to two models in the console — one Amazon Nova and one Anthropic Claude. Call each with `aws bedrock-runtime converse` using the *identical* `--messages` payload and confirm the same code drives both (that is the point of Converse). Then create a Guardrail that denies a topic, re-run the call with `--guardrail-config`, and confirm the blocked-input message comes back. Finally, build a small Knowledge Base over a folder of S3 documents, run `retrieve-and-generate`, and verify the answer cites your sources — you have now stood up grounded, guarded generative inference entirely from the CLI.
