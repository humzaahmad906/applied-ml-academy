# 07 — Serverless: Lambda and API Gateway

Serverless compute runs your code only when something triggers it, scales automatically, and bills to the millisecond with nothing to manage in between. For ML systems, Lambda and API Gateway are the glue and the front door: they orchestrate pipelines, react to data landing in S3, wrap models behind clean HTTP APIs, and run lightweight CPU inference cheaply. This module covers what Lambda can and cannot do, how API Gateway fronts it, the event-driven patterns that show up constantly in ML, and the hard limits you must design around.

## AWS Lambda essentials

A **Lambda function** is a piece of code plus a trigger. You supply a handler; AWS provisions the runtime, executes on each invocation, and scales out by running many concurrent copies. You pay per request and per GB-second of execution — zero cost when idle. Lambda supports managed runtimes (Python, Node, Java, Go, and more) and, importantly for ML, **container images up to 10 GB**, which is how you package models and dependencies too large for a zip.

The limits that shape ML use:

- **Timeout: 15 minutes maximum.** Fine for inference and glue, wrong for training.
- **Memory: up to 10 GB**, and CPU scales with memory — more memory means more vCPU.
- **/tmp scratch space:** configurable up to 10 GB for downloading model weights at runtime.
- **No GPU.** Lambda has no GPU resource type or CUDA exposure, so it is CPU inference only.
- **Cold starts:** the first invocation after idle pays initialization cost, which for a large ML container can be seconds.

```python
import json, boto3, os
# Load the model once, outside the handler, so warm invocations reuse it
model = load_model(os.environ["MODEL_PATH"])

def handler(event, context):
    features = json.loads(event["body"])["features"]
    pred = model.predict(features)
    return {"statusCode": 200, "body": json.dumps({"prediction": pred})}
```

Loading the model at module scope (not inside the handler) is the key optimization — it runs once per cold start and is reused across warm invocations.

## Taming cold starts

Two features fight cold starts. **Provisioned concurrency** keeps a set number of instances initialized and warm, giving consistent low latency for user-facing inference at the cost of paying for the reserved capacity. **SnapStart** takes a snapshot of an initialized function and restores from it quickly — but note it does **not** support container-image functions, which limits it for heavy Python ML images (those rely on provisioned concurrency instead). **Lambda layers** let you share dependencies across functions to keep individual deployment packages smaller.

## API Gateway: the HTTP front door

**Amazon API Gateway** turns a Lambda function (or other backend) into a managed HTTP API with authentication, throttling, and request validation. Two flavors: **HTTP APIs** are lower-latency, lower-cost, and the right default for most services; **REST APIs** add features like request/response transformation, API keys, and usage plans when you need them. API Gateway handles rate limiting (throttling) so a traffic spike does not overwhelm your backend, and it integrates with IAM, Cognito, or custom authorizers for auth.

For the simplest case, a **Lambda function URL** gives a function its own HTTPS endpoint with no API Gateway at all — good for internal or single-consumer inference, though you lose the richer controls.

## Event-driven ML patterns

The real power of serverless in ML is reacting to events:

- **S3 event → Lambda:** a new file lands in a bucket and triggers preprocessing, validation, or a trigger into a training pipeline. This is the backbone of ingestion.
- **EventBridge → Lambda:** schedule retraining nightly, or fan out on a custom event ("new model registered").
- **SQS → Lambda:** buffer bursty inference or processing requests in a queue and let Lambda drain them at a controlled rate, with retries and dead-letter queues for failures.

```bash
# Trigger a preprocessing Lambda whenever raw data lands in S3
aws s3api put-bucket-notification-configuration \
  --bucket my-ml-data \
  --notification-configuration '{
    "LambdaFunctionConfigurations": [{
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:<acct>:function:preprocess",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": "raw/"}]}}
    }]
  }'
```

## When Lambda is right — and when it is not

Lambda is ideal for lightweight CPU inference (small models, classical ML, embeddings from a small model), for orchestration and glue between services, for event reactions, and for pre/post-processing around a heavier endpoint. Lambda is the *wrong* tool when you need a GPU, when inference exceeds ~15 minutes or the payload is large, when you need consistently ultra-low latency without paying for provisioned concurrency, or when the model is too big to initialize quickly. Those cases belong on SageMaker endpoints, ECS/EKS, or asynchronous inference. A common hybrid: API Gateway → Lambda for auth, validation, and light routing, with the Lambda calling a SageMaker endpoint for the actual heavy prediction.

## How this fits the whole ML solution

Lambda is the nervous system of the architecture. It is what fires when data arrives, what a scheduler invokes to kick off retraining, what wraps a model behind a clean API, and what stitches Step Functions stages together. It rarely does the heavy ML lifting itself, but it connects every other component — ingestion to training, training to registry, endpoint to consumer — while costing nothing when nothing is happening. In the end-to-end system it is the low-cost connective glue that makes the pieces event-driven rather than manually run.

## Key takeaways

- Lambda runs code on triggers, scales automatically, bills per request/GB-second, and supports container images up to 10 GB.
- Hard limits: 15-minute timeout, 10 GB memory, 10 GB /tmp, and **no GPU** — CPU inference and glue only.
- Fight cold starts with provisioned concurrency; SnapStart helps managed runtimes but not container images; use layers to slim packages.
- API Gateway (HTTP APIs by default) adds auth, throttling, and validation; function URLs are the no-frills alternative.
- Event-driven patterns (S3 → Lambda, EventBridge schedules, SQS buffering) make the ML pipeline reactive; offload heavy inference to endpoints.

## Try it

Build a Lambda function packaged as a container image that loads a small model from S3 into `/tmp` at cold start and serves predictions. Front it with an HTTP API in API Gateway and call it with `curl`. Measure cold-start versus warm latency, then enable provisioned concurrency and measure again. Finally, add an S3 event trigger so that uploading a file to a `raw/` prefix invokes a second Lambda that validates the file — confirming the event-driven ingestion pattern end to end.
