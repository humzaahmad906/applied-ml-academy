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

There are two ways to package a function, and the choice is mostly about size. A **zip** deployment (plus optional layers) is capped at **250 MB unzipped**, which classical ML with numpy/scikit fits into but a CUDA-less PyTorch build often does not. A **container image** goes up to **10 GB**, which is why nearly all deep-learning Lambdas ship as images built on the AWS-provided base. Creating each looks like:

```bash
# Zip package
aws lambda create-function --function-name predict \
  --runtime python3.12 --handler app.handler --timeout 30 --memory-size 1024 \
  --role arn:aws:iam::<acct>:role/lambda-exec \
  --zip-file fileb://function.zip

# Container image (deep-learning models)
aws lambda create-function --function-name predict-img \
  --package-type Image \
  --code ImageUri=<acct>.dkr.ecr.us-east-1.amazonaws.com/lambda-model:v1 \
  --role arn:aws:iam::<acct>:role/lambda-exec --timeout 60 --memory-size 3008
```

You ship new code and tune the resource knobs with two separate calls — a frequent point of confusion, since `update-function-code` changes only the artifact while `update-function-configuration` changes memory, timeout, env vars, and ephemeral storage. Memory is the single most important lever because vCPU scales linearly with it: a function stuck on CPU-bound inference often runs *cheaper* at higher memory because it finishes proportionally faster.

```bash
aws lambda update-function-code --function-name predict --zip-file fileb://function.zip
aws lambda update-function-configuration --function-name predict \
  --timeout 120 --memory-size 4096 --ephemeral-storage '{"Size": 4096}'
aws lambda get-function --function-name predict          # inspect current config + code location
aws lambda invoke --function-name predict --payload '{"features":[1,2,3]}' out.json
```

The `/tmp` ephemeral storage (512 MB by default, up to 10 GB) is where you download model weights from S3 at cold start when they are too large to bake into the image — set it explicitly with `--ephemeral-storage` or the download will silently run out of space at 512 MB.

## Taming cold starts

Two features fight cold starts. **Provisioned concurrency** keeps a set number of instances initialized and warm, giving consistent low latency for user-facing inference at the cost of paying for the reserved capacity. **SnapStart** takes a snapshot of an initialized function and restores from it quickly — but note it does **not** support container-image functions, which limits it for heavy Python ML images (those rely on provisioned concurrency instead). **Lambda layers** let you share dependencies across functions to keep individual deployment packages smaller.

It is worth being precise about the two concurrency controls because they solve opposite problems and are easy to confuse. **Provisioned concurrency** attaches to a *published version or alias* (not `$LATEST`) and pre-warms N environments so there is no cold start — you pay for them whether traffic arrives or not. **Reserved concurrency** (`put-function-concurrency`) instead *caps* how many concurrent executions a function may use, carving them out of the account's pool; it protects a downstream system (a database, a SageMaker endpoint) from being overwhelmed and stops one function from starving others. The usual pattern is to publish a version, alias it, and put provisioned concurrency on the alias:

```bash
aws lambda publish-version --function-name predict
aws lambda create-alias --function-name predict --name prod --function-version 3
aws lambda put-provisioned-concurrency-config --function-name predict \
  --qualifier prod --provisioned-concurrent-executions 5
# Cap total concurrency so a spike can't overrun a downstream endpoint
aws lambda put-function-concurrency --function-name predict --reserved-concurrent-executions 50
```

The account-wide default concurrency limit is **1,000** (raisable via a quota request), and provisioned concurrency scales with **Application Auto Scaling** if you want it to track a schedule or utilization target. The SnapStart caveat bites hardest in ML: because it excludes container images, the large PyTorch/CUDA functions that most need faster cold starts cannot use it — provisioned concurrency is the only real lever there.

## API Gateway: the HTTP front door

**Amazon API Gateway** turns a Lambda function (or other backend) into a managed HTTP API with authentication, throttling, and request validation. Two flavors: **HTTP APIs** are lower-latency, lower-cost, and the right default for most services; **REST APIs** add features like request/response transformation, API keys, and usage plans when you need them. API Gateway handles rate limiting (throttling) so a traffic spike does not overwhelm your backend, and it integrates with IAM, Cognito, or custom authorizers for auth.

Wiring an **HTTP API** to a Lambda is a four-call sequence: create the API, create an integration pointing at the function, create a route that maps a method+path to that integration, and deploy a stage. Then you grant API Gateway permission to invoke the function — a step people forget, producing a silent 500 until fixed:

```bash
aws apigatewayv2 create-api --name ml-api --protocol-type HTTP
aws apigatewayv2 create-integration --api-id <id> --integration-type AWS_PROXY \
  --integration-uri arn:aws:lambda:us-east-1:<acct>:function:predict \
  --payload-format-version 2.0
aws apigatewayv2 create-route --api-id <id> --route-key 'POST /predict' \
  --target integrations/<integration-id>
aws apigatewayv2 create-stage --api-id <id> --stage-name prod --auto-deploy
aws lambda add-permission --function-name predict --statement-id apigw \
  --action lambda:InvokeFunction --principal apigateway.amazonaws.com \
  --source-arn 'arn:aws:execute-api:us-east-1:<acct>:<id>/*/*/predict'
```

**Throttling** is what keeps a traffic spike from cascading into your model backend. HTTP APIs apply account- and route-level rate/burst limits; REST APIs go further with **usage plans and API keys** for per-consumer quotas — the reason to reach for the older, pricier `aws apigateway` (REST) service instead of `aws apigatewayv2` (HTTP) is exactly those richer controls: request/response transformation, API keys, per-client usage plans, and private/edge endpoint types. For most internal ML services, HTTP APIs are the right default and REST is overkill.

For the simplest case, a **Lambda function URL** gives a function its own HTTPS endpoint with no API Gateway at all — good for internal or single-consumer inference, though you lose the richer controls. One caveat: for large inference responses, streaming a function URL (`RESPONSE_STREAM` invoke mode) avoids the buffered 6 MB response cap that a synchronous invoke imposes.

```bash
aws lambda create-function-url-config --function-name predict --auth-type AWS_IAM
```

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

S3, EventBridge, and other *push* sources invoke Lambda directly and you wire them with `add-permission` plus the source's own notification config (as above). Queue and stream sources — **SQS, Kinesis, DynamoDB Streams** — work the other way: Lambda *polls* them, and you configure that poller with an **event source mapping**. The `--batch-size` and `--maximum-batching-window-in-seconds` knobs are the throughput/latency dial: a larger batch amortizes per-invocation overhead and downstream calls (great for feeding a SageMaker endpoint that prefers batched requests), while a longer window waits to fill that batch. For streams, `--parallelization-factor` fans out per shard.

```bash
# SQS → Lambda: drain up to 10 messages per invocation
aws lambda create-event-source-mapping --function-name process \
  --event-source-arn arn:aws:sqs:us-east-1:<acct>:inference-queue \
  --batch-size 10

# Kinesis stream → Lambda: larger batches, bounded wait, per-shard fan-out
aws lambda create-event-source-mapping --function-name process \
  --event-source-arn arn:aws:kinesis:us-east-1:<acct>:stream/events \
  --starting-position LATEST --batch-size 100 \
  --maximum-batching-window-in-seconds 5 --parallelization-factor 2
```

A gotcha with SQS: the function's **reserved concurrency must be high enough** for the poller to scale, and a partial batch failure will redeliver the *whole* batch unless you report per-message failures — so idempotent handlers and a dead-letter queue are not optional at volume.

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

## CLI cheat-sheet

```bash
# --- Lambda: create (zip or container) ---
aws lambda create-function --function-name predict --runtime python3.12 \
  --handler app.handler --timeout 30 --memory-size 1024 \
  --role arn:aws:iam::<acct>:role/lambda-exec --zip-file fileb://function.zip
aws lambda create-function --function-name predict-img --package-type Image \
  --code ImageUri=<acct>.dkr.ecr.us-east-1.amazonaws.com/lambda-model:v1 \
  --role arn:aws:iam::<acct>:role/lambda-exec --timeout 60 --memory-size 3008

# --- Lambda: update, inspect, invoke ---
aws lambda update-function-code --function-name predict --zip-file fileb://function.zip
aws lambda update-function-configuration --function-name predict \
  --timeout 120 --memory-size 4096 --ephemeral-storage '{"Size": 4096}'
aws lambda get-function --function-name predict
aws lambda invoke --function-name predict --payload '{"features":[1,2,3]}' out.json

# --- Lambda: versions, aliases, concurrency ---
aws lambda publish-version --function-name predict
aws lambda create-alias --function-name predict --name prod --function-version 3
aws lambda put-provisioned-concurrency-config --function-name predict \
  --qualifier prod --provisioned-concurrent-executions 5     # pre-warm, no cold start
aws lambda put-function-concurrency --function-name predict \
  --reserved-concurrent-executions 50                        # cap total concurrency

# --- Lambda: event source mappings (poll SQS/Kinesis/DynamoDB) ---
aws lambda create-event-source-mapping --function-name process \
  --event-source-arn arn:aws:sqs:us-east-1:<acct>:inference-queue --batch-size 10
aws lambda create-event-source-mapping --function-name process \
  --event-source-arn arn:aws:kinesis:us-east-1:<acct>:stream/events \
  --starting-position LATEST --batch-size 100 \
  --maximum-batching-window-in-seconds 5 --parallelization-factor 2

# --- Lambda: function URL + invoke permission for API Gateway ---
aws lambda create-function-url-config --function-name predict --auth-type AWS_IAM
aws lambda add-permission --function-name predict --statement-id apigw \
  --action lambda:InvokeFunction --principal apigateway.amazonaws.com \
  --source-arn 'arn:aws:execute-api:us-east-1:<acct>:<id>/*/*/predict'

# --- API Gateway: HTTP API (apigatewayv2) ---
aws apigatewayv2 create-api --name ml-api --protocol-type HTTP
aws apigatewayv2 create-integration --api-id <id> --integration-type AWS_PROXY \
  --integration-uri arn:aws:lambda:us-east-1:<acct>:function:predict --payload-format-version 2.0
aws apigatewayv2 create-route --api-id <id> --route-key 'POST /predict' \
  --target integrations/<integration-id>
aws apigatewayv2 create-stage --api-id <id> --stage-name prod --auto-deploy
# REST API (apigateway) when you need usage plans / API keys / transforms

# --- S3 event trigger ---
aws s3api put-bucket-notification-configuration --bucket my-ml-data \
  --notification-configuration file://s3-notify.json
```

## Try it

Build a Lambda function packaged as a container image that loads a small model from S3 into `/tmp` at cold start and serves predictions. Front it with an HTTP API in API Gateway and call it with `curl`. Measure cold-start versus warm latency, then enable provisioned concurrency and measure again. Finally, add an S3 event trigger so that uploading a file to a `raw/` prefix invokes a second Lambda that validates the file — confirming the event-driven ingestion pattern end to end.
