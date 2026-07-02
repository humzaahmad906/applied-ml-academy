# 10 — SageMaker AI: Real-Time and Serverless Inference

Training produces an artifact; inference turns it into value. SageMaker AI offers four distinct ways to serve predictions, each tuned to a different traffic and latency profile, plus features for hosting many models efficiently and scaling to real demand. Choosing the wrong inference option is a common and expensive mistake — a real-time endpoint idling overnight, or a serverless setup timing out on a heavy model. This module maps the options to workloads and shows how to deploy and autoscale them.

## The four inference options

**Real-time endpoints** are persistent HTTPS endpoints backed by always-on instances, giving low, consistent latency for interactive requests. This is the default for user-facing prediction where latency matters and traffic is steady. You pay for the instances by the hour whether or not requests arrive.

**Serverless inference** runs with no instances to manage — SageMaker provisions capacity per request and scales to zero when idle, so you pay only for what you use. Ideal for intermittent or unpredictable traffic. The tradeoffs are bounded: payloads up to about 4 MB and processing up to about 60 seconds, plus cold starts on the first request after idle. Great for spiky, cost-sensitive workloads; wrong for large payloads or heavy models.

**Asynchronous inference** queues incoming requests and processes them in the background, returning results to S3. It handles large payloads (up to ~1 GB) and long processing (up to ~1 hour), and — crucially — it can **autoscale to zero instances** when the queue is empty, so you pay nothing between bursts. This is the sweet spot for expensive predictions (large documents, long-running generative jobs) that do not need a synchronous response.

**Batch transform** is not an endpoint at all: it spins up instances, runs inference over a whole dataset in S3, writes results back to S3, and shuts down. Use it for offline scoring of large datasets, backfills, and preprocessing where there is no need for a live service.

| Option | Latency | Payload / Duration | Idle cost | Best for |
| --- | --- | --- | --- | --- |
| Real-time | Low, consistent | Small, sub-second | Pay while running | Interactive, steady traffic |
| Serverless | Low, cold starts | ~4 MB / ~60 s | Zero | Intermittent, spiky |
| Asynchronous | Near-real-time | ~1 GB / ~1 hr | Zero (scales to 0) | Large/expensive requests |
| Batch transform | N/A (offline) | Whole datasets | Only during run | Bulk offline scoring |

## Deploying an endpoint

The SDK pattern is `Model` → `deploy()` → `Predictor`:

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://my-ml-data/models/model.tar.gz",
    role=role,
    framework_version="2.4",
    py_version="py311",
    entry_point="inference.py",
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    endpoint_name="fraud-scorer",
)
result = predictor.predict({"features": [0.1, 0.9, 0.3]})
```

For serverless, deploy with a `ServerlessInferenceConfig` (memory size and max concurrency) instead of an instance type; for asynchronous, provide an `AsyncInferenceConfig` with an S3 output location.

Under the SDK's `deploy()` sit three distinct API calls, and it pays to know them because updates, rollbacks, and CI/CD all work at this level: **`CreateModel`** (names the container image + artifact + role), **`CreateEndpointConfig`** (declares the production variants — instance type, count, and the serverless/async config), and **`CreateEndpoint`** (spins up the config as a live endpoint). The separation is deliberate: to ship a new model you create a *new* config and call `update-endpoint`, which does a managed blue/green swap with zero downtime, and you can roll back by pointing the endpoint at the previous config.

```bash
aws sagemaker create-model --model-name fraud-v3 \
  --primary-container Image=<ecr-image>,ModelDataUrl=s3://my-ml-data/models/model.tar.gz \
  --execution-role-arn arn:aws:iam::123456789012:role/SageMakerRole

# Real-time variant
aws sagemaker create-endpoint-config --endpoint-config-name fraud-v3-cfg \
  --production-variants VariantName=AllTraffic,ModelName=fraud-v3,InstanceType=ml.g5.xlarge,InitialInstanceCount=1

aws sagemaker create-endpoint --endpoint-name fraud-scorer --endpoint-config-name fraud-v3-cfg
aws sagemaker describe-endpoint --endpoint-name fraud-scorer --query 'EndpointStatus'

# Ship v4: new config, then swap in place (managed blue/green, no downtime)
aws sagemaker update-endpoint --endpoint-name fraud-scorer --endpoint-config-name fraud-v4-cfg
```

The serverless and async variants are the *same* `create-endpoint-config` call with a different production-variant shape — serverless swaps the instance fields for a `ServerlessConfig`, and async is a real-time variant plus a top-level `--async-inference-config`:

```bash
# Serverless variant (no instance type; memory + concurrency instead)
aws sagemaker create-endpoint-config --endpoint-config-name fraud-serverless-cfg \
  --production-variants VariantName=AllTraffic,ModelName=fraud-v3,ServerlessConfig={MemorySizeInMB=4096,MaxConcurrency=20}

# Async variant (real-time variant + S3 output location for results)
aws sagemaker create-endpoint-config --endpoint-config-name fraud-async-cfg \
  --production-variants VariantName=AllTraffic,ModelName=fraud-v3,InstanceType=ml.g5.xlarge,InitialInstanceCount=1 \
  --async-inference-config OutputConfig={S3OutputPath=s3://my-ml-data/async-out/}
```

Invocation lives in a *separate* service, `sagemaker-runtime` (a common surprise — `aws sagemaker invoke-endpoint` does not exist). Real-time is `invoke-endpoint`; async is `invoke-endpoint-async`, which takes an S3 input location and returns immediately with the S3 path where the result will land.

```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name fraud-scorer \
  --content-type application/json --body '{"features":[0.1,0.9,0.3]}' /dev/stdout

aws sagemaker-runtime invoke-endpoint-async --endpoint-name fraud-scorer \
  --content-type application/json --input-location s3://my-ml-data/async-in/req1.json
```

**Batch transform** is not an endpoint — it is its own job that reads a dataset from S3, scores it, and shuts down, so there is nothing to invoke or tear down afterward:

```bash
aws sagemaker create-transform-job --transform-job-name nightly-score \
  --model-name fraud-v3 \
  --transform-input DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://my-ml-data/score-in/}} \
  --transform-output S3OutputPath=s3://my-ml-data/score-out/ \
  --transform-resources InstanceType=ml.m5.xlarge,InstanceCount=4
```

## Hosting many models efficiently

Two features cut the cost of hosting large model fleets:

**Multi-model endpoints (MME)** host many models behind a single endpoint on shared instances; SageMaker loads a model into memory on demand and evicts idle ones. For organizations serving hundreds of low-traffic models (per-customer or per-segment models), MME can cut inference cost by 80–90% versus one endpoint per model, since you are not paying for idle capacity per model.

**Inference components** let you deploy multiple models — or multiple copies of a model — onto a shared endpoint with fine-grained control over how much CPU, memory, and how many accelerators each gets, and independent scaling per component. This is the modern way to pack several models onto expensive GPU instances at high utilization instead of dedicating a whole instance to each. **Multi-container endpoints** similarly place multiple containers behind one endpoint.

MME hosts all its models from a single S3 prefix — you add a model by uploading its `model.tar.gz` there (no redeploy) and invoke a specific one with the `--target-model` argument; the eviction of idle models is what causes the occasional cold-start latency spike, the main MME gotcha to plan around.

```bash
# Invoke a specific model on a multi-model endpoint
aws sagemaker-runtime invoke-endpoint --endpoint-name customer-models \
  --target-model customer-42.tar.gz \
  --content-type application/json --body '{"features":[...]}' /dev/stdout
```

Inference components are the newer, more explicit path: you create an endpoint with an empty config, then attach a `create-inference-component` per model declaring its compute (`--compute-resource-requirements`) and copy count, and scale each component independently — including scale-to-zero for components that go quiet.

```bash
aws sagemaker create-inference-component --inference-component-name ranker \
  --endpoint-name shared-gpu --variant-name AllTraffic \
  --specification 'ModelName=ranker-v2,ComputeResourceRequirements={NumberOfAcceleratorDevicesRequired=1,MinMemoryRequiredInMb=8192}' \
  --runtime-config CopyCount=2
```

## Autoscaling

Real-time endpoints scale with **application autoscaling** policies that adjust instance count based on a metric — most commonly `SageMakerVariantInvocationsPerInstance` (requests per instance) or a custom CloudWatch metric — between a min and max you set:

```python
import boto3
aas = boto3.client("application-autoscaling")
rid = "endpoint/fraud-scorer/variant/AllTraffic"
aas.register_scalable_target(
    ServiceNamespace="sagemaker", ResourceId=rid,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1, MaxCapacity=8,
)
aas.put_scaling_policy(
    PolicyName="invocations-target", ServiceNamespace="sagemaker",
    ResourceId=rid, ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 750.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"},
        "ScaleInCooldown": 300, "ScaleOutCooldown": 60,
    },
)
```

Set `MinCapacity` thoughtfully: 1 keeps a warm baseline, while asynchronous inference lets you scale the minimum to 0 for true pay-per-use. Deploying new versions safely uses **production variants** and traffic shifting (blue/green or canary) so you can roll out a new model behind the same endpoint without downtime.

The scale-to-zero pattern for async is the same two calls with `MinCapacity=0` and a policy on the `ApproximateBacklogSizePerInstance` metric — the queue depth, not invocation rate, is what drives async scaling from and back to zero:

```python
aas.register_scalable_target(
    ServiceNamespace="sagemaker", ResourceId="endpoint/fraud-async/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=0, MaxCapacity=4,   # true zero idle cost between bursts
)
```

Two operational gotchas: target-tracking only scales *out* on a breach and relies on the cooldowns to avoid thrashing (a too-short `ScaleInCooldown` flaps the fleet), and the `ResourceId` string format is exact — a typo in `endpoint/<name>/variant/<variant>` fails silently by simply never scaling. The raw CLI equivalents are `aws application-autoscaling register-scalable-target` and `put-scaling-policy` with the same arguments.

## How this fits the whole ML solution

Inference is where the model meets consumers, and each option corresponds to a real place in the architecture: real-time behind the API front door, serverless for a spiky internal feature, asynchronous for heavy generative jobs, batch transform for nightly scoring feeding the warehouse. The endpoint reads the artifact the model registry promoted, sits inside the VPC, is fronted by API Gateway/Lambda, and emits the metrics that monitoring watches for drift. Picking the right option per workload is a direct, ongoing lever on both latency and cost across the whole system.

## Key takeaways

- Four options: real-time (steady, low latency), serverless (spiky, scale-to-zero, ~4 MB/~60 s), asynchronous (large/expensive, scales to zero), batch transform (offline bulk scoring).
- Deploy via `Model.deploy()` → `Predictor`; choose serverless/async by passing the matching config instead of an instance type.
- Multi-model endpoints cut cost 80–90% for many low-traffic models; inference components pack multiple models onto shared (GPU) endpoints at high utilization.
- Autoscale real-time endpoints on invocations-per-instance; use asynchronous inference to reach true zero idle cost.
- Roll out new models safely with production variants and canary/blue-green traffic shifting.

## CLI cheat-sheet

```bash
# --- Deploy (Model -> EndpointConfig -> Endpoint) ---
aws sagemaker create-model --model-name fraud-v3 \
  --primary-container Image=<ecr-image>,ModelDataUrl=s3://.../model.tar.gz \
  --execution-role-arn <role>
aws sagemaker create-endpoint-config --endpoint-config-name fraud-cfg \
  --production-variants VariantName=AllTraffic,ModelName=fraud-v3,InstanceType=ml.g5.xlarge,InitialInstanceCount=1
aws sagemaker create-endpoint --endpoint-name fraud-scorer --endpoint-config-name fraud-cfg
aws sagemaker describe-endpoint --endpoint-name fraud-scorer --query 'EndpointStatus'
aws sagemaker update-endpoint --endpoint-name fraud-scorer --endpoint-config-name fraud-v4-cfg  # blue/green
aws sagemaker delete-endpoint --endpoint-name fraud-scorer   # STOP paying: delete idle endpoints

# Serverless / async variants (same create-endpoint-config, different variant shape)
#   ServerlessConfig={MemorySizeInMB=4096,MaxConcurrency=20}
#   --async-inference-config OutputConfig={S3OutputPath=s3://.../async-out/}

# --- Invoke (separate service: sagemaker-runtime) ---
aws sagemaker-runtime invoke-endpoint --endpoint-name fraud-scorer \
  --content-type application/json --body '{"features":[0.1,0.9]}' /dev/stdout
aws sagemaker-runtime invoke-endpoint-async --endpoint-name fraud-scorer \
  --content-type application/json --input-location s3://.../async-in/req.json
aws sagemaker-runtime invoke-endpoint --endpoint-name customer-models \
  --target-model customer-42.tar.gz --content-type application/json --body '{...}' /dev/stdout  # MME

# --- Batch transform (no endpoint) ---
aws sagemaker create-transform-job --transform-job-name nightly-score --model-name fraud-v3 \
  --transform-input DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://.../in/}} \
  --transform-output S3OutputPath=s3://.../out/ \
  --transform-resources InstanceType=ml.m5.xlarge,InstanceCount=4

# --- Inference components (pack models on shared GPU) ---
aws sagemaker create-inference-component --inference-component-name ranker \
  --endpoint-name shared-gpu --variant-name AllTraffic \
  --specification 'ModelName=ranker-v2,ComputeResourceRequirements={NumberOfAcceleratorDevicesRequired=1,MinMemoryRequiredInMb=8192}' \
  --runtime-config CopyCount=2

# --- Autoscaling (application-autoscaling; namespace sagemaker) ---
aws application-autoscaling register-scalable-target --service-namespace sagemaker \
  --resource-id endpoint/fraud-scorer/variant/AllTraffic \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --min-capacity 1 --max-capacity 8            # min 0 for async scale-to-zero
aws application-autoscaling put-scaling-policy --service-namespace sagemaker \
  --resource-id endpoint/fraud-scorer/variant/AllTraffic \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --policy-name invocations-target --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://policy.json
# policy.json metric: SageMakerVariantInvocationsPerInstance (real-time)
#                     ApproximateBacklogSizePerInstance      (async queue depth)
```

## Try it

Deploy the same model three ways — a real-time endpoint, a serverless endpoint, and an asynchronous endpoint — and send each a burst of requests followed by an idle period. Compare latency, cold-start behavior, and cost. Then attach a target-tracking autoscaling policy to the real-time endpoint and drive load until it scales out. Finally, host two small models on a single multi-model endpoint and confirm both respond, observing how SageMaker loads each on demand.
