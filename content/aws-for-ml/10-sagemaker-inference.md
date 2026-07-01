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

## Hosting many models efficiently

Two features cut the cost of hosting large model fleets:

**Multi-model endpoints (MME)** host many models behind a single endpoint on shared instances; SageMaker loads a model into memory on demand and evicts idle ones. For organizations serving hundreds of low-traffic models (per-customer or per-segment models), MME can cut inference cost by 80–90% versus one endpoint per model, since you are not paying for idle capacity per model.

**Inference components** let you deploy multiple models — or multiple copies of a model — onto a shared endpoint with fine-grained control over how much CPU, memory, and how many accelerators each gets, and independent scaling per component. This is the modern way to pack several models onto expensive GPU instances at high utilization instead of dedicating a whole instance to each. **Multi-container endpoints** similarly place multiple containers behind one endpoint.

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

## How this fits the whole ML solution

Inference is where the model meets consumers, and each option corresponds to a real place in the architecture: real-time behind the API front door, serverless for a spiky internal feature, asynchronous for heavy generative jobs, batch transform for nightly scoring feeding the warehouse. The endpoint reads the artifact the model registry promoted, sits inside the VPC, is fronted by API Gateway/Lambda, and emits the metrics that monitoring watches for drift. Picking the right option per workload is a direct, ongoing lever on both latency and cost across the whole system.

## Key takeaways

- Four options: real-time (steady, low latency), serverless (spiky, scale-to-zero, ~4 MB/~60 s), asynchronous (large/expensive, scales to zero), batch transform (offline bulk scoring).
- Deploy via `Model.deploy()` → `Predictor`; choose serverless/async by passing the matching config instead of an instance type.
- Multi-model endpoints cut cost 80–90% for many low-traffic models; inference components pack multiple models onto shared (GPU) endpoints at high utilization.
- Autoscale real-time endpoints on invocations-per-instance; use asynchronous inference to reach true zero idle cost.
- Roll out new models safely with production variants and canary/blue-green traffic shifting.

## Try it

Deploy the same model three ways — a real-time endpoint, a serverless endpoint, and an asynchronous endpoint — and send each a burst of requests followed by an idle period. Compare latency, cold-start behavior, and cost. Then attach a target-tracking autoscaling policy to the real-time endpoint and drive load until it scales out. Finally, host two small models on a single multi-model endpoint and confirm both respond, observing how SageMaker loads each on demand.
