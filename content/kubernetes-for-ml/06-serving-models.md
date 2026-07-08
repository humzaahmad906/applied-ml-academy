# 06 — Serving Models

Training produces a checkpoint; serving turns it into a product. You *can* serve a model with the raw primitives from lesson 01 — a Deployment running your inference container, a Service in front, an HPA to scale — and for a small predictive model that is a perfectly good answer. But modern ML serving, especially LLM serving, has demands that raw Deployments do not address: loading multi-gigabyte weights, autoscaling on the right signals, canary rollouts of new model versions, and the disaggregated, KV-cache-aware architectures that make large-model inference economical. This lesson covers the serving stack — KServe as the Kubernetes-native inference platform, vLLM and its production stack, NVIDIA Dynamo and llm-d for disaggregated LLM serving, and the rollout and autoscaling patterns that keep a GPU endpoint both reliable and affordable.

## The baseline: Deployment + Service + probes

Before reaching for a platform, know the manual pattern, because everything else is built on it. An inference server is a Deployment whose pods load the model at startup and expose an HTTP or gRPC endpoint, fronted by a Service, with **probes** that gate traffic correctly.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: {name: sentiment}
spec:
  replicas: 2
  selector: {matchLabels: {app: sentiment}}
  template:
    metadata: {labels: {app: sentiment}}
    spec:
      containers:
        - name: server
          image: myco/sentiment-serve:v4
          ports: [{containerPort: 8080}]
          resources: {limits: {nvidia.com/gpu: 1, memory: "24Gi"}}
          startupProbe:                 # give slow model loads time before liveness kicks in
            httpGet: {path: /healthz, port: 8080}
            failureThreshold: 30
            periodSeconds: 10
          readinessProbe:               # only route traffic once weights are loaded
            httpGet: {path: /ready, port: 8080}
```

The **startupProbe** is the detail people miss: loading a large model can take minutes, and without it the liveness probe fails and Kubernetes kills the pod in a loop before it ever finishes loading. The **readinessProbe** ensures the Service never routes a request to a pod still warming up. Get these two right and most serving flakiness disappears.

## KServe: the inference platform

Writing that Deployment, its Service, an HPA, a canary rollout, and a monitoring stack by hand for every model does not scale across a platform. **KServe** (which became a **CNCF incubating** project in 2025, having earlier moved from KFServing under the LF AI & Data foundation) abstracts all of it behind a single `InferenceService` custom resource. You declare the model, its framework, and resource needs; KServe generates the Deployment, autoscaling, routing, and a standard prediction API. It is framework-agnostic — scikit-learn, XGBoost, PyTorch, TensorFlow, and, importantly, vLLM and TensorRT-LLM for LLMs.

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata: {name: sentiment}
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 10
    model:
      modelFormat: {name: sklearn}
      storageUri: gs://myco-models/sentiment/v4    # KServe pulls the model for you
      resources: {limits: {cpu: "2", memory: "4Gi"}}
```

KServe has two operating modes. **Serverless mode** builds on **Knative** and supports true **scale-to-zero** — the endpoint drops to zero pods when idle and cold-starts on the next request. This is excellent for predictive/small models with intermittent traffic and is a large cost win. **Raw Deployment mode** uses standard Kubernetes Deployments + HPA without the Knative dependency, which is simpler to operate and the usual choice for always-warm GPU serving. The rule of thumb: serverless for cheap, bursty CPU/small-model inference; raw mode with a warm floor for GPU/LLM serving where cold starts are too expensive.

## Serving LLMs: vLLM, and why it is different

Large language models break the simple serving model in two ways: the weights are huge (loading is slow, so scale-to-zero hurts), and generation is autoregressive (each request occupies the GPU for many token-steps, so throughput depends on smart batching). **vLLM** is the de facto open-source LLM inference engine; its **PagedAttention** and **continuous batching** dramatically raise GPU throughput versus naive serving. You can run vLLM directly as a container, but two higher-level options exist:

- **KServe with a vLLM runtime.** Point an `InferenceService` at a vLLM serving runtime and get vLLM's throughput inside KServe's platform (routing, autoscaling, canary). KServe also added a purpose-built **`LLMInferenceService`** resource for generative workloads, with LLM-aware features like KV-cache-aware routing and prefix caching.
- **vLLM Production Stack.** An official Helm chart from the vLLM project that bundles the serving engine with a prefix- and session-aware request router, KV-cache offloading, and Prometheus/Grafana observability — a batteries-included way to run vLLM at multi-replica scale without KServe.

## Disaggregated serving: Dynamo and llm-d

The frontier of LLM serving in 2026 is **disaggregation**. An LLM request has two phases with opposite hardware profiles: **prefill** (processing the prompt) is compute-bound, while **decode** (generating tokens one at a time) is memory-bandwidth-bound. Running both on the same GPU pool wastes one resource or the other. Disaggregated serving splits prefill and decode onto separate, independently-scaled GPU pools and streams the KV cache between them, plus routes requests to whichever replica already has relevant KV cache warm.

Two projects lead here, and both target Kubernetes:

- **NVIDIA Dynamo** reached GA in early 2026. It is an open-source inference framework for disaggregated serving with an LLM-aware router (tracks KV-cache overlap), an SLA-based planner that autoscales on time-to-first-token and inter-token-latency targets, and a KV block manager that tiers cache across GPU/CPU/SSD. It integrates with Kubernetes for gang-scheduled, topology-aware placement and claims large throughput gains over monolithic serving.
- **llm-d** launched in 2025 (Red Hat, Google, IBM, NVIDIA, CoreWeave) as a Kubernetes-native distributed-inference framework with prefill/decode disaggregation, tiered KV-cache offloading, and SLO-aware autoscaling. It is the engine underneath KServe's `LLMInferenceService`.

These are complementary more than competing — Dynamo can accelerate an llm-d deployment — but both are **early-adopter-stage** technology as of 2026, and the headline throughput multipliers come from vendor benchmarks. Treat them as "know they exist and what problem they solve"; reach for them only when a single-node vLLM deployment genuinely cannot meet your latency or cost targets, and expect the APIs to still be moving. *(Maturity and benchmark claims here are vendor-sourced and fast-moving — verify against current docs before committing.)*

## Online, batch, and streaming inference

Not all serving is a synchronous HTTP request. Three shapes recur, and Kubernetes handles each differently. **Online (real-time)** inference — a request comes in, a prediction goes out in milliseconds — is the Deployment/KServe pattern above. **Batch (offline)** inference — score a million rows overnight — is not a service at all; it is a **Job** (lesson 04), often an indexed Job fanning out across shards, that reads inputs from object storage and writes predictions back, then exits. Do not run batch scoring through your online endpoint: it will swamp the autoscaler and starve real-time traffic. **Streaming** inference — score events off a Kafka or PubSub topic as they arrive — is a long-running consumer Deployment, ideally autoscaled by **KEDA** on the topic's lag (lesson 03), so it scales with the event backlog and to zero when the stream is quiet.

The mistake to avoid is forcing everything through one always-on GPU endpoint. Matching the serving shape to the workload — Job for batch, KEDA-scaled consumer for streaming, warm Deployment/KServe for online — is what keeps GPU spend proportional to actual work rather than provisioned for the peak of all three at once.

## Rollouts: canary and rollback

A new model version is a production risk — it may be slower, more expensive, or worse on real traffic than offline evals suggested. You never flip 100% of traffic to it blind. The **canary** pattern sends a small slice (say 10%) to the new version, watches metrics, and ramps up only if it holds — or rolls back instantly by shifting traffic away. KServe makes this a one-field change with `canaryTrafficPercent`:

```yaml
spec:
  predictor:
    canaryTrafficPercent: 10          # 10% to the new revision, 90% to the last-good
    model:
      storageUri: gs://myco-models/sentiment/v5
```

KServe keeps the previous revision live and splits traffic, so promotion and rollback are traffic shifts, not redeploys. On raw Kubernetes you get a coarser version via the Deployment's rolling update, or a finer one with a service mesh (Istio/Linkerd) or Gateway API traffic splitting. Either way the principle is the same as canarying any service — but the stakes are higher because model quality regressions are silent, so pair the canary with **online metric comparison** (latency, error rate, and a business/quality signal) before you promote.

## GPU inference autoscaling

Autoscaling GPU serving is where cost is won. The lessons from lesson 03 apply directly: scale on **queue depth / pending requests**, not GPU-utilization percentage, because a saturated LLM server can show modest GPU-util while requests pile up in its queue. KServe's autoscaling (Knative concurrency in serverless mode, HPA/KEDA in raw mode) can target vLLM's own metrics — running vs waiting requests, KV-cache utilization. The scale-to-zero caveat is the same: fine for small models, but for LLMs keep a **warm floor of one replica** and autoscale above it, because loading tens of gigabytes onto a GPU from cold cannot hide behind a single request. Where cost demands aggressive savings, scale the *node pool* down for batch work while keeping the serving pool warm.

## Key takeaways

- Baseline serving is a **Deployment + Service + HPA**; the make-or-break detail is a **startupProbe** (so slow model loads are not killed) plus a **readinessProbe** (so traffic waits for weights to load).
- **KServe** (CNCF incubating as of 2025) abstracts serving behind an `InferenceService`: it generates the Deployment, autoscaling, routing, and a standard API, framework-agnostic. **Serverless mode** (Knative) gives scale-to-zero for small models; **raw mode** is simpler for always-warm GPU serving.
- **vLLM** (PagedAttention + continuous batching) is the standard LLM engine; run it via a KServe vLLM runtime / `LLMInferenceService`, or the official **vLLM Production Stack** Helm chart.
- **Disaggregated serving** splits compute-bound **prefill** from memory-bound **decode** onto separate pools. **NVIDIA Dynamo** (GA 2026) and **llm-d** (2025) lead here — powerful but early-adopter-stage with vendor-sourced benchmarks; use only when single-node vLLM cannot meet targets.
- Roll out with **canary** traffic splitting (KServe `canaryTrafficPercent`) and online metric comparison; autoscale on **queue depth**, keep a **warm floor** for LLMs rather than true scale-to-zero.

## Try it

1. Serve a small model as a raw Deployment + Service with both a startupProbe and readinessProbe. Deliberately omit the startupProbe on a slow-loading model and watch the liveness-probe kill loop; add it back and confirm the pod stabilizes.
2. Install KServe and deploy the same model as an `InferenceService`; compare how much YAML you wrote versus the manual version, and hit the standard prediction endpoint.
3. Deploy an LLM with a vLLM runtime under KServe, then load-test it and expose vLLM's `num_requests_waiting` metric to the autoscaler; confirm replicas track queue depth, not GPU-util.
4. Do a canary: deploy v2 with `canaryTrafficPercent: 20`, send traffic, verify roughly 20% hits v2, then promote to 100% — and practice an instant rollback by setting it back to 0.
5. Read the KServe serverless vs raw-mode docs and decide, for one small model and one LLM, which mode each should use and why.
