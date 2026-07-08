# 03 — Scheduling and Autoscaling

A GPU cluster has two kinds of waste and two matching autoscalers. **Pods** can be over- or under-provisioned relative to what they actually use, and pod-level autoscaling fixes that by adding or removing replicas. **Nodes** can sit idle or run out, and node-level autoscaling fixes that by adding or removing machines. Get both wrong and you either burn money on idle H100s or drop traffic when load spikes. This lesson covers the full stack: how resource requests and limits drive the scheduler, how the Horizontal Pod Autoscaler and KEDA scale replicas (including the event- and queue-driven scaling ML workloads really need), how the Cluster Autoscaler and Karpenter scale the node fleet, and how to build scale-to-zero for inference without paying the cold-start tax.

## Requests and limits: the contract with the scheduler

Every scaling decision starts with `resources.requests` and `resources.limits`, so getting them right is prerequisite to everything else.

- **Requests** are what the scheduler reserves. A pod requesting `cpu: "4"` will only land on a node with 4 free CPU. Requests are the basis for bin-packing and for the node autoscaler's math.
- **Limits** are the hard ceiling the kubelet enforces. Exceed a **memory** limit and the pod is OOM-killed. Exceed a **CPU** limit and the pod is throttled (slowed, not killed).

The classic mistakes both cost money. Set requests too high and the scheduler reserves capacity nobody uses, so nodes fill up on paper while sitting idle — and the node autoscaler dutifully adds more. Set requests too low and pods pack tightly, then throttle or OOM under real load. The discipline is to measure actual usage (`kubectl top pods`, or Prometheus history) and set requests near the real steady-state, with limits giving headroom for spikes. For GPUs there is no such nuance: the request equals the limit and is a whole integer, so a GPU pod's footprint is exact.

## Horizontal Pod Autoscaler (HPA)

The **HPA** is core Kubernetes. It watches a metric across the pods of a Deployment and adjusts the replica count to hold that metric near a target. The textbook version scales on CPU:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-serve
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-serve
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: {type: Utilization, averageUtilization: 65}
```

For ML inference, CPU utilization is usually the *wrong* signal. A GPU model server can be at 100% GPU with low CPU, or the real bottleneck is request queue depth. HPA supports `Pods` (custom per-pod metrics) and `External` (metrics from outside the cluster) metric types precisely so you can scale on what matters: **request queue depth**, **p99 latency**, or vLLM's own counters (running/waiting requests, KV-cache utilization). You expose these through the Prometheus Adapter or a cloud's managed custom-metrics pipeline, then target them instead of CPU. The practitioner consensus is blunt: scaling GPU inference on GPU-utilization percentage behaves badly; scale on **queue depth** or **pending requests** instead.

## KEDA: event-driven and scale-to-zero

HPA has two limits that bite ML workloads: it cannot scale to zero (its floor is 1), and wiring arbitrary external metrics into it is awkward. **KEDA** (Kubernetes Event-Driven Autoscaling, a CNCF graduated project) fills both gaps. KEDA generates an HPA under the hood but adds 60-plus **scalers** for external event sources — Kafka lag, SQS/PubSub queue depth, Redis list length, Prometheus queries — and, crucially, it can **scale to and from zero**.

This is the natural fit for two ML patterns. First, **queue-driven batch scoring**: scale worker pods on the depth of a message queue, up when work piles up and down to zero when the queue drains, so you pay only while there is work.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: batch-scorer
spec:
  scaleTargetRef:
    name: batch-scorer          # the Deployment to scale
  minReplicaCount: 0            # scale to zero when the queue is empty
  maxReplicaCount: 50
  triggers:
    - type: aws-sqs-queue
      metadata:
        queueURL: https://sqs.us-east-1.amazonaws.com/123/scoring-jobs
        queueLength: "20"       # aim for ~20 messages per pod
        awsRegion: us-east-1
```

Second, **scheduled scaling** — scale a training-notebook pool up during working hours and to zero overnight. KEDA has a `cron` scaler for exactly this. Recent KEDA versions also added GPU-oriented external scaler support, so you can drive GPU pod counts from custom GPU metrics. The mental model: use plain HPA for steady request-rate services, reach for KEDA when the signal is a queue, an external event, or you need a hard zero floor.

## Scaling the node fleet: Cluster Autoscaler vs Karpenter

Pod autoscaling only helps if there is a node with room. When pods go `Pending` for lack of capacity, a **node autoscaler** adds machines; when nodes sit underutilized, it removes them. Two tools dominate.

**Cluster Autoscaler** is the mature, multi-cloud default. It works against pre-defined node groups (an EKS node group, a GKE node pool, an Azure VMSS): when a pod cannot schedule, it grows the matching group's desired count; when a node is empty and its pods can fit elsewhere, it drains and removes it. It is battle-tested across AWS, GCP, Azure, and bare metal. The downside is that you must define the node groups up front, and scale-up can take several minutes.

**Karpenter** takes a different approach: it skips node groups entirely and provisions nodes directly from cloud APIs, picking the cheapest instance type that satisfies the pending pods' combined requirements. Because it reasons about the actual pods, it right-sizes nodes and typically brings capacity up far faster (roughly a minute versus several). For heterogeneous GPU workloads on AWS — where you want the specific GPU instance type a job needs, and want it now — Karpenter is the clear default and is built into EKS Auto Mode. Its maturity varies by cloud: production-default on AWS, GA on Azure (AKS Node Auto Provisioning), and only preview/community on GCP as of this writing. Karpenter lives under `kubernetes-sigs` (SIG Autoscaling); note it is *not* an independent CNCF project despite common misattribution.

A Karpenter `NodePool` declares the instance families and limits it may provision from:

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu
spec:
  template:
    spec:
      requirements:
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values: ["p5", "g6"]        # H100 and L4 families
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]
  limits:
    nvidia.com/gpu: 64                 # cap total GPUs this pool will provision
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized  # pack down aggressively
```

Practical guidance: on AWS, Karpenter for GPU workloads; on GCP or genuinely multi-cloud setups, Cluster Autoscaler remains the safer choice. Either way, `min-nodes=0` / consolidation is what lets idle GPU pools disappear.

## Scale-to-zero for inference — and its catch

Scale-to-zero is the dream for intermittent inference: pay nothing when idle. The catch is the **cold start**. When a request arrives and there are zero replicas, you must schedule a pod, possibly provision a GPU node (minutes), pull a multi-gigabyte image, and load model weights onto the GPU (tens of seconds to minutes for an LLM). For a lightweight predictive model this is tolerable; for a large GPU model it is often unacceptable.

Two realistic postures. For **predictive / small models**, use KServe's serverless mode (backed by Knative) or KEDA to genuinely scale to zero, accepting a few seconds of cold start on the first request. For **GPU-heavy LLM serving**, the 2026 practitioner pattern is **not** true zero: keep a warm floor of one replica and autoscale above it on queue depth, because loading a large model onto a GPU from cold is too slow to hide behind a single request. If cost pressure demands it, scale the *node pool* to zero for batch and keep the serving pool warm — you are choosing which cold start you can afford.

The layered picture: **HPA/KEDA** decide how many pods to run; the **node autoscaler** (Cluster Autoscaler or Karpenter) decides how many machines to run under them. They react to the same `Pending` pressure from opposite ends, and tuning both — plus honest resource requests — is what keeps a GPU cluster both responsive and cheap.

## Key takeaways

- **Requests** drive scheduling and node-autoscaler math; **limits** cap usage (memory over-limit = OOM-kill, CPU over-limit = throttle). Set requests near real steady-state usage — too high wastes reserved capacity, too low causes throttling/OOM.
- **HPA** scales replicas on a metric; for inference, scale on **queue depth or pending requests**, not CPU or GPU-utilization percentage. Use `Pods`/`External` metric types via a Prometheus adapter.
- **KEDA** (CNCF graduated) adds 60+ event sources and **scale-to-zero** — the right tool for queue-driven batch scoring and scheduled (cron) scaling of notebook pools.
- Node scaling: **Cluster Autoscaler** is the mature multi-cloud default (uses node groups); **Karpenter** provisions nodes directly, faster and cheaper, and is the AWS default for heterogeneous GPU work (GA on Azure, preview on GCP).
- **Scale-to-zero** works for small/predictive models but not large GPU models (cold start = node provision + image pull + weight load). For LLM serving, keep a **warm floor of 1** and autoscale above it.

## Try it

1. Deploy a CPU-bound HTTP service, attach the CPU-based HPA above, then drive load (`hey` or `ab`) and watch `kubectl get hpa -w` add replicas, then scale back down when load stops.
2. Set a pod's memory limit deliberately low, run a workload that exceeds it, and observe the `OOMKilled` status in `kubectl describe pod`. Then do the same for CPU and observe throttling instead of a kill.
3. Stand up a KEDA `ScaledObject` with `minReplicaCount: 0` against a queue, push messages, and watch pods scale from zero; drain the queue and watch them return to zero.
4. Enable node autoscaling with a floor of zero on a GPU pool, submit a GPU job, and time how long a node takes to appear — that number is your cold-start budget.
5. Compare: expose vLLM's `num_requests_waiting` metric to HPA and scale on it; note how much better it tracks real load than CPU utilization does.
