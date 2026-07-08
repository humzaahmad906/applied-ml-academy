# 07 — Storage, Data, and Observability

A training job is only as good as the data it can read and the checkpoints it can write, and a serving fleet is only as reliable as your ability to see what it is doing. These two concerns — getting terabytes of data in and out of pods, and knowing whether the cluster is healthy and the GPUs are actually busy — are where ML-on-Kubernetes stops being about scheduling and starts being about plumbing. This lesson covers the storage stack (PersistentVolumeClaims and CSI, and the modern pattern of mounting object storage directly for datasets), secrets management, and the observability stack every serious cluster runs: Prometheus and Grafana for metrics, plus DCGM for the GPU metrics that generic monitoring misses entirely.

## PVCs and CSI: block and file storage

Pods are ephemeral; their local disk vanishes when they die. Anything that must survive — a checkpoint, a model artifact, a shared dataset — lives on a volume backed by real storage. Kubernetes exposes this through the **PersistentVolumeClaim (PVC)**: a pod asks for storage of a given size and access mode, and a **CSI (Container Storage Interface)** driver provisions the actual disk from the cloud (EBS, GCE PD, Azure Disk) or a file system (EFS, Filestore, Azure Files).

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: checkpoints}
spec:
  accessModes: ["ReadWriteOnce"]      # one node mounts it read-write
  storageClassName: premium-ssd       # fast block storage for checkpoints
  resources: {requests: {storage: 500Gi}}
```

The **access mode** is the detail that trips people up. `ReadWriteOnce` (RWO) — the common, cheap default backed by block storage — can be mounted read-write by pods on **one node only**. That is fine for a single-node checkpoint volume, but a distributed job whose workers span nodes cannot all mount the same RWO volume. For shared read-write across nodes you need `ReadWriteMany` (RWX), backed by a networked file system (EFS, Filestore, Azure Files) — more expensive and often slower, so use it deliberately. For checkpoints, prefer fast RWO block storage per node and sync to object storage; reserve RWX for genuinely shared mutable state.

## The dataset problem: mount object storage directly

Training datasets are frequently terabytes and live in object storage — S3, GCS, Azure Blob — because that is where they are cheap and durable. The naive approach is to copy the whole dataset onto a PVC before training starts, which wastes time, disk, and money and does not scale. The modern pattern (2025-2026) is to **mount the object store directly** as a POSIX-like file system via a CSI driver, so training code that expects file paths works unchanged and data streams on demand:

- **AWS Mountpoint for S3 CSI Driver** mounts an S3 bucket as a read-optimized file system, with a per-node local cache (RAM or local NVMe) for hot data.
- **GCS FUSE CSI Driver** does the same for GCS on GKE, and recently added **FUSE Profiles** that auto-tune cache settings for training, checkpointing, or inference access patterns.

```yaml
      volumes:
        - name: dataset
          csi:
            driver: s3.csi.aws.com
            volumeAttributes:
              bucketName: myco-training-data
              mountOptions: "--cache /local-nvme --read-only"
      containers:
        - name: trainer
          volumeMounts:
            - {name: dataset, mountPath: /data, readOnly: true}
```

The winning pattern for large-scale training: keep the dataset in object storage as the source of truth, mount it via CSI so no pre-copy is needed, back it with a **local NVMe cache** on the GPU node for throughput, and write **checkpoints** either to the same mount (GCS FUSE has an optimized checkpointing profile) or to a fast PVC that you async-copy to object storage. This avoids the data-copy bottleneck while keeping GPUs fed. The failure mode to watch: object-store mounts have higher latency and lower random-IO performance than local disk, so for many-small-files datasets, pack into sharded formats (WebDataset, TFRecord, Parquet) that read sequentially.

## Secrets: keys out of the manifest

Training and serving pods need credentials — a Hugging Face token to pull gated weights, cloud credentials, a database password. These never belong in the image or in plain YAML. Kubernetes **Secrets** hold them, mounted as env vars or files:

```yaml
      containers:
        - name: trainer
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef: {name: hf-creds, key: token}
```

But a raw Kubernetes Secret is only base64-encoded, not encrypted, and readable by anyone with namespace access. Production clusters do two things: enable **encryption at rest** for Secrets in `etcd`, and pull real secrets from an external manager — **External Secrets Operator** or the **Secrets Store CSI Driver** syncing from AWS Secrets Manager, GCP Secret Manager, or Vault. Better still, use **workload identity** (IRSA on EKS, Workload Identity on GKE, Workload Identity on AKS) so pods authenticate to cloud APIs as a service account with **no long-lived key** to store or leak at all. That is the pattern to prefer wherever the cloud supports it.

## Observability: you cannot operate what you cannot see

A GPU cluster running blind is a cluster silently wasting money and dropping requests. The de facto stack is **Prometheus** (scrapes and stores time-series metrics) plus **Grafana** (dashboards and alerts), usually installed together via the `kube-prometheus-stack` Helm chart. Out of the box this gives you cluster health: node CPU/memory, pod restarts, pending pods, and the standard Kubernetes signals.

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

The gap this leaves is the one that matters most for ML: **generic monitoring cannot see the GPU**. CPU and memory graphs tell you nothing about whether your expensive H100s are actually doing work.

## DCGM: the GPU metrics that matter

NVIDIA **DCGM (Data Center GPU Manager)** and its **DCGM Exporter** fill that gap, exporting per-GPU metrics into Prometheus: **GPU utilization**, **memory used**, **SM (streaming-multiprocessor) activity**, **temperature**, **power draw**, **NVLink/PCIe throughput**, and — crucially — **ECC errors and XID errors** that flag failing hardware. The GPU Operator (lesson 02) deploys the DCGM exporter for you; otherwise install it as a DaemonSet on GPU nodes. With those metrics in Grafana you can finally answer the questions that determine cost and reliability:

- **Are the GPUs busy?** Low SM activity or GPU-util on nodes you are paying for means idle accelerators — the single biggest source of waste. This is the metric that justifies MIG/time-slicing (lesson 02) or tighter queueing (lesson 04).
- **Is anything about to fail?** Rising ECC/XID errors, thermal throttling, or power anomalies predict node failures before they crash a multi-day training run.
- **What should autoscaling scale on?** Feed DCGM (and vLLM queue) metrics to HPA/KEDA (lesson 03) so scaling reacts to real GPU pressure.

The single most valuable dashboard on an ML cluster is **GPU utilization over cost**: it turns "we spend $40k/month on GPUs" into "we spend $40k/month and they are 35% utilized," which is the sentence that drives every efficiency decision — right-sizing requests, enabling sharing, tightening quotas.

## Logs and traces: the other half of observability

Metrics tell you *that* something is wrong; **logs** tell you *why*. Pod logs are ephemeral — they vanish when a pod is deleted, which for a completed training Job is immediately. A cluster you can operate ships logs off the node to a durable store: the common stacks are **Loki** (lightweight, integrates directly into Grafana alongside your metrics) or the older **EFK** (Elasticsearch/Fluentd/Kibana). A logging agent runs as a DaemonSet on every node, tails container stdout/stderr, and forwards it. For ML specifically, this is what lets you read *why* a training run crashed at 2 a.m. after its pod is long gone, or grep across all serving replicas for the request that produced a bad prediction. For request-level performance in a multi-service serving path (router to model to post-processor), **distributed tracing** (OpenTelemetry to Tempo or Jaeger) shows where the latency actually goes. Metrics, logs, and traces are the three pillars; a serious ML cluster runs all three, with metrics+DCGM as the always-on layer and logs/traces as the debugging layer you reach for when metrics flag a problem.

## Cost and operations notes

Storage and observability are both quiet cost centers. **Orphaned PVCs** outlive the Jobs that created them and bill indefinitely — reclaim them (a `Retain` vs `Delete` reclaim policy decision) and audit regularly. **High-performance RWX file storage** is far pricier than object storage; use it only where genuinely needed. On the metrics side, Prometheus retention and cardinality can themselves become expensive at scale — cap retention and avoid high-cardinality labels. But the return dwarfs the cost: a DCGM dashboard that reveals a fleet running at 30% utilization typically pays for the entire observability stack many times over in the first optimization it enables.

## Key takeaways

- **PVC + CSI** provides durable storage; mind the **access mode** — `ReadWriteOnce` (block, cheap, single-node) vs `ReadWriteMany` (networked file, pricier, multi-node shared). Use RWO+object-storage sync for checkpoints, RWX only when truly shared.
- For **datasets**, mount object storage directly via a CSI driver (**Mountpoint-S3**, **GCS FUSE**) with a **local NVMe cache** — no pre-copy, code sees file paths. Shard many-small-files data into sequential formats to beat object-store latency.
- Keep credentials in **Secrets** (encrypted at rest, ideally synced from an external manager), but prefer **workload identity** (IRSA / Workload Identity) so pods use short-lived credentials with no stored keys.
- **Prometheus + Grafana** (`kube-prometheus-stack`) covers cluster health but **cannot see the GPU**; **DCGM Exporter** (deployed by the GPU Operator) adds per-GPU util, memory, SM activity, power, and ECC/XID error metrics.
- The highest-value ML dashboard is **GPU utilization vs cost** — it exposes idle accelerators (the biggest waste) and predicts hardware failure before it kills long runs.

## Try it

1. Create a `ReadWriteOnce` PVC, mount it in a Job that writes a checkpoint file, delete the Job, and confirm a new pod can re-read the file — then try to mount it from pods on two different nodes and observe the RWO restriction.
2. Mount an S3 or GCS bucket via the CSI driver into a pod, `ls` the dataset from inside the container without copying it, and note the mount options for the local cache.
3. Store a token in a Secret, consume it as an env var, then reconfigure the pod to use workload identity instead and confirm it authenticates with no stored key.
4. Install `kube-prometheus-stack`, open Grafana, and find the built-in cluster dashboards — note that none of them show GPU utilization.
5. Deploy the DCGM exporter (or confirm the GPU Operator did), import a DCGM Grafana dashboard, run a GPU job, and watch SM activity and memory climb. Leave a GPU node idle and confirm the dashboard shows the waste.
