# 06 — Advanced Topics: Everything Else Worth Knowing — Part 3 of 5: Streaming, Kubernetes, and Observability Internals

This is part 3 of the Advanced Topics reference catalog. Here we cover streaming internals (Flink/Kafka), Kubernetes for ML at depth, and observability done properly.

## Phase 8 — Streaming Internals (Flink, Kafka)

### Watermarks

Flink's answer to "when have I seen all events for a window?"

```
watermark = max_event_time_seen - allowed_lateness
```

When the watermark passes the end of a window, the window closes. Late events go to a side output.

### State Backends

- **HashMapStateBackend** — JVM heap; fast; size-limited
- **EmbeddedRocksDBStateBackend** — disk-spillable; TB-scale state; slower per access
- **Remote state** (Flink 1.18+) — emerging

### Checkpointing

Periodic asynchronous snapshots of all state to durable storage (S3). On failure, restore from the last checkpoint. Exactly-once via the Chandy-Lamport algorithm.

### Savepoints

User-triggered durable snapshots. Stop the job, change code, restart from savepoint. The feature that makes Flink production-ready.

### Kafka KRaft

Kafka removed Zookeeper. KRaft mode is Kafka's own Raft-based metadata layer. Standard for new deployments.

### Exactly-Once Across Kafka and a Sink

Two-phase commit: prepare → barrier flows through job → commit on completion. Requires sink support (Iceberg, Kafka transactional producer, JDBC with transactions).

### Exercises

1. Run a Flink job that processes a Kafka topic and writes to Iceberg with exactly-once semantics. Trigger a failure mid-run. Verify no duplicates.
2. Trigger a savepoint. Modify the job. Restore.
3. Read [Streaming 101](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/) and [102](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-102/) by Tyler Akidau.

---

## Phase 9 — Kubernetes for ML at Depth

### GPU Scheduling

- **NVIDIA Device Plugin** — exposes `nvidia.com/gpu` as a schedulable resource
- **MIG (Multi-Instance GPU)** — partition an A100/H100 into smaller instances. Schedule like multiple GPUs.
- **GPU Operator** — installs drivers + plugin + monitoring in one operator
- **Time-slicing** — allow multiple pods to share a GPU (no isolation; for non-prod)

### Networking for ML

- **NCCL** for GPU collective comms; needs proper IPC and shared-memory setup
- **InfiniBand / RoCE** via SR-IOV or Multus
- **Topology-aware scheduling** — co-locate pods that need to communicate
- **NodePort vs LoadBalancer vs Ingress** — picking the right service type for serving

### Storage

- **PV / PVC abstractions**
- **CSI drivers** for cloud storage (EBS, EFS, GCS, FSx for Lustre)
- **Lustre / FSx Lustre / WekaFS** for high-throughput training data
- **Tiered storage**: hot SSD per-node + warm shared filesystem + cold S3

### Autoscaling

- **HPA** — pod replicas by CPU/memory or custom metric
- **VPA** — pod resource requests
- **KEDA** — event-driven autoscaling (Kafka lag, queue depth, custom)
- **Cluster Autoscaler** — node count
- **Karpenter** (AWS) — more advanced node provisioning

### Operators You'll Touch

- Kubeflow Training Operator (PyTorchJob, TFJob, XGBoostJob, ...)
- KubeRay (RayCluster, RayJob, RayService)
- KServe (InferenceService)
- Spark Operator (SparkApplication)
- Flink Operator (FlinkDeployment)
- Argo Workflows (Workflow CRD)

### Multi-Tenancy

- Namespaces + RBAC
- Network Policies (Calico, Cilium)
- ResourceQuotas, LimitRanges
- Hierarchical Namespaces (HNC) for organizational structure
- Pod Security Standards
- vCluster for true multi-tenant Kubernetes-in-Kubernetes

### Exercises

1. Set up MIG on a single A100 (or simulate). Schedule three small pods on three MIG instances.
2. Run Karpenter in a kind cluster (with a karpenter-on-kind setup). Watch it provision nodes for a workload.
3. Build a GitOps stack: Argo CD watching a Git repo; every commit reconciles the cluster.

---

## Phase 10 — Observability for ML, Done Properly

### The Stack

- **Metrics:** Prometheus + Mimir (long retention)
- **Logs:** Loki + Vector (collector)
- **Traces:** Tempo + OpenTelemetry SDKs
- **Profiles:** Pyroscope + Grafana for continuous profiling
- **Dashboards / alerting:** Grafana + Grafana OnCall / PagerDuty

All Grafana-stack — the dominant open-source observability stack.

### What to Instrument

In an ML service, in addition to standard request/error/duration:

- Model version label on every metric
- Feature distribution histograms (per feature, per window)
- Prediction distribution histograms
- Per-slice metrics (group, country, segment)
- Feature freshness (max age of features at prediction time)
- Cache hit/miss for features and predictions
- LLM-specific: tokens in, tokens out, time-to-first-token, tokens/sec, KV cache utilization
- GPU-specific: utilization, memory, power, temperature

### Slicing

Always slice. Aggregate hides:

- Geo slice — a region degraded
- Tenant / customer slice — one big tenant broken
- Device slice — mobile vs desktop
- Cohort slice — new users vs existing

### SLOs and Error Budgets

- Define SLOs: availability, latency, freshness, accuracy
- Burn-rate alerting: alert when current burn rate would exhaust the budget early
- Tie deployment freezes to error budget exhaustion. Forces the team to invest in reliability.

### LLM-Specific Observability

- Token cost per request, per endpoint, per tenant
- Time-to-first-token (latency-critical for streaming UX)
- Generation length distribution
- Cache hit rates (semantic + exact)
- Refusal / safety filter triggers
- Eval scores over time (drift in LLM quality is real)

Tools: Langfuse, W&B Weave, Braintrust, Helicone.

### Exercises

1. Instrument your serving service with Prometheus + OpenTelemetry.
2. Build a Grafana dashboard with at least 12 panels covering system + model + business metrics.
3. Define 3 SLOs. Implement burn-rate alerts.
4. For an LLM project, add Langfuse tracing. Look at traces; identify a slow span.

---

## You can now

- Reason about Flink watermarks, state backends, checkpointing, and savepoints, and design exactly-once delivery across Kafka and a sink.
- Configure GPU scheduling (device plugin, MIG, time-slicing), ML-aware networking and storage, and autoscaling (HPA/VPA/KEDA/Karpenter) for a Kubernetes ML platform.
- Instrument an ML service with the right observability stack — metrics, logs, traces, slicing by segment, SLOs with burn-rate alerts, and LLM-specific signals like time-to-first-token and cache hit rate.
