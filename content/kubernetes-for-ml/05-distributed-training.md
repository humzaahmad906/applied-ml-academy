# 05 — Distributed Training

When a model or its data no longer fits or trains fast enough on a single GPU, you go distributed: many GPUs, often across many nodes, working on one training run. The hard part is not the ML — PyTorch's DistributedDataParallel handles the gradient synchronization — it is the orchestration. Every worker needs to know how many peers exist, which one is rank 0, how to reach the others over the network, and they must all start together or the whole collective hangs. Doing this by hand with raw pods and a headless service is possible but tedious and fragile. This lesson covers the two dominant Kubernetes-native paths — the Kubeflow Trainer (the successor to the Training Operator's PyTorchJob) and KubeRay — plus the multi-node NCCL and networking realities that make or break throughput.

## What distributed training needs from Kubernetes

PyTorch DDP (and FSDP for sharding large models) runs one process per GPU. To bootstrap, `torchrun` needs four things wired into every worker: the **world size** (total processes), each process's **rank**, the **rendezvous address** (where workers find each other, usually rank 0's hostname), and a **backend** — **NCCL** for GPU-to-GPU communication. On Kubernetes that translates to three requirements:

1. **Stable, resolvable hostnames** for every worker, so rank 3 can reach rank 0. This is a **headless service** giving each pod its own DNS record.
2. **Gang scheduling** — all workers scheduled together, or none. A DDP job with three of four workers up will hang forever waiting for the fourth (the deadlock from lesson 04).
3. **Correct rank/world-size env vars** injected per pod.

Both tools below exist to generate exactly this plumbing so you do not hand-write it.

## Kubeflow Trainer: the modern path

The Kubeflow **Training Operator** was for years the standard way to run distributed PyTorch on Kubernetes, via a `PyTorchJob` custom resource. In 2025 the project was redesigned and renamed to **Kubeflow Trainer** (the repo moved from `training-operator` to `trainer`), and it introduced a new API. The single unified **TrainJob** CRD now replaces the old per-framework resources (`PyTorchJob`, `TFJob`, `MPIJob`, `XGBoostJob`, `JAXJob`). Trainer joined the PyTorch ecosystem officially. The v2 API is still marked **alpha**, so treat exact field names as subject to change — but it is the direction the project is going, and it is what to learn for new work.

The v2 design splits responsibilities. Platform engineers define a **ClusterTrainingRuntime** (or namespaced `TrainingRuntime`) that encodes the framework, image, parallelism strategy, and networking — the reusable "how." Data scientists submit a lightweight **TrainJob** that references a runtime and supplies the "what": number of nodes, resources, and the training command or a fine-tuning config.

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: llama-sft
  namespace: team-nlp
spec:
  runtimeRef:
    name: torch-distributed        # a ClusterTrainingRuntime installed by the platform
    kind: ClusterTrainingRuntime
  trainer:
    numNodes: 4                     # 4 worker pods
    numProcPerNode: "8"             # 8 GPUs each -> world size 32
    resourcesPerNode:
      limits:
        nvidia.com/gpu: 8
    command: ["torchrun", "sft.py", "--model=llama-3-8b"]
```

Under the hood, Trainer v2 builds the job from a **JobSet** — a `kubernetes-sigs` primitive that groups several indexed Jobs into one unit with a headless service and startup ordering. You rarely write a JobSet directly; Trainer (and other tools) generate it for you, and it is what provides the stable per-pod hostnames DDP needs. Trainer integrates natively with **Kueue** for admission and quota, so a TrainJob is suspended until its team's GPU quota is free, then admitted as a gang. Submit and watch it exactly like any workload:

```bash
kubectl apply -f llama-sft.yaml
kubectl get trainjob -n team-nlp
kubectl logs -f -l trainer.kubeflow.org/trainjob-name=llama-sft -n team-nlp
```

**Legacy note:** enormous numbers of production clusters still run the v1 `PyTorchJob` (API group `kubeflow.org/v1`). Its source is frozen on the `release-1.9` branch and receives no new features, but it works, and a migration guide exists. If you inherit a cluster, expect to see `PyTorchJob`; for greenfield, use `TrainJob`.

## KubeRay: distributed training the Ray way

The other major path is **Ray on Kubernetes** via **KubeRay** (maintained by the Ray project, not CNCF). Ray is a general distributed-computing framework; Ray Train wraps PyTorch/FSDP distribution with a Pythonic API, and KubeRay makes a Ray cluster a native Kubernetes object. Three CRDs: **RayCluster** (a long-running cluster of a head plus workers), **RayJob** (spin up a cluster, run a job, tear it down — the batch-training pattern), and **RayService** (serving, covered in lesson 06).

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata: {name: train-fsdp}
spec:
  shutdownAfterJobFinishes: true
  entrypoint: python train_ray.py
  rayClusterSpec:
    headGroupSpec:
      template:
        spec:
          containers:
            - name: ray-head
              image: myco/ray-train:v1
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 4
        template:
          spec:
            containers:
              - name: ray-worker
                image: myco/ray-train:v1
                resources: {limits: {nvidia.com/gpu: 8}}
```

Ray handles rank assignment and rendezvous itself, so you write less scheduling YAML and more Python. KubeRay integrates with **Kueue**, **Volcano**, and NVIDIA's KAI scheduler for gang scheduling and quota. Choose KubeRay when you want programmatic control over the distributed logic, are running data-parallel plus data-processing pipelines together, or already live in the Ray ecosystem; choose Kubeflow Trainer when you want a declarative, opinionated "submit a training job" experience that a platform team standardizes.

## DDP, FSDP, and elastic training

Which distribution strategy you run shapes how you configure the job. **DDP (DistributedDataParallel)** replicates the full model on every GPU and all-reduces gradients each step — simple and fast, but it requires the whole model to fit in one GPU's memory. When it does not — a large LLM whose weights, gradients, and optimizer states blow past 80 GB — you switch to **FSDP (Fully Sharded Data Parallel)**, which shards those tensors across GPUs and gathers them just-in-time per layer. FSDP trades extra communication for the ability to train models far larger than one device, and it is the default for serious LLM fine-tuning. From Kubernetes' point of view both look identical — a set of pods running `torchrun` — so the choice lives in your training code and the runtime's parallelism config, not in the CRD. What changes at the cluster level is the *communication intensity*: FSDP gathers and re-shards parameters constantly, so it is even more sensitive to interconnect quality than DDP, which makes the NCCL and topology concerns below non-negotiable.

A third consideration is **elasticity**. `torchrun` supports elastic training, where the world size can shrink or grow as workers leave or join — valuable on Spot/preemptible nodes where a worker can vanish mid-run. Kubeflow Trainer and the legacy PyTorchJob both expose elastic policies (min/max replicas) that let a job continue on fewer workers rather than failing outright when a node is reclaimed. Combined with frequent checkpointing, elasticity is what makes running large training on cheap preemptible GPUs viable instead of reckless.

## Multi-node NCCL: where throughput is won or lost

Getting the job scheduled is half the battle; getting it *fast* is the other half. Multi-node training is bottlenecked by the all-reduce that synchronizes gradients every step, and that runs over **NCCL**. Several Kubernetes-specific realities decide whether you get near-linear scaling or a job that spends more time communicating than computing:

- **High-speed interconnect must be plumbed through.** Cloud GPU nodes offer RDMA fabrics — AWS EFA, GCP GPUDirect-TCPX/RDMA, InfiniBand on-prem. These are not automatic in a container. You attach secondary network interfaces (via Multus and the cloud's RDMA device plugin) and set the NCCL env vars (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`) so NCCL uses the fast fabric, not the slow default pod network. Skip this and your 8-node H100 job crawls.
- **Topology-aware placement.** Bandwidth within a node (NVLink) dwarfs bandwidth between nodes. A scheduler that packs a job's pods onto nodes in the same rack or with the same fast switch (Volcano and KAI do this; JobSet exposes topology hints) cuts cross-node traffic. Random placement leaves throughput on the table.
- **`sh[m]` / shared memory.** PyTorch DataLoader workers and NCCL use `/dev/shm`; the container default (64 MB) is far too small and causes cryptic hangs. Mount a large `emptyDir` with `medium: Memory` at `/dev/shm`.
- **Debugging.** Set `NCCL_DEBUG=INFO` and read the logs — NCCL prints exactly which transport (NVLink, IB, socket) it selected per pair. If it says "socket" where you expected "IB," your fast fabric is not wired up.

```yaml
      containers:
        - name: worker
          env:
            - {name: NCCL_DEBUG, value: "INFO"}
            - {name: NCCL_SOCKET_IFNAME, value: "eth0"}
          volumeMounts:
            - {name: dshm, mountPath: /dev/shm}
      volumes:
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 16Gi}
```

## Cost and operations notes

Multi-node GPU training is the single most expensive thing most ML teams run, so a few operational habits pay for themselves. **Checkpoint to durable storage frequently** (lesson 07) — a preemption or node failure mid-run should cost minutes, not days, especially on Spot nodes. **Gang scheduling is mandatory**, not optional: without it, a partially-scheduled job burns the GPUs it holds while making zero progress. And measure **scaling efficiency**: if going from 4 to 8 nodes gives you 1.4x instead of ~2x speedup, your interconnect or topology is the culprit, and the fix is networking, not more GPUs.

## Key takeaways

- Distributed PyTorch (DDP/FSDP) needs stable per-worker hostnames, gang scheduling, and correct rank/world-size env vars. Kubeflow Trainer and KubeRay exist to generate this plumbing so you do not hand-write it.
- **Kubeflow Trainer** (2025 successor to the Training Operator) uses one unified **TrainJob** CRD replacing `PyTorchJob`/`TFJob`/`MPIJob`; it splits reusable `ClusterTrainingRuntime` (platform) from lightweight `TrainJob` (user), builds on **JobSet**, and integrates with Kueue. The v2 API is **alpha** — pin versions. Legacy `PyTorchJob` still runs in many clusters.
- **KubeRay** (RayCluster/RayJob/RayService) is the Ray-native path — more Python, less YAML, good for mixed compute+training pipelines; integrates with Kueue/Volcano/KAI.
- **NCCL over the right fabric** decides throughput: plumb RDMA/EFA/InfiniBand through with the right device plugins and `NCCL_*` env vars, use **topology-aware** placement, and mount a large `/dev/shm`. `NCCL_DEBUG=INFO` tells you which transport was chosen.
- Multi-node training is the most expensive workload you run: checkpoint often (survive Spot preemption), require gang scheduling, and watch scaling efficiency to catch networking bottlenecks.

## Try it

1. Install Kubeflow Trainer, apply a small 2-node `TrainJob` running a toy DDP script, and confirm from the logs that all ranks join and the world size matches `numNodes * numProcPerNode`.
2. Break gang scheduling on purpose: request more GPUs than the cluster has for a 4-worker job and watch it hang partially scheduled. Then route it through Kueue/Volcano and confirm all-or-nothing admission.
3. Set `NCCL_DEBUG=INFO` on a multi-node run and read which transport NCCL selects between pods — identify whether it is using the fast fabric or falling back to sockets.
4. Reproduce the `/dev/shm` failure: run a DataLoader with many workers without the tmpfs mount, watch it hang or error, then add the `emptyDir: {medium: Memory}` mount and confirm it fixes it.
5. Convert the same training to a KubeRay `RayJob` with `shutdownAfterJobFinishes: true` and compare the developer experience against the TrainJob version.
