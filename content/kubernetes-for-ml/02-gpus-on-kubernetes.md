# 02 — GPUs on Kubernetes

Kubernetes does not know what a GPU is out of the box. To the vanilla scheduler, a node has CPU and memory, and that is it — GPUs are invisible. Making expensive accelerators schedulable, isolable, and shareable is the first real problem an ML platform team solves, and it is where a lot of clusters get stuck. This lesson covers how GPUs become a first-class schedulable resource (the NVIDIA device plugin and, increasingly, the GPU Operator), how you request them in a pod spec, how to slice one physical GPU into many logical ones with MIG and time-slicing so you stop wasting them, how node pools keep GPUs off your cheap nodes, and where the whole framework is heading with Dynamic Resource Allocation.

## Making GPUs visible: device plugin vs GPU Operator

Two pieces have to be in place before a pod can use a GPU. The node needs the **NVIDIA driver** and a container runtime configured to inject the GPU into containers (the NVIDIA Container Toolkit). And Kubernetes needs the **device plugin**, a DaemonSet that discovers the GPUs on each node and advertises them to the kubelet as an extended resource named `nvidia.com/gpu`. Once advertised, the scheduler can treat GPUs like CPU or memory — as a countable resource pods request.

You can install these by hand, but almost nobody does anymore. The **NVIDIA GPU Operator** (GA, actively developed) automates the entire stack — driver, container toolkit, device plugin, DCGM metrics exporter, and GPU Feature Discovery — through a single `ClusterPolicy` custom resource. One Helm install replaces a fragile manual sequence of driver installs and DaemonSet configs, and it keeps everything version-matched as nodes come and go.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install --wait gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace
```

On the managed clouds this is often even simpler: GKE, EKS, and AKS can install the GPU drivers for you when you create a GPU node pool, and GKE offers the operator or its own device plugin depending on configuration. The GPU Operator still deploys the classic device plugin under the hood; it does not replace it, it manages it. After it settles, confirm your nodes now advertise GPUs:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

## Requesting a GPU in a pod spec

A GPU is requested exactly like CPU or memory, under `resources.limits`. There is one crucial quirk: GPUs are an **integer, non-oversubscribable** resource. You cannot request a fractional GPU (`0.5`) the way you can with CPU, and you set the count under `limits` (Kubernetes treats the request as equal to the limit for extended resources). If no node has a free GPU, the pod sits `Pending` until one appears.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-train
spec:
  restartPolicy: Never
  containers:
    - name: trainer
      image: us-central1-docker.pkg.dev/myco/ml-images/trainer:v1
      command: ["python", "train.py"]
      resources:
        limits:
          nvidia.com/gpu: 1        # request one whole GPU
          cpu: "8"
          memory: "48Gi"
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-h100-80gb   # pin to the GPU type you want
```

The `nodeSelector` (or a more expressive `nodeAffinity`) matters more than beginners expect. Without it, a pod asking for `nvidia.com/gpu: 1` will accept *any* GPU node — an L4 or an H100 alike — and you may land expensive training on the wrong silicon. Label keys differ by cloud (`cloud.google.com/gke-accelerator`, `nvidia.com/gpu.product` from the GPU Operator's feature discovery, or your own labels), so check what your nodes actually expose with `kubectl get nodes --show-labels`.

## Taints and tolerations: keeping non-GPU pods off GPU nodes

A GPU node is expensive. You do not want a logging DaemonSet or a random CPU-only pod scheduled onto it, occupying CPU and blocking a GPU job from fitting. The standard pattern is to **taint** GPU nodes so nothing lands there unless it explicitly **tolerates** the taint. Managed clouds apply a taint like `nvidia.com/gpu=present:NoSchedule` automatically; you add the matching toleration to GPU pods.

```yaml
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

Taints (repel pods) plus node selectors (attract pods) together give you clean separation: GPU workloads land on GPU nodes, everything else stays on cheap CPU nodes.

## Sharing one GPU: MIG and time-slicing

A whole H100 assigned to a pod that runs a tiny model at 5% utilization is money set on fire. Two mechanisms let multiple workloads share one physical GPU, and they solve different problems.

**MIG (Multi-Instance GPU)** is hardware partitioning, available on A100, A30, H100, H200, and Blackwell-class (B200) GPUs. It carves one GPU into up to seven instances, each with its own isolated slice of memory, compute, and cache. Because the isolation is in hardware, one instance cannot interfere with another's memory or crash it — this is the right choice for multi-tenant clusters where you must guarantee isolation. You configure MIG profiles through the GPU Operator, and each instance is then advertised as a schedulable resource:

```yaml
      resources:
        limits:
          nvidia.com/mig-1g.10gb: 1   # one MIG slice: 1 compute unit, 10 GB
```

**Time-slicing** is software sharing: the device plugin advertises N logical replicas of one physical GPU, and the GPU time-shares among the pods that land on it. There is **no memory or fault isolation** — pods can starve or OOM each other — but it works on any GPU, including older ones without MIG support, and it is ideal for bursty, low-utilization, trusted workloads like development notebooks or lightweight inference. You enable it with a ConfigMap consumed by the GPU Operator:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4          # advertise each physical GPU as 4 slots
```

The decision is a cost-versus-isolation trade-off. MIG gives strong isolation at fixed partition sizes and only on newer hardware; time-slicing gives maximum flexibility and hardware coverage but zero isolation. You can even combine them — time-slice within a MIG instance for very fine-grained sharing. For a shared research cluster, a common pattern is MIG on the big GPUs for tenant isolation and time-slicing on a pool of L4s for cheap notebook workloads.

## Node pools: organizing the fleet

You rarely want a single flat set of identical nodes. Real ML clusters have **node pools** (called node groups on EKS): a cheap CPU pool for data prep and web services, an L4 pool for inference, an H100 pool for training, and often a **Spot/preemptible** pool for cost-tolerant batch work at a steep discount. Each pool is a separately scaled, separately labeled group of machines.

```bash
# GKE: an autoscaling H100 pool that can scale to zero when idle
gcloud container node-pools create h100-train \
  --cluster=ml-cluster --region=us-central1 \
  --machine-type=a3-highgpu-8g \
  --accelerator=type=nvidia-h100-80gb,count=8 \
  --enable-autoscaling --min-nodes=0 --max-nodes=4 \
  --node-taints=nvidia.com/gpu=present:NoSchedule
```

The `--min-nodes=0` is the single most important cost lever on a GPU cluster: it lets the pool **scale to zero** when no GPU pod is pending, so you pay nothing for idle accelerators, and the cluster autoscaler spins nodes back up when a GPU job arrives. The trade-off is cold-start latency — provisioning an 8×H100 node takes minutes — which is fine for training but usually not for latency-sensitive serving, where you keep a warm minimum. Lesson 03 covers autoscaling in full.

## Where this is heading: Dynamic Resource Allocation

The device-plugin model — count GPUs, request an integer — is showing its age, because a GPU is not just a count: it has a product name, a memory size, a MIG profile, a driver version, and an interconnect topology. **Dynamic Resource Allocation (DRA)** is the successor framework, and it reached GA in a recent Kubernetes release. DRA lets a pod request a device by *attributes* — "give me a GPU with at least 40 GB and NVLink to its neighbor" — rather than a bare count. NVIDIA has a DRA driver for its GPUs, and the newest GPU Operator releases require the classic device plugin to be disabled when the DRA driver is in use, which signals the direction of travel.

For now the practical guidance is: the device plugin and `nvidia.com/gpu` counts are what you will use on most clusters today, and every managed cloud still supports them. DRA is the future and worth watching, but treat its exact APIs as still-settling — pin versions and read your provider's docs before adopting it in production. *(GA-in-1.34 and the specific operator version gates are fast-moving; verify against current release notes for your cluster version.)*

## Key takeaways

- GPUs are invisible to Kubernetes until the **NVIDIA device plugin** advertises them as `nvidia.com/gpu`; the **GPU Operator** (GA) automates the whole stack — driver, toolkit, plugin, DCGM metrics — via one `ClusterPolicy`.
- Request GPUs as an **integer under `resources.limits`** (no fractions); always add a `nodeSelector`/affinity to pin the *right* GPU type, and a **toleration** so the pod can land on tainted GPU nodes.
- **MIG** is hardware partitioning with true memory/fault isolation (A100/H100/H200/Blackwell); **time-slicing** is software sharing with no isolation but works on any GPU. Choose MIG for multi-tenant isolation, time-slicing for cheap bursty/notebook workloads.
- Organize GPUs into **node pools** (CPU / L4 / H100 / Spot), taint them, and set `--min-nodes=0` so idle GPU pools **scale to zero** — the biggest cost lever on a GPU cluster.
- **DRA** is the emerging successor to the device plugin (attribute-based GPU requests, GA in recent Kubernetes); use device-plugin counts today, watch DRA, and pin versions since its APIs are still settling.

## Try it

1. On a cluster with at least one GPU node, run the `kubectl get nodes -o custom-columns=...nvidia\.com/gpu` command to confirm GPUs are advertised. If they are not, the device plugin or GPU Operator is not installed.
2. Submit the `gpu-train` pod above (swap in a container that runs `nvidia-smi`), then `kubectl logs gpu-train` to confirm the GPU is visible inside the container.
3. Request `nvidia.com/gpu: 8` on a cluster that only has single-GPU nodes and watch the pod sit `Pending`; run `kubectl describe pod` and read the "insufficient nvidia.com/gpu" event.
4. Enable time-slicing with a ConfigMap advertising `replicas: 2`, then schedule two GPU pods onto a single physical GPU and confirm both run (and note there is no memory isolation between them).
5. Create an autoscaling GPU node pool with `--min-nodes=0`, submit one GPU job, watch a node appear; delete the job and watch the pool scale back to zero.
