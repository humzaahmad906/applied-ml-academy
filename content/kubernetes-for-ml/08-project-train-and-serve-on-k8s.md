# 08 — Project: Train and Serve on Kubernetes

You now have every piece: the primitives, GPUs, scheduling and autoscaling, batch queues, distributed training, serving, and storage plus observability. This capstone wires them into one end-to-end system on a real GPU cluster — the same shape a platform team runs in production. The goal: fine-tune a model as a queued, gang-scheduled batch job that reads its dataset from object storage and checkpoints back to it; register the resulting weights; serve them behind an autoscaling, canary-capable endpoint; and watch the whole thing through a GPU-aware dashboard. This is not eight disconnected features — it is one pipeline, and the value is in how the pieces hand off to each other.

## The architecture

The system has four planes, each drawing on an earlier lesson:

- **Cluster foundation** — a GKE/EKS/AKS cluster with a cheap CPU node pool and two GPU node pools: an H100 pool for training (scales to zero) and an L4 pool for serving (warm floor of one). GPU Operator installed; DCGM exporting metrics. *(Lessons 01, 02, 07.)*
- **Batch/training plane** — Kueue for quota and admission, a Kubeflow **TrainJob** for the fine-tune, reading data via an object-storage CSI mount and writing checkpoints back. *(Lessons 04, 05, 07.)*
- **Serving plane** — a KServe `InferenceService` (or vLLM runtime for an LLM) on the L4 pool, autoscaling on queue depth, with canary rollout. *(Lessons 03, 06.)*
- **Observability plane** — Prometheus + Grafana + DCGM, one dashboard showing GPU utilization, training progress, and serving latency/queue depth. *(Lesson 07.)*

The thread connecting them is **object storage as the handoff**: training reads the dataset from it and writes the checkpoint to it; serving reads that checkpoint from it. The bucket is the seam between the two halves of the ML lifecycle, and keeping the seam in durable object storage is what makes each half independently restartable.

## Step 1 — Cluster and namespace foundation

Create the cluster with the three node pools and install the GPU stack. Then carve a namespace for the project with a quota, so this project cannot starve others on a shared cluster.

```bash
# Two GPU pools: training scales to zero, serving keeps a warm floor
gcloud container node-pools create h100-train --cluster=ml-cluster --region=us-central1 \
  --machine-type=a3-highgpu-8g --accelerator=type=nvidia-h100-80gb,count=8 \
  --enable-autoscaling --min-nodes=0 --max-nodes=2 \
  --node-taints=nvidia.com/gpu=present:NoSchedule
gcloud container node-pools create l4-serve --cluster=ml-cluster --region=us-central1 \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --enable-autoscaling --min-nodes=1 --max-nodes=6 \
  --node-taints=nvidia.com/gpu=present:NoSchedule

helm install --wait gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace
kubectl create namespace mlp
```

```yaml
apiVersion: v1
kind: ResourceQuota
metadata: {name: mlp-quota, namespace: mlp}
spec:
  hard:
    requests.nvidia.com/gpu: "8"     # this project caps at 8 GPUs
```

## Step 2 — Data and secrets

Put the dataset in a bucket (the source of truth) and expose the token the training job needs to pull the base model. Prefer workload identity so no key is stored; use a Secret only as a fallback.

```bash
gsutil -m cp -r ./dataset gs://myco-training-data/sft/
kubectl create secret generic hf-creds -n mlp --from-literal=token="$HF_TOKEN"
```

The training pod will mount `gs://myco-training-data` via the GCS FUSE CSI driver rather than copying it — the lesson 07 pattern — and write checkpoints to `gs://myco-models`.

## Step 3 — Queued, gang-scheduled fine-tune

Set up a Kueue ClusterQueue for the project, then submit the fine-tune as a Kubeflow TrainJob. Kueue suspends it until the 8-GPU quota is free, then admits it as a gang; the H100 pool scales up from zero to run it and back down when done.

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata: {name: mlp}
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: ["nvidia.com/gpu"]
      flavors:
        - name: h100
          resources: [{name: "nvidia.com/gpu", nominalQuota: 8}]
---
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: sft-run
  namespace: mlp
  labels: {kueue.x-k8s.io/queue-name: mlp-local}
spec:
  runtimeRef: {name: torch-distributed, kind: ClusterTrainingRuntime}
  trainer:
    numNodes: 1
    numProcPerNode: "8"                       # 8-GPU single-node DDP
    resourcesPerNode: {limits: {nvidia.com/gpu: 8}}
    command: ["torchrun", "sft.py",
              "--data=/data/sft", "--out=gs://myco-models/sft/v1"]
    env:
      - {name: HF_TOKEN, valueFrom: {secretKeyRef: {name: hf-creds, key: token}}}
      - {name: NCCL_DEBUG, value: "INFO"}
```

```bash
kubectl apply -f queue.yaml -f trainjob.yaml
kubectl get workloads -n mlp                  # watch Kueue admit it
kubectl get nodes -w                          # watch the H100 pool scale from zero
kubectl logs -f -l trainer.kubeflow.org/trainjob-name=sft-run -n mlp
```

Confirm the run **checkpoints to `gs://myco-models` periodically** — this is what lets a Spot preemption or node failure cost minutes, not the whole run. When it finishes, the TrainJob completes, the H100 pool drains to zero, and you stop paying for training GPUs.

## Step 4 — Serve the trained model

Deploy the checkpoint as a KServe `InferenceService` on the L4 serving pool, autoscaling on request load with a warm floor of one (never zero for a GPU model — the cold start is too slow, per lesson 06).

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata: {name: sft-serve, namespace: mlp}
spec:
  predictor:
    minReplicas: 1                            # warm floor: never cold-start from zero
    maxReplicas: 6
    model:
      modelFormat: {name: pytorch}
      storageUri: gs://myco-models/sft/v1
      resources: {limits: {nvidia.com/gpu: 1}}
```

```bash
kubectl apply -f serve.yaml
kubectl get inferenceservice sft-serve -n mlp   # wait for READY=True
# hit the endpoint
curl -s "$(kubectl get isvc sft-serve -n mlp -o jsonpath='{.status.url}')/v1/models/sft-serve:predict" \
  -d @sample.json
```

Startup and readiness probes (managed by KServe) ensure traffic waits until the weights are loaded. Load-test it and watch replicas scale on the serving pool.

## Step 5 — Canary a new version

The real test of a serving platform is shipping v2 safely. Train an improved checkpoint to `gs://myco-models/sft/v2`, then canary 10% of traffic to it and compare live metrics before promoting.

```yaml
spec:
  predictor:
    canaryTrafficPercent: 10
    model: {storageUri: gs://myco-models/sft/v2}
```

Send traffic, confirm ~10% hits v2, compare latency and quality against v1, then either promote to 100% (remove the canary field) or roll back instantly (set it to 0). This is the lesson 06 pattern applied to the model you just trained.

## Step 6 — Observe the whole thing

Open Grafana and build the one dashboard that matters: **GPU utilization across both pools**, **training throughput** (from the TrainJob), and **serving p99 latency and queue depth** (from vLLM/KServe metrics), all from Prometheus + DCGM. This closes the loop — you can now see that the H100 pool sat at 90% during training then went to zero, that the L4 serving pool tracks request load, and whether any GPU is showing ECC errors. If serving GPU-util is low while latency is high, that is your signal to scale on queue depth, not utilization (lesson 03). If the training pool sat at 40%, your data pipeline (lesson 07) is starving the GPUs.

## Reading the cost of what you ran

It is worth making the economics concrete, because the whole architecture exists to control them. Suppose the fine-tune takes 3 hours on one 8×H100 node. An 8×H100 node runs roughly on the order of $25-30/hour on-demand (varies by cloud and region), so the training run costs on the order of $75-90 — but only because the pool **scaled to zero** before and after. Had you left that node running continuously "so it's ready," you would pay that hourly rate around the clock, roughly $18-20k/month, to have it sit idle between runs. The scale-to-zero training pool is the difference between paying for 3 hours and paying for a month. *(These figures are order-of-magnitude illustrations, not quotes — check current pricing for your cloud and region.)*

The serving side has the opposite shape. A single L4 runs on the order of $0.70-1/hour, and you keep a warm floor of one because a cold start — provisioning a node, pulling the image, loading weights onto the GPU — would add tens of seconds to a user's first request. Here you are deliberately paying for idle capacity to buy latency, and the numbers justify it: the warm L4 floor costs a few hundred dollars a month, trivial next to the cost of a bad user experience. The lesson of the capstone is that *training and serving want opposite cost postures* — training scales to zero and tolerates cold starts, serving stays warm and cannot — and Kubernetes lets you express both on the same cluster with two node pools and two autoscaling policies.

## What you built

You built the exact pipeline a production ML platform runs: a fair-shared, quota-governed cluster where training jobs queue and gang-schedule onto scale-to-zero GPU pools, read data from object storage and checkpoint back to it, then serve the result behind an autoscaling, canary-capable endpoint — all observable through GPU-aware dashboards. Every design choice traces to a cost or reliability reason: scale-to-zero training pools so idle H100s cost nothing, warm serving floors so users never hit a cold start, queues so no team starves another, checkpoints so preemptions are cheap, and DCGM so you can see the utilization that justifies all of it.

## Key takeaways

- A production ML platform on Kubernetes is one pipeline with four planes — foundation, batch/training, serving, observability — stitched together by **object storage as the handoff** between training output and serving input.
- Training runs as a **queued, gang-scheduled TrainJob** on a **scale-to-zero GPU pool**, reading data via CSI mount and **checkpointing to object storage** so preemptions and failures are cheap.
- Serving runs on a separate pool with a **warm floor of one** (never cold-start a GPU model), autoscales on **queue depth**, and ships new versions via **canary + instant rollback**.
- The **GPU-utilization-vs-cost dashboard** (Prometheus + DCGM) closes the loop: it proves training pools drain to zero, serving tracks load, and surfaces the idle-GPU waste and hardware errors that drive every optimization.
- Every choice is a cost or reliability decision — the architecture is not features stacked up, it is trade-offs made explicit.

## Try it

1. Stand up the cluster foundation (three node pools, GPU Operator, namespace + quota) and confirm `kubectl get nodes` shows the pools and the H100 pool at zero nodes.
2. Run the full fine-tune as a queued TrainJob; watch Kueue admit it, the H100 pool scale up, checkpoints appear in the bucket, and the pool drain back to zero when done.
3. Serve the checkpoint via KServe with a warm floor, hit the endpoint, then load-test it and confirm the serving pool autoscales.
4. Train a v2, canary 10% of traffic to it, compare metrics, and practice both a promotion and a rollback.
5. Build the GPU-utilization-vs-cost Grafana dashboard and use it to answer: did the training pool stay busy, does serving track load, and is any GPU idle or erroring? Write down one optimization the dashboard reveals.
