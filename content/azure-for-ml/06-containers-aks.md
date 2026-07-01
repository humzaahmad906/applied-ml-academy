# 06 — Containers: ACR, ACI, AKS

Containers are how modern ML environments become reproducible and portable. The version-matching nightmare — CUDA against driver against framework against Python — disappears when you bake everything into one image that runs identically on your laptop, on the training cluster, and behind the serving endpoint. Azure's container story has three services that fit different points on a spectrum from "run one container quickly" to "run a fleet of GPU inference pods with autoscaling": **Azure Container Registry** stores images, **Azure Container Instances** runs single containers on demand, and **Azure Kubernetes Service** orchestrates production-scale clusters. In the end-to-end solution, containers are the packaging layer under training environments and model serving alike.

## Azure Container Registry (ACR)

**Azure Container Registry** is a private, managed Docker/OCI registry. It holds the images your training jobs and endpoints run — both the environment images you build and the ones Azure Machine Learning builds for you when you define an environment. It comes in three tiers: **Basic** (dev), **Standard** (most production), and **Premium**, which adds geo-replication, private endpoints, and content trust for larger or regulated deployments.

```bash
# Create a registry and build an image inside Azure (no local Docker needed)
az acr create -n acrmlx -g rg-mlx-dev --sku Standard --admin-enabled false
az acr build -r acrmlx -t serving/fraud-model:1.0 -f Dockerfile .
```

`az acr build` runs the build on Azure using **ACR Tasks**, which is convenient on machines without a local Docker daemon and lets you set up automated rebuilds when a base image gets a security patch. Authenticate with **managed identity**, not the admin user (keep `--admin-enabled false`): grant consuming resources the **AcrPull** role and image-pushing pipelines **AcrPush**. This keeps registry credentials out of your code entirely.

An ML-specific detail: Azure Machine Learning environments are backed by ACR. When you define an environment from a base image plus a conda/pip spec, the workspace materializes it as an image in the workspace's attached registry, caches it, and reuses it across jobs so you are not rebuilding dependencies every run.

## Azure Container Instances (ACI)

**Azure Container Instances** runs a single container (or a small group) without any cluster to manage — you hand it an image and it runs, billed per second while active. It is the fastest path from image to running workload. For ML it fits a few narrow but useful cases: a quick smoke test of a serving container, a lightweight always-on preprocessing service, or an event-driven batch task kicked off by another service. It supports limited GPU scenarios but is not the place for high-throughput GPU inference.

```bash
az container create -g rg-mlx-dev --name aci-smoketest \
  --image acrmlx.azurecr.io/serving/fraud-model:1.0 \
  --cpu 2 --memory 4 --ports 8080 \
  --acr-identity id-mlplatform --assign-identity id-mlplatform
```

The honest guidance: ACI is for **quick, transient, single-container** needs. For production real-time serving with autoscaling, health probes, and rolling updates, use Azure Machine Learning **managed online endpoints** (covered later) or AKS. Managed online endpoints are the right default for most model serving because they give you the serving features without operating Kubernetes; reach for AKS when you need cluster-level control.

## Azure Kubernetes Service (AKS)

**Azure Kubernetes Service** is managed Kubernetes: Azure runs the control plane, you run **node pools** (each a scale set of workers). AKS is the heavy machinery for production ML serving when you need fine-grained control — custom autoscaling on GPU or queue-depth metrics, multi-model routing, sidecars, service mesh, or colocating inference with other microservices. The cost of that power is that you now operate Kubernetes.

The ML-relevant building blocks:

- **GPU node pools.** Create a node pool on a GPU SKU (NC-series for inference) so GPU-bound pods land on GPU nodes. The NVIDIA device plugin exposes GPUs to the scheduler; AKS can install and manage the GPU drivers for you.
- **Cluster autoscaler / node autoprovisioning.** Scales node counts with demand, including scaling GPU pools down when idle so you stop paying for expensive accelerators between bursts.
- **Horizontal Pod Autoscaler / KEDA.** Scales pod replicas on CPU, GPU, or event-source metrics (queue length, request rate) — KEDA is the event-driven autoscaler that lets serving scale on the depth of an ingestion queue rather than raw CPU.

```bash
# System pool (CPU) plus a scale-to-zero GPU pool for inference
az aks create -g rg-mlx-dev -n aks-mlx \
  --node-count 2 --node-vm-size Standard_D4s_v5 \
  --enable-managed-identity --network-plugin azure --generate-ssh-keys

az aks nodepool add -g rg-mlx-dev --cluster-name aks-mlx \
  --name gpupool --node-vm-size Standard_NC24ads_A100_v4 \
  --node-count 0 --min-count 0 --max-count 4 --enable-cluster-autoscaler \
  --node-taints sku=gpu:NoSchedule   # keep non-GPU pods off pricey nodes
```

The taint plus a matching toleration on your serving pods ensures only GPU workloads schedule onto GPU nodes, and `--min-count 0` lets the pool deallocate entirely when no inference is running.

## KAITO: LLM serving on AKS the easy way

Serving large language models on Kubernetes used to mean hand-writing GPU node specs, driver setup, model download logic, and inference-server configuration. The **Kubernetes AI Toolchain Operator (KAITO)** — available as a GA add-on on AKS — collapses that into a declarative resource. You describe the model you want, and KAITO's node provisioner automatically provisions appropriately sized GPU nodes, downloads the model, and stands up an inference service (integrating with **vLLM** for high-throughput serving) behind an OpenAI-compatible API.

```bash
# Enable the AI toolchain operator (KAITO) add-on on an existing cluster
az aks update -g rg-mlx-dev -n aks-mlx \
  --enable-ai-toolchain-operator --enable-oidc-issuer
```

```yaml
# A KAITO workspace declaratively serves a model; it provisions the GPU nodes
apiVersion: kaito.sh/v1beta1
kind: Workspace
metadata:
  name: workspace-llm
resource:
  instanceType: "Standard_NC24ads_A100_v4"
  labelSelector:
    matchLabels: { apps: llm-inference }
inference:
  preset:
    name: "llama-3.1-8b-instruct"
```

KAITO also arranges fast model loading — provisioning NVMe-backed storage on each GPU node and placing model weights there so cold starts are quick. For teams that want to self-host open-weight models on their own AKS cluster (for data-residency or cost reasons) rather than call a hosted API, KAITO is the path of least resistance.

## Choosing among ACR, ACI, AKS — and managed endpoints

- **ACR** is not optional — it is where every image lives; use it regardless of how you run containers.
- **ACI** for quick, transient, single-container tasks: smoke tests, light preprocessing, event-triggered jobs.
- **Managed online endpoints** (Azure Machine Learning) for most production real-time serving — you get autoscaling, blue-green rollout, and monitoring without operating a cluster.
- **AKS** when you need cluster-level control, custom autoscaling, self-hosted LLMs via KAITO, or colocation with other microservices — and you have the ops capacity to run Kubernetes.

## How containers fit the whole solution

Containers thread through the entire architecture. Your **training environment** is a container image in ACR, reused across every job for reproducibility. Your **serving artifact** is a container image (or a curated inference image the endpoint builds) pulled with AcrPull by managed online endpoints or AKS pods. CI/CD (a later topic) builds and pushes these images to ACR on every merge, and deployments reference immutable image tags so you always know exactly what code produced a prediction. If your solution includes self-hosted GenAI, KAITO-on-AKS serves open-weight models from the same registry. The registry is the pipeline's handoff point: build once, run anywhere in the system.

## Key takeaways

- **Containers make ML environments reproducible** end to end — the same image trains, evaluates, and serves.
- **ACR** stores all images; authenticate with **managed identity + AcrPull/AcrPush**, keep the admin user off, and use `az acr build` to build in-cloud.
- **ACI** is for quick single-container tasks; it is not production serving.
- **AKS** gives cluster-grade control (GPU node pools, cluster autoscaler, KEDA, taints to protect GPU nodes) at the cost of operating Kubernetes; **KAITO** makes self-hosting LLMs on AKS declarative.
- For most real-time serving, prefer **managed online endpoints** over hand-rolled AKS — reserve AKS for when you genuinely need its control.

## Try it

Create a Standard ACR with the admin user disabled. Build a small serving image with `az acr build` from a simple `Dockerfile` (Python + your model-loading code). Grant your managed identity `AcrPull`, then run the image once on ACI as a smoke test and hit its endpoint. As a stretch goal, create an AKS cluster with a scale-to-zero GPU node pool (`--min-count 0`) plus a GPU taint, and confirm the pool shows zero nodes while idle — the same cost discipline you applied to training clusters, now for serving.
