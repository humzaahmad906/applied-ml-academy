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

Day-to-day, you spend most of your ACR time inspecting and moving images rather than creating registries. `az acr login` exchanges your Azure token for a short-lived registry credential so a local `docker push`/`pull` works without the admin user; `az acr repository list` and `az acr repository show-tags` are how you see what is actually stored and which tag a deployment should pin. Pulling public base images through your own registry (rather than straight from Docker Hub, which rate-limits and can vanish) is worth the habit — `az acr import` copies an image in server-side, so your builds no longer depend on an upstream registry being available:

```bash
az acr login -n acrmlx                                   # token -> docker credential, no admin user
az acr repository list -n acrmlx -o table                # what images exist
az acr repository show-tags -n acrmlx --repository serving/fraud-model \
  --orderby time_desc --top 10 -o table                  # newest tags first
az acr repository show -n acrmlx --repository serving/fraud-model  # size, last update, manifest count

# Import a public base image once so builds don't depend on Docker Hub uptime / rate limits
az acr import -n acrmlx --source docker.io/library/python:3.12-slim --image base/python:3.12-slim

# Run a one-off command in the cloud, or wire an automated rebuild on base-image patch / git push
az acr run -r acrmlx --cmd 'acr build -t serving/fraud-model:{{.Run.ID}} .' /dev/null
az acr task create -n rebuild-serving -r acrmlx \
  --image serving/fraud-model:{{.Run.ID}} --context https://github.com/org/serving.git \
  --file Dockerfile --git-access-token $PAT --base-image-trigger-enabled true
```

An `az acr task` with `--base-image-trigger-enabled true` is the piece most teams skip: it rebuilds and re-pushes your serving image automatically whenever the base image it depends on is patched, so a CVE fix in `python:3.12-slim` propagates to your fleet without a human remembering to rebuild.

**Premium tier** unlocks the features regulated or multi-region ML platforms need: geo-replication (so a pull in a second region is local, not cross-continent), private endpoints (so the registry is only reachable inside your VNet), customer-managed keys, and content trust. Replication and private-endpoint wiring are Premium-only, so budget for the tier if the platform is more than single-region dev:

```bash
az acr update -n acrmlx --sku Premium                    # upgrade in place; no image loss
az acr replication create -r acrmlx -l westus2           # pulls in westus2 now served locally
az acr private-endpoint-connection list -r acrmlx -o table   # inspect PE connections (Premium)
```

An ML-specific detail: Azure Machine Learning environments are backed by ACR. When you define an environment from a base image plus a conda/pip spec, the workspace materializes it as an image in the workspace's attached registry, caches it, and reuses it across jobs so you are not rebuilding dependencies every run.

## Azure Container Instances (ACI)

**Azure Container Instances** runs a single container (or a small group) without any cluster to manage — you hand it an image and it runs, billed per second while active. It is the fastest path from image to running workload. For ML it fits a few narrow but useful cases: a quick smoke test of a serving container, a lightweight always-on preprocessing service, or an event-driven batch task kicked off by another service. It supports limited GPU scenarios but is not the place for high-throughput GPU inference.

```bash
az container create -g rg-mlx-dev --name aci-smoketest \
  --image acrmlx.azurecr.io/serving/fraud-model:1.0 \
  --cpu 2 --memory 4 --ports 8080 \
  --acr-identity id-mlplatform --assign-identity id-mlplatform \
  --restart-policy Never --environment-variables MODEL_VERSION=1.0
```

Note the identity flags: `--assign-identity` gives the container group the managed identity, and `--acr-identity` tells ACI to use that same identity to *pull* the image — so you never pass a registry password. `--restart-policy Never` suits a batch/smoke-test container that should run once and stop (the default `Always` keeps a long-running service alive). Once it is running, the operational loop is logs, exec, show, delete:

```bash
az container logs   -g rg-mlx-dev --name aci-smoketest             # stdout/stderr
az container logs   -g rg-mlx-dev --name aci-smoketest --follow    # stream live
az container exec   -g rg-mlx-dev --name aci-smoketest --exec-command "/bin/bash"  # shell in
az container show   -g rg-mlx-dev --name aci-smoketest \
  --query "{state:instanceView.state, ip:ipAddress.ip}" -o table   # state + assigned IP
az container delete -g rg-mlx-dev --name aci-smoketest --yes       # bill stops on delete
```

Because ACI bills per second while a group exists, the `delete` is the cost-control step — a forgotten always-on container quietly accrues charges. `--restart-policy OnFailure` is the middle ground for an event-driven batch task that should retry on a crash but not loop forever.

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

Before you can run `kubectl` against the cluster you must merge its credentials into your kubeconfig — this is the step people forget and then wonder why `kubectl` targets the wrong context:

```bash
az aks get-credentials -g rg-mlx-dev -n aks-mlx        # merges into ~/.kube/config
kubectl get nodes -o wide                              # confirm nodes + versions
kubectl get nodes -L agentpool                         # see which node belongs to which pool
kubectl describe node <gpu-node> | grep nvidia.com/gpu # confirm GPUs are advertised to the scheduler
kubectl get pods -A                                    # everything running cluster-wide
```

By default AKS installs and manages the NVIDIA driver on GPU node pools for you. If you run the NVIDIA GPU Operator instead (to control driver versions yourself), create the pool with `--gpu-driver none` so AKS does *not* also install a driver — note the older `--skip-gpu-driver-install` tag was retired in August 2025, so `--gpu-driver none` is now the correct flag. The common node-pool lifecycle operations — listing, scaling, upgrading the Kubernetes version, and toggling the autoscaler — are all `az aks nodepool` verbs:

```bash
az aks nodepool list -g rg-mlx-dev --cluster-name aks-mlx -o table
az aks nodepool add  -g rg-mlx-dev --cluster-name aks-mlx --name gpunodriver \
  --node-vm-size Standard_NC24ads_A100_v4 --gpu-driver none   # you'll run the GPU Operator
az aks nodepool scale   -g rg-mlx-dev --cluster-name aks-mlx --name gpupool --node-count 2
az aks nodepool update  -g rg-mlx-dev --cluster-name aks-mlx --name gpupool \
  --update-cluster-autoscaler --min-count 0 --max-count 8      # widen autoscaler bounds
az aks nodepool upgrade -g rg-mlx-dev --cluster-name aks-mlx --name gpupool --kubernetes-version 1.31.1

# Control-plane / cluster-wide operations
az aks get-versions -l eastus2 -o table                        # upgrade targets available in region
az aks upgrade -g rg-mlx-dev -n aks-mlx --kubernetes-version 1.31.1
az aks scale   -g rg-mlx-dev -n aks-mlx --node-count 3 --nodepool-name nodepool1
```

Scaling and autoscaler bounds are per node pool, not per cluster — `az aks nodepool scale` sets a fixed count, while `--update-cluster-autoscaler` (re)configures the min/max range that the autoscaler moves within.

## KAITO: LLM serving on AKS the easy way

Serving large language models on Kubernetes used to mean hand-writing GPU node specs, driver setup, model download logic, and inference-server configuration. The **Kubernetes AI Toolchain Operator (KAITO)** — available as a GA add-on on AKS — collapses that into a declarative resource. You describe the model you want, and KAITO's node provisioner automatically provisions appropriately sized GPU nodes, downloads the model, and stands up an inference service (integrating with **vLLM** for high-throughput serving) behind an OpenAI-compatible API.

```bash
# Enable the AI toolchain operator (KAITO) add-on on an existing cluster
az aks update -g rg-mlx-dev -n aks-mlx \
  --enable-ai-toolchain-operator --enable-oidc-issuer

# Or provision it at creation time (both flags are required together)
az aks create -g rg-mlx-dev -n aks-kaito \
  --node-count 2 --enable-managed-identity \
  --enable-ai-toolchain-operator --enable-oidc-issuer --generate-ssh-keys
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

## CLI cheat-sheet

```bash
# --- ACR: registry ---
az acr create -n acrmlx -g rg-mlx-dev --sku Standard --admin-enabled false
az acr update -n acrmlx --sku Premium                       # unlock geo-rep / private endpoints
az acr login  -n acrmlx                                     # token -> local docker credential
az acr show   -n acrmlx -o table
az acr credential show -n acrmlx                            # only if admin user is enabled

# --- ACR: images & repositories ---
az acr build -r acrmlx -t serving/fraud-model:1.0 -f Dockerfile .
az acr repository list       -n acrmlx -o table
az acr repository show-tags  -n acrmlx --repository serving/fraud-model --orderby time_desc --top 10 -o table
az acr repository show       -n acrmlx --repository serving/fraud-model
az acr repository delete     -n acrmlx --image serving/fraud-model:0.9 --yes
az acr import -n acrmlx --source docker.io/library/python:3.12-slim --image base/python:3.12-slim

# --- ACR: tasks & automated rebuilds ---
az acr run  -r acrmlx --cmd 'acr build -t serving/fraud-model:{{.Run.ID}} .' /dev/null
az acr task create -n rebuild-serving -r acrmlx --image serving/fraud-model:{{.Run.ID}} \
  --context https://github.com/org/serving.git --file Dockerfile --git-access-token $PAT \
  --base-image-trigger-enabled true
az acr task run  -n rebuild-serving -r acrmlx
az acr task list -r acrmlx -o table

# --- ACR: Premium replication & private endpoint ---
az acr replication create -r acrmlx -l westus2
az acr replication list   -r acrmlx -o table
az acr private-endpoint-connection list -r acrmlx -o table

# --- ACI: single-container runs ---
az container create -g rg-mlx-dev --name aci-smoketest --image acrmlx.azurecr.io/serving/fraud-model:1.0 \
  --cpu 2 --memory 4 --ports 8080 --acr-identity id-mlplatform --assign-identity id-mlplatform \
  --restart-policy Never
az container logs   -g rg-mlx-dev --name aci-smoketest --follow
az container exec   -g rg-mlx-dev --name aci-smoketest --exec-command "/bin/bash"
az container show   -g rg-mlx-dev --name aci-smoketest --query instanceView.state -o tsv
az container delete -g rg-mlx-dev --name aci-smoketest --yes     # stops billing

# --- AKS: cluster & credentials ---
az aks create -g rg-mlx-dev -n aks-mlx --node-count 2 --node-vm-size Standard_D4s_v5 \
  --enable-managed-identity --network-plugin azure --generate-ssh-keys
az aks get-credentials -g rg-mlx-dev -n aks-mlx
az aks get-versions -l eastus2 -o table
az aks upgrade -g rg-mlx-dev -n aks-mlx --kubernetes-version 1.31.1
az aks update  -g rg-mlx-dev -n aks-mlx --enable-ai-toolchain-operator --enable-oidc-issuer  # KAITO

# --- AKS: node pools (GPU) ---
az aks nodepool add   -g rg-mlx-dev --cluster-name aks-mlx --name gpupool \
  --node-vm-size Standard_NC24ads_A100_v4 --node-count 0 --min-count 0 --max-count 4 \
  --enable-cluster-autoscaler --node-taints sku=gpu:NoSchedule
az aks nodepool add   -g rg-mlx-dev --cluster-name aks-mlx --name gpunodriver \
  --node-vm-size Standard_NC24ads_A100_v4 --gpu-driver none        # bring your own GPU Operator
az aks nodepool list  -g rg-mlx-dev --cluster-name aks-mlx -o table
az aks nodepool scale -g rg-mlx-dev --cluster-name aks-mlx --name gpupool --node-count 2
az aks nodepool update  -g rg-mlx-dev --cluster-name aks-mlx --name gpupool --update-cluster-autoscaler --min-count 0 --max-count 8
az aks nodepool upgrade -g rg-mlx-dev --cluster-name aks-mlx --name gpupool --kubernetes-version 1.31.1

# --- kubectl: GPU sanity ---
kubectl get nodes -L agentpool
kubectl describe node <gpu-node> | grep nvidia.com/gpu
kubectl get pods -A
```

## Try it

Create a Standard ACR with the admin user disabled. Build a small serving image with `az acr build` from a simple `Dockerfile` (Python + your model-loading code). Grant your managed identity `AcrPull`, then run the image once on ACI as a smoke test and hit its endpoint. As a stretch goal, create an AKS cluster with a scale-to-zero GPU node pool (`--min-count 0`) plus a GPU taint, and confirm the pool shows zero nodes while idle — the same cost discipline you applied to training clusters, now for serving.
