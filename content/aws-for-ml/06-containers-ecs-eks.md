# 06 — Containers: ECR, ECS, EKS, Fargate

Containers are how modern ML code travels reproducibly from a laptop to training clusters to production endpoints. AWS offers a registry to store images and several orchestrators to run them, each with different tradeoffs for control, operational burden, and GPU support. This module explains ECR, ECS, EKS, and Fargate, and — most importantly for an ML engineer — when to reach for containers directly versus letting SageMaker handle it.

## ECR: the image registry

**Amazon Elastic Container Registry** stores your Docker/OCI images. It offers **private** registries (the default for your proprietary training and serving images) and **public** registries for sharing. Two features matter for ML: image scanning for vulnerabilities, and **pull-through cache**, which mirrors upstream registries (like the AWS Deep Learning Container gallery or public GPU base images) into your account so builds do not depend on external availability and pulls stay fast and private.

```bash
aws ecr create-repository --repository-name ml-serving
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    <acct>.dkr.ecr.us-east-1.amazonaws.com
docker build -t ml-serving:v1 .
docker tag ml-serving:v1 <acct>.dkr.ecr.us-east-1.amazonaws.com/ml-serving:v1
docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/ml-serving:v1
```

A strong practice is to base your images on **AWS Deep Learning Containers** so training and serving share the same framework build, and push both to ECR — that image is then the unit of deployment across every orchestrator below.

## ECS: AWS-native orchestration

**Elastic Container Service** is AWS's own container orchestrator — simpler than Kubernetes and deeply integrated with AWS. You define a **task definition** (which image, CPU/memory, environment, IAM role) and run it as a **service** behind a load balancer or as one-off tasks. ECS has two launch types:

- **EC2 launch type**: you manage a fleet of EC2 instances that ECS schedules tasks onto. This is required for **GPU** workloads, because you choose GPU instances for the fleet.
- **Fargate launch type**: serverless containers — you specify CPU/memory and AWS runs them with no servers to manage.

ECS is a great fit when you want container serving without operating Kubernetes and your team is AWS-centric.

## EKS: managed Kubernetes

**Elastic Kubernetes Service** runs upstream Kubernetes with an AWS-managed control plane. Choose EKS when you already have Kubernetes expertise, need portability across clouds/on-prem, or want the rich ecosystem (Kubeflow, KServe, custom operators). For ML, EKS is the standard when you are building your own model-serving platform at scale.

GPU on EKS is well supported: **EKS-optimized accelerated AMIs** ship with the NVIDIA drivers for G/P instances and support for Inferentia/Trainium, and the **NVIDIA device plugin** exposes GPUs to pods. **Karpenter** is the autoscaler of choice — it provisions right-sized GPU nodes on demand based on pending pods (including Spot capacity), and EKS Auto Mode can manage GPU nodes natively. This lets a serving platform scale GPU capacity to actual traffic instead of running an idle fleet.

## Fargate: serverless containers (and the GPU caveat)

**Fargate** removes server management entirely: you hand it a container and resource sizing, and it runs. It works with both ECS and EKS. The critical ML limitation: **Fargate does not support GPUs.** Any GPU container workload must use the EC2 launch type (ECS) or GPU node groups (EKS). Fargate is excellent for CPU-based pieces of an ML system — preprocessing, feature computation, orchestration glue, CPU inference for small models — where its zero-ops model shines and you are not paying for idle instances.

## Choosing an orchestrator for ML

- **CPU inference, preprocessing, glue, spiky traffic** → Fargate (ECS or EKS), or Lambda for the smallest/most bursty pieces.
- **GPU serving, AWS-centric team, no Kubernetes desire** → ECS on EC2 GPU instances.
- **GPU serving at scale, Kubernetes expertise, portability** → EKS with GPU node groups + Karpenter.
- **You want managed ML serving with autoscaling, multi-model, and MLOps built in** → SageMaker endpoints (covered later), which run containers for you without you operating any orchestrator.

## Containers vs SageMaker for serving

This is the recurring decision. **SageMaker endpoints** give you managed real-time/serverless/async inference, autoscaling, multi-model hosting, and tight integration with the model registry and monitoring — with no cluster to run. **Containers on ECS/EKS** give you maximum control: custom serving frameworks, non-standard routing, multi-model gateways of your own design, and reuse of existing Kubernetes tooling. Many teams use both: SageMaker for the bulk of standard endpoints, and a bespoke EKS service for the one workload with special requirements. Because both consume the same ECR image, you can move between them without rebuilding your model container.

## How this fits the whole ML solution

Containers are the portable unit that ties the system together: the same ECR image can run a Fargate preprocessing task, a training job, and an EKS serving pod. The orchestrator you pick for each stage is a control-vs-convenience decision, but the image is shared, so your build pipeline (from the CI/CD story later) produces one artifact that flows through ingestion, training, and serving. Standardizing on DLC-based images in ECR is what keeps "works in training" and "works in production" the same statement.

## Key takeaways

- ECR stores images; use pull-through cache and DLC-based images so builds are private, fast, and consistent across train/serve.
- ECS is AWS-native orchestration (EC2 launch type for GPU, Fargate launch type for serverless CPU); EKS is managed Kubernetes for portable, at-scale serving.
- **Fargate has no GPU support** — GPU workloads need ECS-on-EC2 or EKS GPU node groups; use Karpenter to autoscale GPU nodes.
- Choose containers for control and portability, SageMaker endpoints for managed serving with built-in MLOps — both run the same ECR image.

## Try it

Containerize a small model server (FastAPI + your model) and push it to ECR. Deploy it two ways: first as a Fargate service on ECS behind an Application Load Balancer (CPU only), and confirm it autoscales on request count. Then define an ECS EC2-launch-type service on a `g5.xlarge` and run the same image with GPU inference enabled. Compare cold-start, latency, and cost between the two, and write down which serving path each of your real models would take.
