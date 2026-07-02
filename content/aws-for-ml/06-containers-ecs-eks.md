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

ML images are large (a CUDA + PyTorch base can be 5–10 GB), so an unmanaged repo fills up fast and every stale tag costs storage. A **lifecycle policy** is the fix: it expires images automatically by age or count so you keep, say, the last 10 tagged builds and purge untagged layers after a day. Set it once and the repo stays lean without a cron job.

```bash
# Keep only the 10 most recent tagged images; expire untagged after 1 day
aws ecr put-lifecycle-policy --repository-name ml-serving \
  --lifecycle-policy-text '{
    "rules": [
      {"rulePriority": 1, "description": "expire untagged",
       "selection": {"tagStatus": "untagged", "countType": "sinceImagePushed",
                     "countUnit": "days", "countNumber": 1},
       "action": {"type": "expire"}},
      {"rulePriority": 2, "description": "keep last 10 tagged",
       "selection": {"tagStatus": "tagged", "tagPrefixList": ["v"],
                     "countType": "imageCountMoreThan", "countNumber": 10},
       "action": {"type": "expire"}}
    ]}'
```

**Image scanning** finds CVEs in your dependency layers before they reach production. Basic scanning is on-push and free; **enhanced scanning** (powered by Amazon Inspector) does continuous rescans as new CVEs are published, so an image that was clean last month gets re-flagged. You enable enhanced scanning once at the registry level, then read findings per image.

```bash
aws ecr put-registry-scanning-configuration --scan-type ENHANCED \
  --rules '[{"scanFrequency":"CONTINUOUS_SCAN","repositoryFilters":[{"filter":"*","filterType":"WILDCARD"}]}]'
aws ecr start-image-scan --repository-name ml-serving --image-id imageTag=v1
aws ecr describe-image-scan-findings --repository-name ml-serving --image-id imageTag=v1
```

**Pull-through cache** mirrors an upstream registry (the ECR Public gallery of DLCs, Docker Hub, Quay, GitHub Container Registry) into your account on first pull. Builds then depend on your private ECR, not on external uptime or Docker Hub rate limits, and later pulls are fast and in-region.

```bash
# Mirror the AWS ECR Public gallery under a local prefix
aws ecr create-pull-through-cache-rule \
  --ecr-repository-prefix dlc \
  --upstream-registry-url public.ecr.aws
# Now pull as if it were yours; ECR fetches and caches on first request
docker pull <acct>.dkr.ecr.us-east-1.amazonaws.com/dlc/... 
```

Routine housekeeping — inspecting what is stored and removing specific tags — uses a handful of describe/list/delete commands:

```bash
aws ecr describe-repositories
aws ecr list-images --repository-name ml-serving
aws ecr batch-delete-image --repository-name ml-serving \
  --image-ids imageTag=v0 imageTag=v0-rc1
```

One gotcha: **public** images live in a separate service. To push to a public repo you authenticate against `ecr-public`, and it is only available in `us-east-1`:

```bash
aws ecr-public get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin public.ecr.aws
```

## ECS: AWS-native orchestration

**Elastic Container Service** is AWS's own container orchestrator — simpler than Kubernetes and deeply integrated with AWS. You define a **task definition** (which image, CPU/memory, environment, IAM role) and run it as a **service** behind a load balancer or as one-off tasks. ECS has two launch types:

- **EC2 launch type**: you manage a fleet of EC2 instances that ECS schedules tasks onto. This is required for **GPU** workloads, because you choose GPU instances for the fleet.
- **Fargate launch type**: serverless containers — you specify CPU/memory and AWS runs them with no servers to manage.

ECS is a great fit when you want container serving without operating Kubernetes and your team is AWS-centric.

A task definition is the reproducible spec for one running unit. You register it (versioned as a revision), then a service keeps N copies of that revision healthy behind a load balancer. The minimum Fargate task def declares the launch compatibility, task-level CPU/memory (Fargate accepts only specific pairings — e.g. 256 CPU units with 512/1024/2048 MB), a network mode of `awsvpc`, and a container with its ECR image:

```bash
cat > taskdef.json <<'JSON'
{
  "family": "ml-serving",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024", "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<acct>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<acct>:role/ml-serving-task",
  "containerDefinitions": [{
    "name": "server",
    "image": "<acct>.dkr.ecr.us-east-1.amazonaws.com/ml-serving:v1",
    "portMappings": [{"containerPort": 8080}],
    "logConfiguration": {"logDriver": "awslogs",
      "options": {"awslogs-group": "/ecs/ml-serving",
                  "awslogs-region": "us-east-1", "awslogs-stream-prefix": "server"}}
  }]
}
JSON
aws ecs register-task-definition --cli-input-json file://taskdef.json
```

Note the two roles: the **execution role** lets the ECS agent pull the image and write logs, while the **task role** is what your model code assumes at runtime to read S3 or call other services — a frequent source of "works locally, AccessDenied in ECS" confusion. Standing up a cluster and a Fargate service on it looks like:

```bash
aws ecs create-cluster --cluster-name ml
aws ecs create-service --cluster ml --service-name ml-serving \
  --task-definition ml-serving --desired-count 2 --launch-type FARGATE \
  --network-configuration 'awsvpcConfiguration={subnets=[subnet-abc],securityGroups=[sg-123],assignPublicIp=ENABLED}'
# One-off batch task (e.g. a preprocessing job), not a long-running service
aws ecs run-task --cluster ml --task-definition preprocess --launch-type FARGATE \
  --network-configuration 'awsvpcConfiguration={subnets=[subnet-abc],securityGroups=[sg-123]}'
aws ecs list-tasks --cluster ml
aws ecs describe-services --cluster ml --services ml-serving
```

For a **GPU** task you switch to the EC2 launch type and declare a GPU in `resourceRequirements`; ECS then places the task only on instances that have the GPU free. The container def gains:

```json
"resourceRequirements": [{"type": "GPU", "value": "1"}]
```

To scale a service with traffic, ECS integrates with **Application Auto Scaling** rather than having its own knob. You register the service as a scalable target, then attach a target-tracking policy (e.g. hold average CPU at 60%, or track ALB request count per target). This is the ECS analog of an HPA:

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/ml/ml-serving --min-capacity 2 --max-capacity 20
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/ml/ml-serving --policy-name cpu60 \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 60.0,
    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"}}'
```

For manual or scheduled scaling you can also just bump the count directly:

```bash
aws ecs update-service --cluster ml --service ml-serving --desired-count 5
```

Two further concepts matter in production. **Capacity providers** decouple "run the task" from "where the capacity comes from": the `FARGATE_SPOT` provider runs interruptible tasks at a steep discount (great for fault-tolerant batch inference), and for EC2 launch type an Auto Scaling Group capacity provider with **managed scaling** grows and shrinks the instance fleet to match task demand. A common gotcha is forgetting that Fargate tasks need a route to ECR and CloudWatch — in a private subnet that means either a NAT gateway or VPC endpoints, or the task fails to pull its image with an opaque timeout.

## EKS: managed Kubernetes

**Elastic Kubernetes Service** runs upstream Kubernetes with an AWS-managed control plane. Choose EKS when you already have Kubernetes expertise, need portability across clouds/on-prem, or want the rich ecosystem (Kubeflow, KServe, custom operators). For ML, EKS is the standard when you are building your own model-serving platform at scale.

Standing up an EKS cluster with the CLI is a two-step affair: create the control plane, then attach compute (a managed node group or Fargate profile). Many teams prefer `eksctl`, which wraps both in one declarative command, but the raw AWS CLI shows what is actually happening. After the cluster exists you point `kubectl` at it by writing a kubeconfig entry — the single command every EKS user runs first:

```bash
aws eks create-cluster --name ml --role-arn arn:aws:iam::<acct>:role/eksClusterRole \
  --resources-vpc-config subnetIds=subnet-a,subnet-b
aws eks describe-cluster --name ml --query 'cluster.status'   # wait for ACTIVE
aws eks update-kubeconfig --name ml --region us-east-1        # now kubectl works
```

Compute comes as **managed node groups** — EC2 instances EKS provisions, patches, and drains for you. For GPU you simply choose GPU instance types; EKS selects the accelerated AMI automatically:

```bash
aws eks create-nodegroup --cluster-name ml --nodegroup-name gpu \
  --node-role arn:aws:iam::<acct>:role/eksNodeRole \
  --subnets subnet-a subnet-b \
  --instance-types g5.xlarge \
  --scaling-config minSize=0,maxSize=8,desiredSize=1
aws eks list-nodegroups --cluster-name ml
```

**Add-ons** are the operational glue EKS manages as versioned components rather than loose YAML you maintain by hand — the VPC CNI, CoreDNS, kube-proxy, the EBS CSI driver for persistent volumes, and (for GPU) you still layer in the NVIDIA device plugin. Managing them through the API keeps versions aligned with the cluster:

```bash
aws eks create-addon --cluster-name ml --addon-name aws-ebs-csi-driver
aws eks create-addon --cluster-name ml --addon-name vpc-cni
```

GPU on EKS is well supported: **EKS-optimized accelerated AMIs** ship with the NVIDIA drivers for G/P instances and support for Inferentia/Trainium, and the **NVIDIA device plugin** exposes GPUs to pods (a pod then requests `nvidia.com/gpu: 1` in its resource limits). **Karpenter** is the autoscaler of choice — it runs *inside* the cluster, watches for pending pods, and provisions right-sized GPU nodes on demand (including Spot capacity), consolidating and terminating them when the pods drain. It replaces the older Cluster Autoscaler + fixed node group model, and EKS Auto Mode can manage GPU nodes natively. This lets a serving platform scale GPU capacity to actual traffic instead of running an idle fleet. The practitioner gotcha: setting a GPU node group's `minSize=0` only saves money if your autoscaler can scale it back up on demand — Karpenter does this cleanly, whereas a static desired-count node group will sit idle at whatever you set.

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

## CLI cheat-sheet

```bash
# --- ECR: registry ---
aws ecr create-repository --repository-name ml-serving
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
aws ecr describe-repositories
aws ecr list-images --repository-name ml-serving
aws ecr batch-delete-image --repository-name ml-serving --image-ids imageTag=v0

# --- ECR: lifecycle, scanning, pull-through cache ---
aws ecr put-lifecycle-policy --repository-name ml-serving --lifecycle-policy-text file://lifecycle.json
aws ecr put-registry-scanning-configuration --scan-type ENHANCED --rules file://scan.json
aws ecr start-image-scan --repository-name ml-serving --image-id imageTag=v1
aws ecr describe-image-scan-findings --repository-name ml-serving --image-id imageTag=v1
aws ecr create-pull-through-cache-rule --ecr-repository-prefix dlc --upstream-registry-url public.ecr.aws
aws ecr-public get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin public.ecr.aws   # public repos, us-east-1 only

# --- ECS: cluster, task def, service ---
aws ecs create-cluster --cluster-name ml
aws ecs register-task-definition --cli-input-json file://taskdef.json   # GPU via resourceRequirements
aws ecs create-service --cluster ml --service-name ml-serving \
  --task-definition ml-serving --desired-count 2 --launch-type FARGATE \
  --network-configuration 'awsvpcConfiguration={subnets=[subnet-abc],securityGroups=[sg-123],assignPublicIp=ENABLED}'
aws ecs update-service --cluster ml --service ml-serving --desired-count 5
aws ecs run-task --cluster ml --task-definition preprocess --launch-type FARGATE \
  --network-configuration 'awsvpcConfiguration={subnets=[subnet-abc],securityGroups=[sg-123]}'
aws ecs list-tasks --cluster ml
aws ecs describe-services --cluster ml --services ml-serving

# --- ECS: service autoscaling (Application Auto Scaling) ---
aws application-autoscaling register-scalable-target --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount --resource-id service/ml/ml-serving \
  --min-capacity 2 --max-capacity 20
aws application-autoscaling put-scaling-policy --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount --resource-id service/ml/ml-serving \
  --policy-name cpu60 --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://tt.json

# --- EKS: cluster, kubeconfig, node groups, add-ons ---
aws eks create-cluster --name ml --role-arn arn:aws:iam::<acct>:role/eksClusterRole \
  --resources-vpc-config subnetIds=subnet-a,subnet-b
aws eks describe-cluster --name ml --query 'cluster.status'
aws eks update-kubeconfig --name ml --region us-east-1
aws eks create-nodegroup --cluster-name ml --nodegroup-name gpu \
  --node-role arn:aws:iam::<acct>:role/eksNodeRole --subnets subnet-a subnet-b \
  --instance-types g5.xlarge --scaling-config minSize=0,maxSize=8,desiredSize=1
aws eks list-nodegroups --cluster-name ml
aws eks create-addon --cluster-name ml --addon-name aws-ebs-csi-driver
```

## Try it

Containerize a small model server (FastAPI + your model) and push it to ECR. Deploy it two ways: first as a Fargate service on ECS behind an Application Load Balancer (CPU only), and confirm it autoscales on request count. Then define an ECS EC2-launch-type service on a `g5.xlarge` and run the same image with GPU inference enabled. Compare cold-start, latency, and cost between the two, and write down which serving path each of your real models would take.
