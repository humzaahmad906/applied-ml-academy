# 01 — Kubernetes Fundamentals for ML

You have a model in a container. You can run it on your laptop, on a VM, or behind a single Cloud Run service. So why do ML teams keep ending up on Kubernetes? Because the moment you have more than one GPU node, more than one model, or more than one team competing for expensive accelerators, you need something that packs workloads onto machines, restarts them when they die, and gives every job a fair slice of the cluster. That something is Kubernetes. This lesson builds the vocabulary — pods, deployments, services, namespaces, and `kubectl` — that every later lesson assumes. If you already know Docker, you are halfway there: Kubernetes is the layer that schedules and supervises containers across a fleet of machines instead of one.

## Why ML teams end up on Kubernetes

A single training run on one GPU box does not need Kubernetes. The pressure builds when the shape of the work changes:

- **Shared, expensive hardware.** A cluster of eight H100 nodes costs tens of thousands of dollars a month. You cannot afford to let it idle, and you cannot let one engineer's notebook hog all of it. You need a scheduler that bin-packs jobs and enforces quotas.
- **Many workloads, one platform.** Training jobs, batch scoring, online inference, notebooks, and data pipelines all want the same GPUs at different times. Kubernetes lets them coexist with priorities and preemption.
- **Self-healing and rollout.** An inference service must survive a node failure, roll out a new model version without downtime, and roll back instantly. Kubernetes gives you that declaratively.
- **Portability.** The same manifests run on EKS, GKE, AKS, or on-prem. Teams that refuse to be locked to one cloud's managed ML product land on Kubernetes as the common substrate.

The cost of admission is real: Kubernetes has a steep learning curve and a large operational surface. The rule of thumb is that you adopt it when the alternative — a pile of VMs wired together by scripts — has become more painful than the cluster itself.

## The control plane and nodes

A Kubernetes cluster has two halves. The **control plane** is the brain: the API server (everything talks to it), `etcd` (the database of cluster state), the scheduler (decides which node runs each pod), and controllers (reconcile reality toward your declared intent). The **nodes** are the workers — the machines, often GPU machines in an ML cluster, that actually run your containers.

You almost never manage the control plane yourself. Managed offerings — Amazon EKS, Google GKE, Azure AKS — run it for you. Your job is to declare *what* you want in YAML and let the control plane make it so. This is the single most important mental shift: Kubernetes is **declarative**. You do not tell it "start this container." You tell it "I want three replicas of this to exist," and a controller works continuously to keep three alive, restarting or rescheduling as machines fail.

## Pods: the unit of scheduling

The smallest thing Kubernetes schedules is not a container — it is a **pod**. A pod is one or more containers that share a network namespace (same IP, same localhost) and can share storage volumes. Most ML pods have one main container (your model server or training process), sometimes with a **sidecar** (a logging agent, a metrics exporter, or a storage driver like a GCS FUSE mount).

Here is a minimal pod that runs a training script:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: train-once
spec:
  restartPolicy: Never
  containers:
    - name: trainer
      image: us-central1-docker.pkg.dev/myco/ml-images/trainer:v1
      command: ["python", "train.py"]
      resources:
        requests: {cpu: "4", memory: "16Gi"}
        limits: {cpu: "8", memory: "32Gi"}
```

Two things to notice. First, `resources.requests` is what the scheduler uses to find a node with room; `limits` is the hard ceiling the kubelet enforces. Getting requests right is the core of not wasting money — we return to this in lesson 03. Second, you rarely create bare pods. A pod alone has no supervisor: if its node dies, the pod is gone forever. You wrap pods in higher-level controllers that manage them for you.

## Deployments: keeping replicas alive

A **Deployment** manages a set of identical pods for you. You declare how many replicas you want and the pod template; the Deployment controller creates a ReplicaSet that keeps exactly that many pods running, replaces crashed ones, and handles rolling updates when you change the image.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-serve
spec:
  replicas: 3
  selector:
    matchLabels: {app: fraud-serve}
  template:
    metadata:
      labels: {app: fraud-serve}
    spec:
      containers:
        - name: server
          image: us-central1-docker.pkg.dev/myco/ml-images/fraud-serve:v2
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet: {path: /healthz, port: 8080}
            initialDelaySeconds: 20
```

Deployments are the right home for **stateless online inference**: every replica is interchangeable, and traffic can hit any of them. The `readinessProbe` is not optional in production — it tells Kubernetes when a pod has finished loading the model and can take traffic, so requests are not routed to a pod still warming up. Change the image and re-apply, and the Deployment rolls new pods in and old ones out a few at a time, never dropping below capacity.

For training and batch work you use a **Job** instead of a Deployment — a Job runs pods to completion rather than keeping them running forever. Lesson 04 covers Jobs in depth. For workloads that need stable identities and ordered startup (some distributed training topologies), a **StatefulSet** is the tool; lesson 05 gets there.

## Services: stable addresses for moving pods

Pods are ephemeral. They come and go, and each gets a new IP. So how does traffic reach your three `fraud-serve` replicas when their IPs keep changing? A **Service** gives a stable virtual IP and DNS name in front of a set of pods, load-balancing across whichever pods currently match its label selector.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-serve
spec:
  selector: {app: fraud-serve}
  ports:
    - port: 80
      targetPort: 8080
```

Now any pod in the cluster can reach the model at `http://fraud-serve` regardless of which replicas exist. That is the same service-discovery-by-name idea you saw in Docker Compose, scaled to a cluster. The default `ClusterIP` type is internal only. To expose a service outside the cluster you use a `LoadBalancer` type (provisions a cloud load balancer) or, more commonly for HTTP, an **Ingress** or **Gateway** that routes by hostname and path to many backend services behind one external address.

A special variant, the **headless service** (`clusterIP: None`), gives every pod its own DNS record instead of one virtual IP. That is exactly what multi-node distributed training needs so workers can find each other by hostname — you will see it again in lesson 05.

## Namespaces: dividing one cluster

A **namespace** is a virtual partition of the cluster. Teams, environments, or projects each get one, and you attach **resource quotas** and **RBAC** (role-based access control) per namespace so a runaway training job in `team-vision` cannot starve `team-nlp`. This matters enormously on a shared GPU cluster, where the whole point is that many groups share one expensive fleet without stepping on each other.

```bash
kubectl create namespace team-vision
kubectl config set-context --current --namespace=team-vision
```

Quotas are what turn "we all share the cluster" from a hope into a rule — a `ResourceQuota` can cap a namespace at, say, 8 GPUs and 200 CPU cores. Lesson 04 builds on this with Kueue for queue-based fair-sharing across teams.

## kubectl: your daily driver

`kubectl` is the CLI that talks to the API server. You will live in it. The workflow is almost always: write YAML, `apply` it, then `get`/`describe`/`logs` to see what happened.

```bash
# Point kubectl at your cluster (writes a kubeconfig context)
gcloud container clusters get-credentials ml-cluster --region=us-central1
# or: aws eks update-kubeconfig --name ml-cluster --region us-east-1

kubectl apply -f deployment.yaml       # declare desired state (create or update)
kubectl get pods -o wide               # what's running, on which node
kubectl get deploy,svc                 # deployments and services at a glance
kubectl describe pod fraud-serve-abc   # events: why is it Pending / CrashLoopBackOff?
kubectl logs -f fraud-serve-abc        # stream container logs
kubectl logs fraud-serve-abc --previous # logs from the crashed container
kubectl exec -it fraud-serve-abc -- bash  # shell into a running pod
kubectl top pods                       # live CPU/memory usage (needs metrics-server)
kubectl delete -f deployment.yaml      # tear down
```

Two debugging reflexes will save you hours. When a pod is stuck `Pending`, run `kubectl describe pod` and read the **Events** at the bottom — it almost always says "insufficient nvidia.com/gpu" or "no nodes match node selector," which points straight at a scheduling problem. When a pod is `CrashLoopBackOff`, run `kubectl logs --previous` to see why the last attempt died. Ninety percent of Kubernetes debugging is `describe` for scheduling problems and `logs` for runtime problems.

## How this fits an ML platform

Everything else in this course is built from these primitives. GPU scheduling (lesson 02) is pods requesting `nvidia.com/gpu`. Autoscaling (lesson 03) adjusts Deployment replica counts and node counts. Batch training (lesson 04) is Jobs plus a queue. Distributed training (lesson 05) is a set of pods wired together with a headless service. Model serving (lesson 06) is a Deployment behind a Service, wrapped by a higher-level serving controller. Learn to read and write these four objects — pod, Deployment, Service, namespace — and the rest is composition.

## Key takeaways

- Kubernetes is a **declarative** system: you state desired end-state in YAML, and controllers continuously reconcile reality toward it. ML teams adopt it for shared expensive hardware, mixed workloads, self-healing, and cloud portability.
- The **pod** is the unit of scheduling (one or more co-located containers). You rarely create bare pods — you wrap them in a **Deployment** (stateless replicas, online inference), a **Job** (run-to-completion training/batch), or a **StatefulSet**.
- A **Service** gives a stable name and load-balances across pods whose IPs keep changing; a **headless service** gives each pod its own DNS record for distributed training.
- **Namespaces** plus **ResourceQuota** and **RBAC** partition one cluster across teams — essential for a shared GPU fleet.
- `kubectl apply` to declare, `kubectl get/describe/logs` to observe. `describe` diagnoses scheduling (`Pending`); `logs --previous` diagnoses crashes (`CrashLoopBackOff`).

## Try it

1. Create a small cluster (GKE Autopilot, EKS, or a local `kind`/`minikube` cluster) and run `kubectl get nodes` to confirm `kubectl` is wired up.
2. Write a Deployment for any HTTP container (even `nginx`) with `replicas: 3`, apply it, and watch the three pods appear with `kubectl get pods -w`.
3. Delete one pod with `kubectl delete pod <name>` and watch the Deployment controller immediately recreate a replacement — self-healing in action.
4. Add a `Service` in front of it and, from another pod (`kubectl run tmp --rm -it --image=busybox -- sh`), `wget -qO- http://<service-name>` to confirm service discovery by name works.
5. Create a `team-a` namespace, attach a `ResourceQuota` capping it at 2 CPUs, and try to deploy something that requests 4 — watch the pod fail to schedule, and read the reason in `kubectl describe`.
