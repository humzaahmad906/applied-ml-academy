# 04 — Batch Training and Queues

Deployments are for services that run forever. Training is different: it starts, does work, and finishes. The Kubernetes primitive for run-to-completion work is the **Job**, and for a single researcher on a single GPU that is enough. But a shared GPU cluster is a scarce-resource allocation problem in disguise. Ten people submit training jobs Monday morning; there are four H100 nodes. Who runs first? Who waits? How do you stop one team from grabbing everything, and how do you make sure a distributed job that needs eight GPUs gets *all eight at once* instead of deadlocking on four? Plain Jobs answer none of this. This lesson covers the Job primitive, then the queueing and gang-scheduling layer — Kueue and Volcano — that turns a pile of Jobs into a fair, quota-governed batch system.

## The Job primitive

A **Job** creates one or more pods and considers itself complete when the required number succeed. Unlike a Deployment, it does not restart finished pods — it runs them to completion and stops. The key fields are `completions` (how many successful pod runs constitute done), `parallelism` (how many run at once), and `backoffLimit` (retries before the Job is marked failed).

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-resnet
spec:
  backoffLimit: 2                 # retry a failed pod up to twice
  ttlSecondsAfterFinished: 3600   # auto-clean the Job 1h after it finishes
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: us-central1-docker.pkg.dev/myco/ml-images/trainer:v3
          command: ["python", "train.py", "--epochs=50"]
          resources:
            limits: {nvidia.com/gpu: 1, cpu: "8", memory: "48Gi"}
      tolerations:
        - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
```

Two fields earn their keep in practice. `ttlSecondsAfterFinished` auto-deletes completed Jobs so your cluster does not accumulate thousands of finished-Job records. And for embarrassingly parallel work — scoring a dataset in shards, running a hyperparameter sweep — an **indexed Job** (`completionMode: Indexed`) hands each pod a `JOB_COMPLETION_INDEX` env var so pod 0 takes shard 0, pod 1 takes shard 1, and so on, with `parallelism` controlling how many run concurrently.

```yaml
spec:
  completions: 100
  parallelism: 10
  completionMode: Indexed        # each pod gets JOB_COMPLETION_INDEX 0..99
```

## Why plain Jobs break down on a shared cluster

Submit ten Jobs to a four-node GPU cluster with nothing but the default scheduler and two failure modes appear immediately.

**No fairness or admission control.** The scheduler places pods greedily as GPUs free up, in no particular order. A user who submits 50 jobs monopolizes the cluster; a user who submits one waits behind them. There is no notion of "team A gets at most 8 GPUs" or "interactive jobs jump ahead of overnight sweeps." Jobs that cannot fit sit `Pending` indefinitely, cluttering the queue.

**Partial-allocation deadlock (the gang problem).** A distributed training job needs, say, 8 GPUs *simultaneously* — all workers must be up to start the NCCL collective. The default scheduler places pods one at a time. It may grant 5 GPUs to job A and 3 to job B, at which point neither can start and neither will release what it holds. Both are stuck. This is a resource deadlock, and it is common enough that solving it — **gang scheduling**, the guarantee that either all of a job's pods are scheduled together or none are — is a headline feature of the batch layer.

## Kueue: quota and admission

**Kueue** is the modern answer to fairness and quota. It is a job-level **admission controller**: it does not replace the scheduler, it sits in front of it. You submit a Job (or JobSet, RayJob, or Kubeflow TrainJob — Kueue integrates with all of them), Kueue **suspends** it, and only **admits** it — unsuspends it so the scheduler can place its pods — when the job's queue has quota available. This gives you fair-share, quotas, and priority without swapping out the scheduler your cluster already runs. Kueue lives under `kubernetes-sigs`; note it is *not* independently a CNCF project despite frequent claims otherwise.

The model has three objects. A **ResourceFlavor** describes a kind of hardware (e.g. H100 nodes). A **ClusterQueue** defines a quota pool over flavors — "this pool has 16 H100s" — and policies for borrowing and preemption. A **LocalQueue** is the namespace-scoped handle a user submits to.

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata: {name: team-vision}
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: ["nvidia.com/gpu", "cpu", "memory"]
      flavors:
        - name: h100
          resources:
            - {name: "nvidia.com/gpu", nominalQuota: 16}
            - {name: "cpu", nominalQuota: 400}
            - {name: "memory", nominalQuota: "2000Gi"}
  preemption:
    reclaimWithinCohort: Any        # reclaim borrowed quota when the owner needs it
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata: {name: default, namespace: team-vision}
spec: {clusterQueue: team-vision}
```

A user then attaches a Job to the queue with one label and drops the fields Kueue manages:

```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: default
```

The powerful part is **cohorts and borrowing**. Group several ClusterQueues into a cohort and idle quota flows to whoever needs it: if team-vision is quiet, team-nlp can borrow its H100s — until team-vision submits work, at which point Kueue preempts the borrowed jobs and reclaims the quota. This is how you keep an expensive cluster near 100% utilization while still guaranteeing each team its fair share. Kueue also does gang admission — it admits all of a job's pods together — though it delegates the actual pod placement to whatever scheduler is running.

## Volcano: a full batch scheduler with gang scheduling

Where Kueue admits jobs and leaves placement to the default scheduler, **Volcano** (a CNCF incubating project) *replaces* the scheduler with a batch-oriented one. It provides true **gang scheduling** at the placement level (all-or-nothing pod placement, killing the deadlock described above), fair-share via the Dominant Resource Fairness algorithm, bin-packing, topology-aware placement (co-locate a job's pods on nodes with fast interconnect), and priority preemption. You submit work as a Volcano `Job` or `PodGroup`, and the `minMember` field expresses the gang guarantee:

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata: {name: distributed-train}
spec:
  minAvailable: 8            # gang: schedule all 8, or none
  schedulerName: volcano
  queue: research
  tasks:
    - replicas: 8
      name: worker
      template:
        spec:
          containers:
            - name: worker
              image: us-central1-docker.pkg.dev/myco/ml-images/ddp:v1
              resources: {limits: {nvidia.com/gpu: 1}}
```

**Kueue or Volcano?** They are not mutually exclusive and increasingly interoperate. The rough guidance: reach for **Kueue** when you want quota, fair-share, and multi-tenant admission while keeping the default scheduler and its ecosystem — it is the lighter-touch, more composable choice, and it is what Kubeflow Trainer and KubeRay integrate with natively. Reach for **Volcano** when you need scheduler-level gang scheduling and topology awareness that only a replacement scheduler can guarantee, especially for tightly-coupled distributed training. A common production setup runs Kueue for quota/admission on top of a gang-capable scheduler (Volcano, or NVIDIA's newer KAI Scheduler) for placement. *(The exact division of gang responsibilities between Kueue and the underlying scheduler is evolving; verify against current Kueue docs for your version.)*

## Priorities, preemption, and suspend/resume

Two mechanics make a queue feel fair in practice. The first is **priority**. A `PriorityClass` assigns a numeric weight to a workload; when the cluster is full and a higher-priority job arrives, the scheduler (or Kueue/Volcano) **preempts** lower-priority pods to make room, then reschedules them when capacity frees up. The canonical ML use is separating tiers: interactive debugging and demos get a high priority so a researcher waiting at their keyboard is served immediately, while overnight sweeps and long fine-tunes get a low priority and yield when needed.

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata: {name: interactive}
value: 1000000
globalDefault: false
description: "Interactive/debug jobs preempt batch."
```

The second is **suspend/resume**, which is what Kueue's admission control actually does under the hood. A suspended Job has its pods deleted (or never created), so it consumes zero resources while it waits — unlike a `Pending` pod that still holds a scheduling slot. When quota frees up, Kueue flips `spec.suspend` to false and the Job's pods are created. This is why a queue of a hundred suspended jobs costs nothing: they are inert until admitted. You can watch the state transitions with `kubectl get workloads`, where each job shows as `Pending`, `Admitted`, or `Finished`. Understanding this suspend-based model is the key to reasoning about Kueue: quota is enforced at *admission*, not at scheduling, so the expensive scheduler never even sees a job until its team has room for it.

## Cost and operations notes

Quota-based queueing is fundamentally a **cost** tool. On a cluster where GPUs are the dominant line item, the goal is high utilization without contention. Cohort borrowing pushes utilization up by lending idle quota; preemption keeps borrowing from becoming theft. Pair the batch layer with **Spot/preemptible** GPU nodes for cost-tolerant sweeps — but design jobs to **checkpoint frequently** so a preemption costs minutes, not the whole run. Set `ttlSecondsAfterFinished` everywhere so finished Jobs are reaped. And give interactive/debug jobs a higher priority class than overnight batch, so a researcher waiting at their desk is not stuck behind a 12-hour sweep.

## Key takeaways

- A **Job** runs pods to completion (not forever); use `completions`/`parallelism` for fan-out, `completionMode: Indexed` for sharded work, `backoffLimit` for retries, and `ttlSecondsAfterFinished` to auto-clean.
- Plain Jobs on a shared cluster have **no fairness/quota** and suffer **gang deadlock** — partial GPU allocation where two distributed jobs each hold some GPUs and neither can start.
- **Kueue** (kubernetes-sigs) is an **admission controller**: it suspends jobs and admits them when quota is free, adding fair-share, per-team ClusterQueues, and **cohort borrowing with preemption** — high utilization plus guaranteed shares. It keeps the default scheduler.
- **Volcano** (CNCF incubating) is a **replacement scheduler** offering true **gang scheduling**, DRF fair-share, and topology-aware placement — reach for it when tightly-coupled distributed jobs need all-or-nothing placement.
- Queueing is a cost tool: lend idle quota via cohorts, run sweeps on **Spot with frequent checkpointing**, and prioritize interactive jobs over batch.

## Try it

1. Submit an indexed Job with `completions: 10, parallelism: 3` that prints `$JOB_COMPLETION_INDEX`, and confirm each pod processes a distinct shard.
2. Install Kueue, define a ClusterQueue with a small GPU quota, and submit more GPU jobs than the quota allows — watch some get `Admitted` and others wait `Suspended` in `kubectl get workloads`.
3. Create two ClusterQueues in a cohort, leave one idle, and submit a large job to the other; confirm it *borrows* the idle quota. Then submit to the idle queue and watch preemption reclaim it.
4. Reproduce the gang deadlock: on a 4-GPU cluster, submit two 3-GPU distributed jobs with the default scheduler and watch both stick partially scheduled. Then reschedule them through Volcano with `minAvailable` and confirm the deadlock is gone.
5. Add a high-priority `PriorityClass` to an interactive job and a low one to a batch sweep; fill the cluster with the sweep, submit the interactive job, and watch preemption make room.
