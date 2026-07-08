# 06g — Shadow Deployments and Progressive Delivery

The advanced serving guide already introduced canary, A/B, blue/green, and shadow as a list of "serving patterns" you should recognize. This lesson goes deeper on the two that carry the most weight in a modern release pipeline and are the easiest to get wrong: **shadow (mirror) deployment** as the zero-risk pre-promotion gate, and **automated progressive delivery** as the machinery that ramps a new model to 100% (or rolls it back) on metric evidence instead of a human eyeballing a dashboard. The framing throughout is the deployment-safety spectrum: every release strategy is a point on a curve trading rollout speed against blast radius, and picking the right point per model is a senior-level judgment call.

## The Deployment-Safety Spectrum

Every strategy for replacing a running version with a new one sits somewhere on a line from "fast and dangerous" to "slow and safe." The knob is **how much production traffic hits the new version before you're committed, and how many users a bad version can hurt.**

| Strategy | How it works | Blast radius | Rollback | Serves new version to users? |
|---|---|---|---|---|
| **Recreate** | Kill all old pods, start new ones | Total (downtime + all users) | Redeploy old | Yes, immediately, everyone |
| **Rolling** | Replace pods N at a time | Partial, grows during rollout | Roll forward/back pod by pod | Yes, growing slice |
| **Blue/Green** | Two full environments, flip all traffic at once | All users at the flip instant | Flip back (fast) | Yes, everyone at once |
| **Canary** | Route X% to new, ramp on health | X% of users | Drop X to 0 | Yes, small slice |
| **Shadow / Mirror** | Copy traffic to new version, discard its output | **Zero** | Nothing to roll back | **No — never served** |

Read the table bottom-up and it becomes a release *sequence*, not a menu. The mature pattern is shadow first (prove it doesn't fall over on real traffic, zero user risk), then canary (prove a small real slice is healthy and doesn't regress business metrics), then progressive ramp to 100%. Recreate is for stateful jobs and dev. Rolling is the Kubernetes default, fine for stateless app code but weak for ML because it gives you no isolated comparison of old vs new predictions. Blue/green buys instant rollback but flips *everyone* at once — for a model, a bad version reaches 100% of users until you notice, which is why it's rare where gradual rollout is cheap.

The ML-specific reason to care more than a typical backend team: a new model version can pass every unit test, load fine, return 200s, and still be *quietly worse* — training-serving skew, a feature null in production but dense in training, a preprocessing version mismatch. None of that shows up in a green deploy. It shows up in the *predictions*, which is exactly what shadow and canary let you inspect.

## Shadow (Mirror) Deployment in Depth

Shadow mode sends a **copy** of each live request to the candidate model in parallel with the production model. The production model's response is the only one the user ever sees; the shadow model's prediction is logged and thrown away as far as the user is concerned. You then compare the two prediction streams offline. Because the shadow output is never served, the user-facing risk is exactly zero — the worst a broken shadow model can do is fill a log and burn some compute.

What this buys you that offline replay does not: shadow runs against the *real* production request distribution, through the *real* feature-fetch path, on the *real* serving hardware. It surfaces training-serving skew, feature-availability gaps, runtime numerical differences, cache behavior, and tail-latency problems that a replay of a saved dataset will happily paper over. Shops with strict release discipline make it mandatory — Uber, for example, requires every new model version to sit in shadow for a minimum window before any canary traffic is enabled.

### A traffic-mirroring sketch

At the mesh layer this is a few lines. Istio's `VirtualService` mirrors with a `mirror` destination and a `mirrorPercentage`:

```yaml
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: fraud-model
spec:
  hosts: [fraud-model]
  http:
    - route:
        - destination:            # 100% of real traffic → champion
            host: fraud-model
            subset: v17
          weight: 100
      mirror:                       # a copy → shadow candidate
        host: fraud-model
        subset: v18
      mirrorPercentage:
        value: 10.0                 # start small; ramp toward 100
```

Two things to internalize about mirrored requests. First, they are **fire-and-forget** — the mesh discards the shadow's response, so the shadow model's latency and errors *cannot* affect the user path. Second, Istio appends `-shadow` to the `Host`/`Authority` header of the copy, which is your hook for making the shadow request safe: your shadow service must treat it as read-only. **The single biggest shadow-mode footgun is side effects.** If the shadow path writes to the same database, emits the same Kafka event, calls the same payment API, or increments the same counter as production, you have double-executed every side effect in the system. Shadow requests must be idempotent-by-omission: no writes, no external calls with effects, no metric emission that pollutes production dashboards. Stamp a correlation ID on the request so you can join the champion and shadow predictions for the same input downstream.

An application-layer alternative when you don't run a mesh: dual-dispatch in the service itself. Serve from champion synchronously; enqueue the same input to the candidate on a **non-blocking** async path (a queue consumed by the candidate service) so the shadow's latency never enters the user's critical path. The mesh approach is cleaner because the isolation is enforced by infrastructure, not by remembering to `await` nothing.

### What to measure

Shadow gives you two prediction streams keyed by correlation ID. The comparison pipeline runs offline and watches:

- **Prediction agreement.** For classification, the rate at which champion and candidate agree on the argmax label, plus the confidence gap on disagreements. For ranking, Kendall's tau or NDCG delta between the two orderings. For regression, the distribution of per-request deltas. A large, systematic disagreement is a red flag *even if you can't yet say which model is right* (see the catch below).
- **Score-distribution divergence.** KL divergence (or PSI) between the champion's and candidate's output-score distributions. A shifted distribution often means a skew or preprocessing bug, not a genuinely different model.
- **Operational health.** The candidate's latency (P50/P95/P99), error rate, timeout rate, and null-feature rate — measured on real traffic and real hardware. This is where "passed offline, dies in prod" gets caught.
- **Feature completeness.** How often each feature the candidate expects is actually present at serving time. A feature that was 100% dense in your training warehouse but 40% null in the online store is the classic shadow catch.

## Interleaving and A/B for Models

Shadow tells you the candidate *runs*. It cannot tell you the candidate is *better*, because you never served its predictions, so no user ever reacted to them. To measure "better" on a business metric you need users to actually experience the new model. Two tools, in ascending cost:

**A/B testing** splits users into arms — arm A on champion, arm B on candidate — and compares a business metric (conversion, click-through, fraud caught net of false positives) with a proper power analysis and significance test. This is the gold standard for a causal "is it better" answer and ties directly to the experimentation material in the medium guide. Its weakness is *sensitivity*: to detect a small ranking improvement you may need weeks of traffic and huge samples, because the signal is buried in between-user variance.

**Interleaving** attacks exactly that weakness for ranking/recommendation models. Instead of assigning each *user* to one model, you blend both models' ranked lists into a *single* list shown to *every* user, tag which model contributed each item, and attribute clicks back to the model that surfaced them. Because every user is their own control, between-user variance cancels and the test is dramatically more sensitive. Netflix reported interleaving needs on the order of **1/100th** the samples of the most sensitive A/B metric to pick the preferred algorithm; Airbnb's 2025 search-ranking work reports up to a **100× sensitivity gain and 50× speedup** versus A/B. The standard pipeline uses interleaving as a cheap, high-sensitivity *screening* stage to prune many candidates fast, then confirms the survivor with a full A/B test for the trustworthy business-metric readout. Interleaving does not replace A/B — it decides *what's worth* A/B testing.

## Automated Progressive Rollout

Manually watching a canary dashboard and clicking "promote" does not scale and does not fire at 3 a.m. Progressive delivery automates the loop: **ramp traffic in steps, evaluate metrics at each step against SLO gates, and promote or roll back automatically.** The two dominant Kubernetes tools are **Argo Rollouts** and **Flagger**.

The shared model is an `AnalysisTemplate` (Argo's term) or a metrics block (Flagger's) that queries a metric provider — Prometheus, Datadog, CloudWatch, New Relic, or a custom job — at each step and compares against a threshold. If the metric holds, the controller advances to the next weight; if it breaches, it aborts and rolls back to the stable version automatically.

```yaml
# Argo Rollouts: canary with SLO-gated steps
strategy:
  canary:
    steps:
      - setWeight: 10
      - pause: {duration: 10m}
      - analysis:                       # SLO gate
          templates: [{templateName: success-and-latency}]
      - setWeight: 50
      - pause: {duration: 10m}
      - analysis:
          templates: [{templateName: success-and-latency}]
      - setWeight: 100
---
# AnalysisTemplate: promote only if success rate ≥ 99% AND p95 < 200ms
metrics:
  - name: success-rate
    interval: 1m
    count: 5
    failureLimit: 2                     # 2 breaches → abort → rollback
    successCondition: result >= 0.99
    provider:
      prometheus:
        query: |
          sum(rate(http_requests_total{job="fraud-model",code!~"5.."}[1m]))
          / sum(rate(http_requests_total{job="fraud-model"}[1m]))
```

The controllers differ in philosophy. **Argo Rollouts** replaces the `Deployment` with a `Rollout` CRD and gives you explicit, step-by-step control plus a UI. **Flagger** watches a normal `Deployment` and, once configured, drives the *entire* canary lifecycle automatically without per-release editing. Pick Argo for explicit control and a dashboard; pick Flagger for a hands-off pipeline. Both integrate with the same service meshes (Istio, Linkerd) that do your traffic mirroring, so shadow and canary share infrastructure.

The critical design point for ML: **your SLO gate should include model-quality proxies, not just infra health.** A canary that keeps success rate at 99.99% and P95 under budget can still be quietly recommending garbage. Add gates on prediction-distribution drift versus the stable version, on business-metric proxies you can measure in the canary window (e.g., click-through on the canary slice), and on any fast-feedback label you have. Anything you can express as a Prometheus/Datadog query can be a gate.

## The ML-Specific Catch: Operational vs Quality

Here is the trap that separates people who've shipped models from people who've only shipped services. **Shadow mode — and to a large degree the canary's automated gates — catch operational regressions, not quality regressions.** They can prove the candidate is fast, doesn't error, returns a sane score distribution, and agrees or disagrees with the champion. They *cannot* prove the candidate is more *accurate*, because **there is no ground truth at serving time.** When the fraud model scores a transaction, you don't yet know if it was fraud. When the recommender ranks items, you don't yet know what the user would have clicked. Prediction *agreement* with the champion is not correctness — if both models are wrong the same way, they agree perfectly.

So progressive delivery is necessary but not sufficient, and it must be paired on both sides:

1. **Offline eval before you ever shadow.** The frozen held-out eval set and metric gate (the eval-driven-development discipline the course teaches) is where you establish the candidate is *plausibly better* on labeled data. Shadow and canary then check that this offline win *survives contact with production*. Deploying a model you haven't offline-evaluated and hoping shadow catches quality problems is backwards — shadow can't see quality.
2. **Delayed-label monitoring after you promote.** Ground truth arrives late — the chargeback lands in 30 days, the churn label in a quarter, the human review tomorrow. You must log every served prediction with its correlation ID and *join it to the label when it arrives*, then compute the real quality metric on that delayed-label stream. This is the monitoring loop the course covers, and it is the only place true post-deployment accuracy is measured. It's also what should *trigger* the next retrain when it drifts.

Put together, the full safe-release pipeline is a relay: **offline eval → shadow (operational + skew) → canary/progressive rollout (SLO + fast business proxies) → delayed-label monitoring (true quality) → drift-triggered retrain.** Each stage catches a class of failure the others are blind to. Shadow's zero-risk operational check is a powerful link in that chain, but a team that treats it as the *whole* safety story ships accurate-looking, actually-worse models with great uptime.

---

## Exercises

1. Take a tier-2 model and stand up a v2 candidate in shadow using Istio `mirrorPercentage`. Confirm the shadow's response is discarded and its latency never touches the user path. Deliberately add a side effect (a DB write) to the shadow service and observe the double-execution, then fix it to read-only.
2. Build the comparison pipeline: log champion and candidate predictions keyed by correlation ID, then compute prediction agreement, confidence-gap on disagreements, and KL divergence of the score distributions. Introduce a training-serving skew (drop a feature in the online store) and confirm the pipeline flags it.
3. Configure an Argo Rollouts (or Flagger) canary with a two-step weight ladder gated on a Prometheus `AnalysisTemplate` that checks both success rate and P95 latency. Break the candidate's latency and confirm the rollout aborts and rolls back automatically.
4. Add a model-quality proxy to the SLO gate (prediction-distribution drift vs the stable version). Explain in writing why this gate still cannot certify accuracy, and sketch the delayed-label join that eventually can.

---
## You can now

- Place any release strategy — recreate, rolling, blue/green, canary, shadow — on the deployment-safety spectrum, and explain why the mature ML pipeline sequences shadow → canary → progressive ramp rather than picking one.
- Implement shadow (mirror) deployment with mesh-level traffic mirroring, keep the shadow path fire-and-forget and side-effect-free, and enumerate what to measure: prediction agreement, score-distribution divergence, latency, error rate, and feature completeness.
- Distinguish A/B testing from interleaving for models, tie both to the course's experimentation material, and explain why interleaving is a high-sensitivity screening stage that prunes candidates for a confirming A/B test rather than replacing it.
- Configure automated progressive rollout with Argo Rollouts or Flagger, gate each ramp step on SLO-based `AnalysisTemplate` metrics, and get automatic promotion or rollback — including model-quality proxies, not just infra health.
- Articulate the ML-specific catch — shadow and canary catch operational regressions but not quality, because there is no ground truth at serving time — and pair progressive delivery with offline eval before and delayed-label monitoring after to close the loop.
