# Module 09 — Evaluation, Observability & MLOps

## Why this module matters

"How do you know it works?" is the question that separates candidates who have shipped from candidates who have prototyped. In 2026 this module has doubled in importance: classical ML monitoring (drift, A/B tests) is now table stakes, and on top of it sits the newer discipline of **LLM evaluation** — golden sets, LLM-as-judge with known biases, eval-driven development — which interviewers probe aggressively because most teams are bad at it.

## 1. Offline evaluation done properly

- **Splits that respect reality:** temporal splits for anything time-dependent (random splits leak the future); group-aware splits (no user/document in both train and test); evaluate on the *deployment artifact* (the quantized, prompted, full-pipeline system — not the bare BF16 checkpoint).
- **Slice everything:** aggregate metrics hide regressions; report by segment (language, length, user cohort, document type, difficulty). A model that's +2% overall and −15% on a key segment is usually a launch blocker.
- **Golden/regression sets:** a versioned, hand-audited suite encoding "what good looks like," including past production failures (every incident becomes a test case), edge cases, and out-of-scope inputs that should trigger abstention. Treat it like a test suite: versioned, reviewed, gated in CI.

## 2. LLM evaluation

- **Benchmarks ≠ product evals.** Public benchmarks (MMLU-class, SWE-bench, τ-bench) rank base-model capability and are contaminated/saturating; your product needs **task-specific evals on your distribution**. The 2026 norm is **eval-driven development**: write the eval before (or with) the feature, iterate prompts/models against it, and gate releases on it — evals are to AI engineering what tests are to software engineering.
- **LLM-as-judge:** scalable grading of outputs against rubrics or references; pairwise comparison is more reliable than absolute scoring. Known biases you must name and mitigate: **position bias** (favors first answer — randomize order and average), **verbosity bias** (favors longer — control length or instruct against it), **self-preference** (favors its own family's style — use a different judge model than the generator), rubric drift, and sycophancy toward confident tone. Non-negotiable discipline: **calibrate the judge against a human-labeled subset** (a few hundred examples), report judge-human agreement (e.g., Cohen's κ), and re-calibrate when prompts/models change. A judge you haven't validated is a random number generator with confidence.
- **Stochastic systems need repeated runs:** report variance; for agents use pass@k and pass^k (see the agentic-systems chapter).

## 3. Online evaluation

- **A/B testing fundamentals:** randomize at the right unit (user, not request, when treatment effects persist); pre-compute **power** (minimum detectable effect vs sample size — underpowered tests that "showed nothing" are the most common experimentation sin); guardrail metrics (latency, cost, complaint rate) evaluated alongside the success metric; run full business cycles (weekday/weekend effects); beware peeking (sequential testing methods if you must monitor continuously) and Simpson's-paradox segment effects.
- **Interleaving** for ranking systems: blend results from two rankers in one list, attribute clicks — vastly more sensitive than A/B for ranking changes (needs orders of magnitude less traffic), used as a fast pre-filter before promotion to A/B.
- **Progressive delivery:** offline gates → **shadow mode** (new model scores live traffic, predictions logged not served — catches skew/latency/crash issues with zero user risk) → **canary** (1–5% of traffic, auto-rollback on guardrail breach) → ramp. For LLM products add **holdback cohorts** (long-term quality/retention effects invisible at two-week A/B horizons).
- **Bandits** when experimentation cost is high and the metric is fast (headline/creative selection); classic A/B when effects are slow or you need clean inference.

## 4. Monitoring & observability

- **Classical model monitoring:** input drift (PSI/KL per feature against training snapshots), prediction-distribution drift, *delayed* performance metrics joined when labels mature, feature-pipeline health (null rates, freshness lag — most "model" incidents are upstream data incidents), and training-serving skew checks (the log-and-wait pattern from the data-engineering chapter makes this measurable).
- **LLM/agent observability:** **tracing** is the foundation — every request captures the full tree: prompt version, retrieved context, tool calls with inputs/outputs, tokens and cost per step, latency per span; OpenTelemetry's GenAI semantic conventions are emerging as the standard, with Langfuse / Arize Phoenix / Braintrust-class tools built on it. On top of traces: online quality sampling (judge a % of production outputs continuously), user-feedback capture (thumbs, edits, regenerations — *implicit* signals like "user edited the draft heavily" are richer than thumbs), cost/token dashboards per feature, and **drift in the input distribution of prompts** (new user intents are product signals, not just anomalies).
- **The flywheel, operationalized:** production traces → failure mining (low judge scores, negative feedback, escalations) → labeled and added to golden sets → fixes regression-gated → redeploy. Teams that wire this loop improve weekly; teams that don't, plateau — this is the single most repeatable "senior" observation in the module.

## 5. Incident response for ML systems

"Accuracy dropped but infra metrics look normal" is the most disorienting ML incident class because the monitoring system is working, the servers are up, and yet the product is broken. The failure is invisible to standard SRE tooling. Ordered triage:

**(1) Data in — look here first.** The majority of "model incidents" are actually data incidents. Check: schema drift (a new field was added or a type changed upstream); null or default-value spikes in features the model relies on; upstream backfills (a data warehouse job rewrote historical rows, breaking point-in-time joins); or a new data source going live that shifted the feature distribution without anyone changing the model. Query your feature store or raw log counts before touching model artifacts.

**(2) Features — train/serve skew.** Did the feature engineering logic change in the serving path without a corresponding training rerun? Common sources: a library version bump changed tokenization or normalization; a feature was deprecated in the offline pipeline but the serving code still reads a stale version; timestamp or timezone handling differs between training and serving. Compare the feature value distributions logged at serving time against the training-data snapshot — this diff is the skew monitor from the data-engineering chapter.

**(3) Model artifact — wrong version, silent quant regression, tokenizer mismatch.** Confirm the deployed artifact hash matches the intended checkpoint. Silent quantization regressions happen when a new serving engine version changes the quantization kernel for a given model without surfacing it as a config change. Tokenizer mismatches (a new tokenizer shipped with the model weights but the serving config pins the old one) produce subtly wrong outputs that score fine on perplexity but fail on task metrics. Run the golden set against the deployed artifact, not just the training checkpoint.

**(4) Serving config — truncation, temperature, prompt template change.** Check: was `max_tokens` reduced (truncated outputs fail silently); did temperature or top-p change (affects output distribution, not perplexity); was the system prompt or few-shot template modified? Prompt changes are the most common unreviewed config change at companies without a prompt registry — they go in as a one-line edit and nobody treats them as a model deploy.

**(5) Eval itself — broken golden set, judge drift.** Before concluding the model regressed, verify the eval is still valid: did someone edit the golden set and introduce a scoring bug? Did a model-API version change affect your LLM judge's scoring distribution? Run the judge on a known-good snapshot to confirm it still produces the expected scores. "The eval broke, not the model" is humbling but not uncommon.

**Failure archetypes table:**

| Symptom | Most likely cause | First check |
| --- | --- | --- |
| Accuracy drop, all segments | Data schema change or backfill | Feature null rates, upstream schema diff |
| Accuracy drop, one segment | Distribution shift or feature bug for that segment | Slice feature distributions vs training |
| Gradual accuracy decay over days | Model drift or slow data degradation | Feature PSI trend, label distribution |
| Sudden accuracy drop post-deploy | Wrong artifact, config change, tokenizer mismatch | Artifact hash, serving config diff |
| Eval score dropped, prod looks fine | Broken golden set or judge drift | Replay judge on known-good sample |
| Latency up, accuracy down | Truncation (max_tokens too low) or serving timeout | Response length distribution, timeout logs |

**Postmortem discipline.** Every ML incident that reaches users should produce a postmortem with: root cause in the triage taxonomy above; detection lag (how long from incident start to detection — this is the primary metric for improving your monitoring); the monitoring gap that allowed it; and a concrete mitigation added to the eval or monitoring system. The mitigation is the regression test: every incident becomes a golden-set entry or a monitor alert. Teams that skip this plateau; teams that wire it improve monotonically.

**Two practices worth adopting:** *time-travel evaluation* — replay production traffic against a new candidate at the moment of a past incident, useful for validating that a fix would have caught the regression; and *joint validation pipelines* — stage-validate model + feature + data changes together before any component is promoted, preventing the "individually tested, jointly broken" failure class.

## 6. CI/CD for models (MLOps in 2026 terms)

- **Registry & lineage:** every artifact versioned with its data snapshot, code, config, and eval report (model cards); promotion is a *recorded decision* with eval evidence attached.
- **Pipelines:** training as reproducible, scheduled/triggered DAGs (Airflow/Dagster/Flyte lineage); **retraining triggers** — calendar, drift-threshold, or label-volume — with auto-eval against the champion (champion/challenger) before any promotion.
- **For LLM apps,** treat prompts as first-class versioned software artifacts — the concrete workflow is described in the subsection below.

### 6a. Prompts as code — the concrete workflow

A prompt is not a config value tucked in a string constant. It is a software artifact that determines model behavior, affects output quality, shifts token counts (and therefore cost and latency), and must be reviewed, tested, and deployed with the same discipline as application code. Treating prompts as software engineering artifacts is now standard practice on mature teams.

**The prompt registry.** Every prompt lives in a versioned store — a dedicated prompt-management tool or a home-built table in your feature store — with: a unique name + semantic version, the prompt text, the model and parameter config it was tested with, a pointer to the eval suite it was gated against, and a changelog entry. Retrieval at serving time is by name + version (pinned) or name + "production" alias (the promoted version). The alias approach gives you instant rollback: flip the alias, no redeploy.

**Eval-gated promotion.** The promotion flow: a developer opens a PR changing a prompt → CI runs the registered eval suite against the new prompt version → a score gate enforces a minimum quality threshold (e.g., faithfulness ≥ 0.85, pass@1 ≥ 0.72 on golden set) and a cost gate (prompt change must not increase estimated cost/request by more than X%) → only if both gates pass does the PR merge and the new version become the candidate for promotion. The cost gate is important and usually missing: a prompt that adds 200 tokens of few-shot examples might improve quality by 3% but increase cost by 15% — that is a product decision, not an engineering accident.

**A/B at the prompt level.** After gating, a new prompt version is promoted as a challenger alongside the champion. Traffic is split at the prompt-alias resolution layer — 5% of requests resolve to the new version; the rest to the current champion. Online metrics (task success rate, judge scores sampled on production traffic, user feedback signals) accumulate for the challenger. Promotion to champion requires meeting the online bar, same as any model A/B.

**Non-engineer contribution path.** Most companies have domain experts (product managers, customer success, clinical staff) who understand the task better than engineers but cannot open PRs. The prompt registry enables them: a web UI wraps the registry, allowing non-engineers to draft prompt variants, which then enter the same eval-gate CI pipeline. The gate is the safety net that prevents untested prompts from reaching production regardless of who authored them.

**Ecosystem note.** The tooling here consolidates quickly, and early standalone prompt-management products have not all survived. The durable lesson: prompt management is now a feature of broader observability platforms, not a standalone product category — choose a tool with a strong underlying data model, not just a nice UI, and be prepared to migrate if a niche vendor disappears.

**Pin model API versions explicitly.** Silent upstream model updates are a documented incident class — a provider promotes a new model version under the same name and your prompt behavior changes without any code change on your end. Always pin to an explicit version string (e.g., `gpt-4o-2024-11-20`, not `gpt-4o`) and treat version bumps as a prompt-change deploy: run the eval suite, gate, A/B, promote. Keep rollback one click away — the cheapest reliability feature in existence.

## 7. Operations

### Alert thresholds

ML alerts have a fundamentally different signal-to-noise profile than infra alerts: they fire on degradation, not outage, which means higher false-positive rates and more ambiguous signals. Counter that with tiering — tier 1 (page immediately) for clear pipeline failures and acute performance regressions; tier 2 (ticket, next business hour) for slow drift and model anomalies that require investigation before action. Every threshold below is a calibrated starting point for a medium-traffic production system; tune for your traffic volume, seasonality, and risk tolerance before treating any of these as gospel.

| Signal | Alert threshold | Tier | First action |
| --- | --- | --- | --- |
| Feature null rate | > 5% on any critical feature | 1 | Upstream data triage — pipeline health, schema diff |
| Prediction-distribution PSI | > 0.1 vs 7-day baseline | 2 | Score histogram by segment; may be benign distribution shift |
| Score at hard ceiling | > 0.95 for > 5% of requests | 2 | Possible model collapse or score saturation — check artifact hash, max_tokens |
| Latency p99 | > 2× 7-day baseline | 1 | Serving config, context-length growth, fallback path |
| Cost / request | > 2× 7-day baseline | 2 | Log prompt + response token counts; likely prompt-template or input-length change |
| KV-cache hit rate (LLM/agent) | < 0.40 | 2 | System-prompt prefix likely changed; review prefix caching config |

Two mechanics prevent alert fatigue at scale: **hysteresis** (fire after N consecutive windows breach the threshold, not a single data point — a single spike is noise) and **ownership** (every alert names a team and links a runbook; an unowned alert that fires weekly is a policy failure, not an engineering gap).

### Runbook template

Copy this into your team wiki and fill bracketed fields when wiring a new alert. Triage steps intentionally reference the five-step sequence in the incident-response section (§5) above — the taxonomy already exists; a runbook that restates it is a runbook that will drift out of sync.

```text
RUNBOOK: [Alert name]
Severity: [Tier 1 — page immediately / Tier 2 — ticket, next business hour]
Owner: [Team or rotation alias]

1. CONFIRM. Query the raw metric independently; compare to a 24h baseline. If the
   monitoring system itself is broken, escalate to infra on-call and stop here.

2. LOCALIZE. Follow the triage taxonomy in §5 in order:
     (a) Data in — null spikes, schema drift, upstream backfills
     (b) Feature / train-serve skew — serving vs training distribution diff
     (c) Artifact — confirm deployed hash, tokenizer version
     (d) Serving config — max_tokens, temperature, prompt template changelog
     (e) Eval — verify golden set and judge are not the signal source

3. BLAST RADIUS. What % of traffic is affected? Which segments?

4. MITIGATE. If root cause is unclear and blast radius is growing, rollback the
   most recent deploy (model, prompt, or feature pipeline). Rollback buys time;
   investigate post-stabilization.

5. DOCUMENT. Open a postmortem ticket immediately — a stub is fine. Record the
   current time as the detection timestamp.

6. RESOLVE. Confirm metrics return to baseline for two consecutive alert windows
   before marking closed.

7. POST-INCIDENT. Add root cause as a golden-set entry or a tighter threshold.
   Every incident that reaches users should produce one concrete monitoring artifact.
```

### Postmortem template

The postmortem discipline in §5 describes what to capture; this is the structured form to make that discipline repeatable. Complete within 48 hours of incident close while memory is fresh.

```text
POSTMORTEM: [Title]   Date: [ISO]   Severity: [P1 / P2 / P3]
Author: [Name]   Reviewers: [Names]

SUMMARY
[One paragraph: what broke, user impact, resolution. Readable by a non-ML person.]

TIMELINE (UTC)
[HH:MM] — [Event]
[HH:MM] — [Event]

ROOT CAUSE
[Map to triage taxonomy in §5: data / feature-skew / artifact / serving-config / eval.
Explain the specific mechanism — not just the category.]

IMPACT
  Users affected: [N or %, with segment breakdown if available]
  Duration: [HH:MM]
  Business metric: [recall drop / complaint spike / revenue effect if measurable]

  MTTD (incident start → first alert fire): [HH:MM]
  MTTR (first alert acknowledgment → resolution): [HH:MM]

WHAT WENT WELL
  -

WHAT WENT WRONG
  -

ACTION ITEMS
  | Item | Owner | Due | Type [monitor / golden-set / process / infra] |
  |------|-------|-----|-----------------------------------------------|
  |      |       |     |                                               |
```

Mature systems target MTTD < 30 min for tier-1 alerts and MTTR < 2 hours. If MTTD persistently misses target, the monitoring gap is the engineering priority — not the fix for the incident that revealed it.

### Dashboard and metric export

A Prometheus/Grafana stack (or DataDog with custom ML metrics) needs five ML-specific panels beyond standard infra: **feature null rate by feature name**, **prediction score histogram** (the full distribution, not just mean — collapse and calibration drift appear here before they show up in aggregate metrics), **p50/p95/p99 request latency**, **cost or token count per request** (for LLM systems), and **golden-set pass rate** exported from the latest CI eval run as a time-series metric. Without the score histogram you are flying blind on model collapse.

Exporting a custom metric from a serving process is minimal:

```python
from prometheus_client import Gauge
import logging

log = logging.getLogger(__name__)

_feature_null_rate = Gauge(
    "ml_feature_null_rate",
    "Fraction of serving requests where a feature was null",
    labelnames=["feature_name"],
)

def record_feature_nulls(features: dict[str, float | None]) -> None:
    """Call once per request, immediately after feature fetch."""
    for name, value in features.items():
        is_null = 1.0 if value is None else 0.0
        _feature_null_rate.labels(feature_name=name).set(is_null)
        if is_null:
            log.warning("Null feature at serving time: %s", name)
```

Grafana averages this over a rolling window; alert at > 0.05 on any feature marked critical in your feature registry. For large feature sets, emit only the top-N features by model importance to keep Prometheus cardinality manageable.

### On-call reality

ML on-call pages are rarer than infra pages but harder — "something is subtly wrong" demands more diagnosis before a mitigation path is clear, and the blast radius can be large before anyone notices. Practices that distinguish functional from dysfunctional ML on-call:

**Rotation depth.** Two or three engineers who know the runbook through personal incident experience — primary takes the page, secondary covers escalation and backup. A full-team round-robin guarantees someone on-call who has never debugged the model in production; that is not rotation coverage, it is a lottery. Depth beats breadth.

**Tiered escalation.** Tier-1 pages primary immediately. Tier-2 creates a ticket; primary triages at the next business hour. If tier-1 is unresolved in 30 minutes, page secondary. At 90 minutes, escalate to domain experts (data team, model owner, ML platform team) and engineering leadership simultaneously — "when to escalate" should be a calendar rule, not a judgment call made under production pressure.

**Pager-load hygiene.** If tier-1 fires more than roughly once per week on average, thresholds are miscalibrated or the system has a structural reliability problem that on-call cannot paper over. Track weekly alert volume as an engineering-health metric; include it in sprint retrospectives. Chronic pager load is a retention risk before it becomes an SLA risk.

**Knowledge transfer.** Runbooks and postmortems are the asset that survives individual team members rotating off. New engineers shadow at least two live incidents before taking primary. When a senior engineer leaves, runbook review should be on the offboarding checklist alongside code review.

In an interview, mentioning on-call rotation depth and pager-load hygiene alongside alert thresholds is a reliable signal that you have operated a production ML system — not just designed one.

## References

- The load-bearing discipline is eval-driven development: write the eval before the feature, gate releases on it, and treat the golden set like a versioned test suite. Everything else in this chapter supports that loop.
- The catalog of LLM-as-judge biases (position, verbosity, self-preference, sycophancy) and the non-negotiable practice of calibrating the judge against human labels are the two facts most teams get wrong.
- The online-experiment fundamentals — randomization unit, statistical power, guardrail metrics, interleaving for ranking, progressive delivery — are a mature, well-documented body of practice; study them until the power-analysis math is automatic.
- OpenTelemetry's GenAI semantic conventions are emerging as the tracing standard; reading the data model of a mature open-source tracing/eval platform teaches the observability discipline faster than any prose.

## Project 09 — Instrument and gate a real pipeline

Take the RAG system from the retrieval chapter (or the agent from the agentic-systems chapter). (1) **Tracing:** instrument it end-to-end with an open-source tracing platform — every retrieval, prompt, generation as spans with token/cost metadata. (2) **Eval CI:** wire your golden set into a GitHub Action that runs on every prompt/config change and fails the build if faithfulness or recall drops >2 points vs the registered champion; make a deliberately bad prompt change and watch the gate catch it. (3) **Judge calibration:** hand-label 100 outputs for faithfulness; measure your LLM judge's agreement (κ), then improve the judge prompt and re-measure — document the before/after. (4) **Drift sim:** replay a shifted query distribution (new topic mix) and build the monitor that detects it from embedding-cluster drift in traces. (5) **A/B math:** given baseline 4% thumbs-up rate, compute required sample size to detect a relative 10% lift at 80% power, and write the one-pager experiment design (unit, duration, guardrails). This project converts "I know about evals" into demonstrated infrastructure skill.

## Interview Q&A

**Q1. Design the evaluation strategy for an LLM feature that summarizes legal contracts.**
**A.** High-stakes domain → layered strategy. **Offline:** a golden set of contracts spanning types/lengths/jurisdictions, each with attorney-written reference summaries and, critically, a checklist of must-capture items (parties, term, termination, liability caps, unusual clauses). Metrics: coverage of must-capture items (programmatic + judge), **faithfulness** (every claim traceable to the source — the hallucination metric, judged claim-by-claim, calibrated against attorney labels with reported agreement), and a hard gate on fabricated-clause rate ≈ 0. Slice by contract type and length — long-document degradation is the expected failure. **Pre-launch human eval:** attorneys grade a sample blind against the current manual process. **Online:** the product's implicit signals — edit distance between draft and what the lawyer actually sends/files, time-to-review vs baseline, escalation/complaint rates — plus continuous judge-sampling of production outputs and an error-reporting affordance feeding the golden set. **Process:** every prompt/model change passes the eval gate in CI; canary by customer cohort; and an honest statement of residual risk: this is a review-assist tool with human-in-the-loop by design, and the eval program is what justifies that positioning.

**Q2. What's wrong with "we asked GPT to score outputs 1–10 and the average went up"?**
**A.** Nearly everything, and enumerating it is the answer: (1) **uncalibrated judge** — no measured agreement with human judgment, so the scale's meaning is unknown; absolute scoring is the weakest judge mode (poor inter-run consistency); pairwise champion-vs-challenger with randomized positions is strictly better. (2) **Known biases unmitigated** — verbosity (did outputs just get longer?), self-preference (same model family generating and judging?), position effects if comparative. (3) **No variance treatment** — single stochastic runs, no confidence intervals; is +0.3 noise? (4) **Distribution validity** — scored on what? If the eval set isn't representative (or the team iterated *against* the judge, Goodharting it), the number is decorative. (5) **No slices** — averages hide segment regressions. Minimum credible version: fixed versioned eval set mirroring production traffic; pairwise judging with position randomization; judge calibrated against ≥100 human labels with reported κ; multiple sampled runs with CIs; slice breakdown; and a held-out human-graded subset as the final arbiter for launch decisions.

**Q3. Your new ranking model wins offline (+3% nDCG) but the A/B shows flat engagement. What do you investigate?**
**A.** Ordered checklist. (1) **Experiment validity:** power analysis — was the test even capable of detecting the expected lift (offline nDCG gains translate to small online effects; underpowered flat ≠ no effect)? Randomization unit correct, sample-ratio mismatch, full business cycles, novelty effects decaying? (2) **Serving parity:** is the online model actually the offline model — same features (point-in-time skew!), same preprocessing, quantization applied online but not in eval, latency timeouts silently falling back to the old ranker for a fraction of traffic? Shadow-mode score comparison (offline-predicted vs live-served scores per request) pinpoints this fast. (3) **Metric mapping:** nDCG against *logged* labels measures agreement with historical clicks under the old ranker's exposure bias — the new model may rank well on a counterfactual the logs can't credit (the off-policy evaluation gap); interleaving would detect ranking improvement with far more sensitivity than the A/B. (4) **Funnel position:** a better ranker can be neutralized downstream by the re-rank/policy layer or upstream by candidate generation already saturating quality. (5) **Heterogeneity:** flat average can hide +5% on one cohort, −5% on another — slice the experiment. The senior framing: treat offline-online divergence as a measurement-system bug to be localized, not a verdict on the model.

**Q4. How do you monitor a fraud model whose labels arrive 60 days late?**
**A.** You can't wait 60 days to notice a problem, so build leading indicators in layers: (1) **input monitoring** — per-feature drift (PSI) against the training snapshot, plus upstream pipeline health (null spikes, freshness lag) since data breakage precedes most model incidents; (2) **score-distribution monitoring** — shifts in the score histogram, flag rates, and approval rates by segment; a sudden calibration shift in score deciles is the earliest model-level alarm; (3) **fast proxy labels** — a subset of outcomes arrives quickly (manual-review verdicts within days, customer fraud reports, issuer declines) — monitor precision on the reviewed sample continuously and treat review-queue precision as the canary metric; (4) **matured-label backfill** — as chargebacks land, compute true recall/precision on 60-day-old cohorts and trend it (you're always grading the model's past, which still catches slow decay and validates the proxies); (5) **adversarial drift posture** — fraud shifts because attackers adapt: champion/challenger retraining on a frequent cadence, alerting on novel-pattern clusters (e.g., embedding-space outlier rates), and a kill-switch to a conservative rules fallback. Tie it together with an honest dashboard that displays each metric *with its label-maturity horizon* so nobody mistakes 60-day-old recall for current performance.

**Q5. Shadow deployment vs canary — when each, and what does each catch?**
**A.** **Shadow:** the challenger receives mirrored live traffic and produces predictions that are logged, never served. Zero user risk; catches systems issues (latency under real load, crashes, feature-fetch failures, skew between offline eval scores and live scores) and lets you compare champion-vs-challenger outputs on identical real inputs. What it *cannot* measure: anything requiring user reaction — engagement, feedback loops, downstream behavior change — because no one sees its outputs. **Canary:** serve the challenger to a small real slice (1–5%) with automated guardrail monitoring and instant rollback; this is the first measurement of true user impact and operational behavior under genuine consequence, at bounded blast radius. They're sequential, not alternatives: offline gate → shadow (de-risk systems + skew) → canary (de-risk users) → A/B at power (measure the effect) → ramp. Special cases worth adding: for state-mutating systems (fraud blocking, refunds, agents with side effects), shadow requires care — you must *simulate* the action, not execute it; and for LLM products, "shadow" often means generating responses on logged prompts and judge-scoring them, which is exactly the offline-eval-on-production-distribution pattern from the flywheel.
