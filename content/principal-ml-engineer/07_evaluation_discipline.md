# Module 07 — Evaluation as an Org Discipline

## Why this module matters

Staff engineers build models; principals define what "good" means and hold the organization to it. An org shipping six models a quarter without an evaluation standard is not doing science — it is doing vibes with dashboards, and every quarter it burns experiment capacity on deltas that are noise, ships regressions labeled as wins, and slowly loses the ability to know whether its ML investment is working at all. Evaluation discipline is the highest-leverage standard a principal can install because it compounds: every future launch, every migration (Module 08), every build-vs-buy bake-off (Module 09) inherits it. This module covers the eval hierarchy, the statistics the field routinely skips, offline-online correlation as a first-class project, LLM eval infrastructure, and the launch-gate policy you will write and enforce.

## 1. The job: you own the definition of "good"

At senior level, your eval question is "does my model beat the baseline?" At principal level it is "does this organization have a trustworthy, cheap, fast way to answer that question for *every* model, and does anyone check that the answer predicts business outcomes?" Concretely, the principal owns four artifacts:

1. **The eval standard** — a written policy defining what evidence is required before any model reaches production, at what rigor, reviewed by whom.
2. **The eval infrastructure roadmap** — golden sets, regression suites, judge pipelines, experiment tooling; funded like product infrastructure, not as a side quest.
3. **The offline-online correlation ledger** — periodic evidence that the offline metrics teams optimize actually predict the online metrics the business pays for.
4. **The enforcement mechanism** — launch review where the standard has teeth. A standard nobody blocks a launch over is a blog post.

The failure mode to internalize: every team, left alone, builds the eval that makes its model look good. Not out of dishonesty — out of local incentives. The principal is the only person positioned to impose a shared definition of good, because they review across teams and answer to the business, not to any one model's success. This is influence work as much as technical work: the mechanics are covered in the ML System Design course; here we focus on the org-level standard.

## 2. The eval hierarchy — what each layer proves and cannot prove

Every evaluation activity in an ML org sits on one of five rungs. Each rung answers a different question, at a different cost and latency, with a different failure mode. Confusing the rungs — treating an offline benchmark win as proof of business value — is the single most common eval error at the org level.

**Rung 1: Unit evals (seconds, ~free).** Deterministic checks on individual behaviors: "given this input, the extractor returns this field," "the classifier never assigns `approved` when income is null," "the LLM refuses this jailbreak." Proves: specific behaviors are present; regressions on known cases are caught. Cannot prove: aggregate quality, generalization. These belong in CI, run on every model artifact and every prompt change, and fail the build. A team with 400 unit evals and no benchmark is better protected than a team with a benchmark and no unit evals, because most production incidents are specific behaviors regressing, not aggregate quality collapsing.

**Rung 2: Offline benchmark (minutes–hours, cheap).** A frozen, versioned test set scored with automatic metrics (AUC, nDCG, exact-match, judge score). Proves: aggregate quality on the *sampled distribution*, comparable across model versions. Cannot prove: performance on live traffic (distribution shift, candidate-set skew — see the Module 01 war story in the ML System Design course), business impact, or behavior under the feedback loop. The benchmark's authority decays as production drifts away from the sample; re-sampling cadence is a governance decision (section 7).

**Rung 3: Shadow / backtest (hours–days, moderate).** Score live traffic with the candidate model without acting on its outputs, or replay historical decisions. Proves: the model runs at production scale, on production inputs, with production features; prediction distributions and latency are observable before any user sees them. Cannot prove: outcome impact — shadow mode never observes counterfactual user behavior, and backtests inherit the logging policy of the old model (you only have labels for what the old system chose to do; off-policy correction helps but has variance that explodes exactly where the models disagree most, which is exactly where you care).

**Rung 4: A/B experiment (weeks, expensive).** Randomized controlled exposure. Proves: causal effect on measured online metrics, within the experiment's power, duration, and population. Cannot prove: long-term effects beyond the window (novelty, ecosystem shifts), effects on unmeasured metrics, or effects under full rollout when interference exists (section 8).

**Rung 5: Business metric (quarters, the whole point).** Revenue, retention, cost per resolution, loss rate. Proves: whether the org's ML investment matters. Cannot prove: which model caused what — attribution at this rung is confounded by everything. The connection between rung 4 and rung 5 is itself a modeling exercise (metric hierarchies, holdback experiments).

| Rung | Latency | Cost | Proves | Cannot prove |
|---|---|---|---|---|
| 1 Unit evals | seconds | ~free | specific behaviors present; known regressions caught | aggregate quality |
| 2 Offline benchmark | min–hrs | cheap | aggregate quality on the sampled distribution | live-traffic performance, business impact |
| 3 Shadow / backtest | hrs–days | moderate | runs at scale on production inputs; distributions observable | outcome impact (no counterfactual) |
| 4 A/B experiment | weeks | expensive | causal effect within power, duration, population | long-term effects, unmeasured metrics, interference-free rollout |
| 5 Business metric | quarters | the point | whether the ML investment matters | attribution to any one model |

The hierarchy's operational meaning: **evidence flows upward, cheaply-caught failures flow downward.** A launch should climb the rungs in order, and the org should invest so that failures are caught at the cheapest rung possible. Every incident postmortem should ask: "at which rung *could* this have been caught, and why wasn't it?"

## 3. Statistical rigor the field skips

Most ML orgs run experiments with the statistical sophistication of a 1990s direct-mail campaign, and it costs them real money. Three practices to mandate:

**Power analysis before trusting a delta.** Before a team celebrates "+0.8% CTR," someone must ask whether the experiment could have detected +0.8% in the first place. The back-of-envelope for a proportion metric:

```text
n_per_arm ≈ 16 × p(1−p) / MDE²        (α=0.05, power=0.8, two-sided)

Example: baseline CTR p = 0.05, minimum detectable effect = 2% relative
         → MDE_abs = 0.001
         n ≈ 16 × 0.0475 / 0.000001 ≈ 760 000 users per arm
```

A product surface with 200k weekly actives cannot detect a 2% relative CTR lift in a week, period. Teams in that position must either run longer, use variance reduction (CUPED-style covariate adjustment routinely cuts required n by 30–50%), pick a more sensitive proxy metric, or stop pretending. This is not exotic: a 2024 survey of AI evaluations against human baselines found only about 2% performed a power analysis. The field's default is to be underpowered and not know it — which produces a literature (and an org history) of unreplicable wins. Your standard should require a pre-registered MDE and the implied sample size *before* the experiment starts.

**Interval estimates, not point estimates.** "+0.8% CTR" must always ship with its confidence interval. "+0.8% [−0.1%, +1.7%]" and "+0.8% [+0.6%, +1.0%]" are different business situations that identical dashboards render identically. For offline metrics, bootstrap the test set — it is one function, and installing it in the org's shared eval library is a one-week project that upgrades every dashboard at once:

```python
def paired_bootstrap_delta(metric_fn, preds_a, preds_b, labels, n_boot=2000, seed=0):
    """CI on metric(B) - metric(A), resampling EXAMPLES (paired)."""
    rng = np.random.default_rng(seed)
    n, deltas = len(labels), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)                      # resample with replacement
        deltas.append(metric_fn(labels[idx], preds_b[idx])
                      - metric_fn(labels[idx], preds_a[idx]))
    return np.percentile(deltas, [2.5, 97.5])
```

This converts "the new model is +0.4 AUC points better" into "the delta is +0.4 [−0.3, +1.1], i.e., we know nothing." The pairing matters: bootstrapping the *delta* on the same resampled examples typically tightens the interval 2–5× versus bootstrapping each model separately, because example difficulty is shared and cancels. Two implementation notes: resample at the *user* level, not the example level, when examples cluster within users (otherwise the interval is falsely narrow); and for ranking metrics resample queries, not query-document pairs.

**Multiple-comparison hygiene.** An org running 40 experiments a quarter at α=0.05 expects ~2 false-positive "wins" per quarter from pure noise — and because teams ship the wins and quietly drop the losses, the org's shipped-improvement ledger systematically overstates reality (the winner's curse: conditioned on being declared a winner, the measured effect overestimates the true effect). Mitigations that fit real orgs: (1) hold experiment-level α but require *replication* for surprising wins — a second experiment before the result enters the ledger; (2) apply Benjamini-Hochberg within any single launch that reads 20 metrics, so "we moved 1 of 20 guardrails at p=0.04" is recognized as expected noise; (3) maintain a holdback — 1–5% of users kept on the last-quarter system — so the *cumulative* shipped effect is measured directly rather than summed from individually noisy claims. The holdback is the org's audit; teams that sum their claimed A/B wins routinely report 3× the lift the holdback shows.

## 4. Offline-online correlation as an explicit project

The offline benchmark exists to make iteration cheap: teams try ten ideas offline and A/B the best one. That entire economy rests on an assumption almost no org ever tests — that offline metric gains predict online metric gains. Treat testing it as a real project with an owner and a deliverable, refreshed twice a year.

**Design.** For every experiment the org has run in the past 12–18 months, log-and-join: the offline metric delta of the candidate vs. control (computed on the frozen benchmark at launch time) against the measured online metric delta from the A/B. Plot them. Compute the rank correlation. With 25–40 launch points you have enough to say something. Concretely:

```text
OFFLINE-ONLINE CORRELATION STUDY — design
UNIT        one (candidate, control) launch pair; one row per A/B experiment
JOIN KEYS   model version ↔ experiment ID ↔ benchmark version at launch time
X-AXIS      offline delta (benchmark metric, candidate − control), with CI
Y-AXIS      online delta (primary A/B metric), with CI
TAGS        surface (search/recs/ads), intervention type (architecture /
            features / training data / objective), benchmark version
ANALYSIS    Spearman ρ overall + per tag; fraction of offline wins that
            were online wins (sign agreement); the quadrant plot
PITFALLS    survivorship (teams only A/B their best offline candidates —
            note the range restriction; it attenuates ρ downward);
            underpowered A/Bs add y-axis noise — weight by experiment power
DELIVERABLE 1 page: scatter, ρ by tag, keep/recalibrate/kill call per metric
CADENCE     refreshed every 2 quarters; owner named; presented at eng review
```

The survivorship caveat matters when reading the result: because teams pre-filter on offline wins, the observed points cover a narrow offline range, which mechanically attenuates the correlation. A moderate measured ρ on range-restricted data is decent evidence; a near-zero ρ is damning.

**The three outcomes.**
- **Correlated (ρ > ~0.6):** the offline metric is doing its job; keep it, and publish the plot so teams trust the cheap iteration loop.
- **Uncorrelated (ρ ≈ 0):** the offline metric is a treadmill. Kill it or fix the benchmark (usually the sample is stale or the metric ignores the serving-path filters). Killing a metric is a principal-level act — teams have roadmaps and promotions built on it — which is exactly why nobody below principal does it.
- **Correlated until it isn't (piecewise):** common with proxy metrics under optimization pressure. AUC gains predicted CTR gains for two years, then the last three launches show offline gains and flat online results — the metric is saturated or Goodharted. The correlation study catches this a year before anyone would otherwise admit it.

**A real pattern to expect:** offline nDCG gains from *ranking-architecture* changes correlate with CTR; offline nDCG gains from *training-data* changes often don't, because the data changes shift the score distribution in ways the online candidate filter interacts with. The correlation study should therefore tag experiments by intervention type — the aggregate correlation can hide a subgroup where the proxy is broken.

The deliverable is one page: the scatter plot, the correlation by metric and by intervention type, and a recommendation per metric (keep / recalibrate / kill). Present it at the quarterly eng review. This single page changes more roadmaps than any model launch.

## 5. LLM eval infrastructure

GenAI systems break rung 2 of the hierarchy — there is often no automatic metric — so the org must build one. The components, in build order:

**Golden sets.** 200–1 000 curated input cases per product surface, with either reference outputs or graded rubrics, stratified by intent/difficulty/language, sampled from production traffic (and re-sampled quarterly, because production drifts). Golden sets are versioned artifacts with owners, not files in someone's notebook. Cost reality: expert-graded golden sets run $2–10 per example to build; a 500-example set is a $2–5k artifact that will gate dozens of launches — the cheapest infrastructure the org will ever buy.

**LLM-as-judge, treated as a model in production.** A judge is a model whose predictions gate launches; give it the same discipline you give any production model. The onboarding protocol, as a checklist the org can enforce:

```text
JUDGE ONBOARDING — required before any judge score gates a launch
1. Rubric written and reviewed by the domain owner (not the ML team alone)
2. 100–300 production examples double-graded by 2 human experts
   → inter-HUMAN agreement measured first (κ_hh); if κ_hh < 0.6,
     fix the rubric before blaming any model
3. Judge graded on the same set → κ (judge, human consensus) per dimension
   → publish; κ ≥ 0.6 required for gating use, κ ≥ 0.4 for dashboards only
4. Bias battery: position-swap test, length-correlation test,
   self-family preference test → mitigations documented
5. Judge version + prompt pinned (content-hashed); 100-example anchor set
   frozen for drift detection
6. Re-run steps 3–5 on ANY judge model or prompt change; never splice
   scores across judge versions without re-grading the baseline
```

Step 2's ordering is the part teams skip: if two human experts only agree at κ = 0.5, no judge can be "calibrated" against their labels — the rubric is the broken component. The full discipline per component:
- **Human-agreement calibration.** Before trusting a judge, measure its agreement with expert human grades on 100–300 double-graded examples. Report Cohen's κ or correlation, per rubric dimension. κ < 0.6 means the judge (or the rubric) isn't ready; publish the number so teams know how much to trust judge deltas.
- **Judge-drift audits.** The judge model gets upgraded, its provider changes behavior, or your prompt changes — and every historical score silently shifts. Pin the judge version; re-grade a fixed 100-example anchor set on any judge change and publish the before/after; never compare scores across judge versions without re-grading the baseline.
- **Known biases.** Judges prefer longer answers, answers formatted like their own outputs, the first-presented option (position bias), and self-family models. Mitigations: swap positions and average, length-normalize or instruct explicitly, use a judge from a different model family than the candidate.

**Pairwise vs. rubric scoring.** Pairwise ("which of A/B is better?") is more sensitive and more stable than absolute scores — use it for *model selection*. Rubric scoring ("grade 1–5 on faithfulness, completeness, tone") is auditable and tracks *absolute* quality over time — use it for regression gates and dashboards. Mature orgs run both: pairwise to pick candidates, rubric to gate launches. Absolute scores from judges are poorly calibrated (a 4.2 means nothing by itself); rubric scores only become meaningful anchored to the human-agreement study.

**Contamination hygiene.** Golden sets leak: into fine-tuning data (someone exports "hard cases" into the training mix), into prompts (few-shot examples drawn from the eval set), and into the world (public benchmark items appear in pretraining data). Controls: hash-based exclusion of eval examples from every training/few-shot pipeline; a held-out *sealed* subset (never in any dashboard, graded only at launch review) whose score should track the working set — divergence is the contamination alarm; for public benchmarks, assume contamination and weight your private sets accordingly.

## 6. Launch gates: the policy you write and enforce

The eval standard crystallizes into a launch gate: **no model ships without (1) an eval card, (2) a passing regression suite, (3) guardrail metrics wired to the experiment, (4) written rollback criteria.** The principal writes this policy, gets it ratified once by eng leadership, and then enforces it in launch review — which mostly means being willing to say "not yet" to a team with a deadline, twice, until the org learns the gate is real.

The eval card is one page, produced by the launching team, reviewed at launch:

```text
EVAL CARD — <model/system name> v<X.Y>                    owner: <team>
------------------------------------------------------------------------
INTENDED USE     surface, traffic %, user population, decisions it makes
OFFLINE          benchmark version + date; metric deltas WITH bootstrap CIs;
                 per-slice results (min: top-5 segments + known-risk slices)
REGRESSION SUITE unit-eval pass rate (must be 100% on blocking cases);
                 sealed-set score vs. working-set score (contamination check)
SHADOW           days in shadow; prediction-distribution deltas (PSI/KL) vs.
                 incumbent; latency p50/p99; error rate
EXPERIMENT PLAN  primary metric + pre-registered MDE + computed n + duration;
                 guardrails (latency, cost/req, complaint rate, policy-
                 violation rate) with regression thresholds
KNOWN LIMITS     slices where the model underperforms; failure modes; what
                 this model must NOT be used for
ROLLBACK         trigger conditions (metric X drops Y% for Z hours → auto
                 rollback); who owns the pager; time-to-previous-model
------------------------------------------------------------------------
SIGNOFFS         launching team lead / eval owner / (regulated: compliance)
```

The guardrail line deserves expansion, because "guardrail metrics" is where teams under-specify. Five families, and every Tier 1 launch should name at least one from each:

- **Operational:** latency p99, error/timeout rate, cost per request.
- **User-harm:** complaint rate, policy-violation rate, appeal/override rate, refund rate.
- **Ecosystem:** effects on adjacent surfaces (a recs change that cannibalizes search clicks reads as a recs win and a company wash).
- **Distributional:** worst-slice performance, not just the mean — a +2% average that is −8% on one country is a launch decision, not a footnote.
- **Long-term proxies:** retention cohorts, holdback deltas — slow, but the only defense against wins that borrow from the future.

Two design points. First, the card is *evidence*, not prose — every claim is a number with a source. Second, the rollback section is written *before* launch, when heads are cool; mid-incident is the wrong time to decide what "bad enough" means. Modules 10 covers the reliability side; the gate is where eval and reliability meet.

### Running the launch review

The gate lives or dies in a 30-minute meeting, so design the meeting. What works:

- **The card is pre-read; the meeting is questions only.** No presenting. If the card can't stand alone, it isn't done.
- **A standing question list**, asked every time, so teams prepare for it: Which slice is weakest, and why is shipping it anyway acceptable? What does the incumbent do better? If this launch goes wrong, what's the first metric to move, and who sees it? What did the sealed set show versus the working set? When was the benchmark last re-sampled?
- **Three verdicts only:** ship / ship-with-conditions (named, dated, tracked) / not yet (with the specific missing evidence). "Approved with concerns" is not a verdict; it is liability laundering.
- **The reviewer is not the builder's chain.** Cross-team review — a principal or senior staff engineer from outside the launching org — is what keeps the gate honest when the launching team's director wants the launch in this quarter's review packet.
- **Log every verdict.** The review log (launch, verdict, conditions, outcome 90 days later) is how you calibrate the gate itself: if 100% of launches pass on first attempt, the gate is theater; if 50% bounce, the standard is unclear upstream. Healthy orgs run 70–85% first-pass.

Enforcement calibration matters: gate rigor should scale with blast radius. A three-tier policy works — Tier 1 (user-facing, revenue- or safety-touching): full card, shadow mandatory, principal-level review. Tier 2 (internal-facing or <5% traffic): card + regression suite, team-lead review. Tier 3 (offline/analytics models): card on file, no review. Without tiering, the gate becomes friction theater and teams route around it — the platform-adoption lesson of Module 03 applies to standards too.

## 7. Eval set governance

Eval sets are load-bearing shared state; govern them like schema.

- **Frozen means frozen.** A test set used for comparison is immutable and versioned (`benchmark-v7`, content-hashed). Every reported number cites the version.
- **Adding is OK; rebalancing invalidates history.** Appending new examples (as a new version) keeps old comparisons interpretable — you can re-score old models on the new version. *Removing* examples or reweighting classes silently changes what the metric means; every historical number computed on the old composition is now incomparable, and dashboards that splice across the change are lying. If rebalancing is genuinely needed (the old sample no longer resembles production), do it as a **hard epoch break**: new version, re-score the incumbent and last three models on it, restate the baseline, annotate every dashboard.
- **Ownership.** Each benchmark has a named owner responsible for re-sampling cadence, label quality audits (sample 5% annually for label errors — 3–10% label noise is typical and puts a ceiling on measurable model differences), and the contamination checks from section 5.
- **Access discipline.** Teams iterate against a *dev* split; the test split is scored by CI, not by hand, and the sealed subset (section 5) is scored only at launch review. The moment engineers can eyeball test examples while tuning, the benchmark starts measuring memorization of itself.
- **Provenance.** Record where each example came from (traffic sample date, labeler, sampling filter). When a metric moves unexpectedly, the first diagnostic question is "did the data move?" — unanswerable without provenance, a five-minute query with it.

## 8. A/B subtleties at principal level

Rung 4 has failure modes that only become visible at org scale, and they are the ones a principal is expected to catch in review:

**Novelty and primacy effects.** Users click new things because they are new (novelty inflates early treatment metrics) or resist change (primacy deflates them). Both decay over 1–3 weeks. Standard: minimum two-week experiments for UX-visible changes, and read the *time series* of the treatment effect, not just the aggregate — a lift that is monotonically decaying toward zero across the window is a novelty artifact, and shipping it buys nothing.

**Dilution.** If the change only touches 8% of sessions (say, queries where the new retrieval path triggers), measuring CTR over *all* sessions dilutes the effect ~12×, and your power analysis is off by the same factor. Either measure on triggered sessions with counterfactual trigger logging in control (log would-have-triggered without acting), or scale the MDE accordingly. Half the "flat" experiments in a typical org are diluted, not null.

**Interference between concurrent experiments.** Forty experiments a quarter means users sit in many treatments at once. Usually fine (effects are additive to first order), except when experiments share a constrained surface: two ranking experiments on the same slate, two pricing experiments on the same auction, or any experiment pair where treatment A changes the traffic mix that experiment B measures. Standard: a lightweight experiment registry with declared surfaces; same-surface experiments run in orthogonal layers or mutually exclusive slices. Marketplace and social products add a harder version — treatment users deplete shared inventory or shift network behavior, contaminating control (the SUTVA violation) — where cluster- or region-level randomization is the honest, more expensive answer.

**Feedback loops: the model shapes its own future training data.** The deepest one. Today's model decides what gets shown; what gets shown determines what gets clicked; clicks become tomorrow's training data. Consequences: popularity bias compounds; the A/B comparison is unfair to the challenger (trained on data the incumbent's policy generated); and offline backtests systematically favor models similar to the incumbent. Mitigations to mandate as standard: log propensities (or at least a small uniform-random exploration slice, 0.5–1% of traffic, which doubles as unbiased eval data); periodically retrain the incumbent on challenger-collected data before declaring the bake-off done; and treat any "the new model wins offline but keeps losing online to the incumbent" pattern as a possible feedback artifact, not a modeling failure.

## Worked example

**Setting.** You are the principal for ML at a 900-person commerce company: ~80 ML engineers across 7 teams (search ranking, recommendations, ads, fraud, support LLM, pricing, logistics ETA), shipping ~6 models per quarter. Symptoms at your arrival: each team has its own eval scripts; two quarters ago fraud shipped a "win" that the finance team later showed increased chargebacks; the support LLM has no regression suite and broke twice on prompt changes; the experiment dashboard shows 14 claimed CTR wins in 12 months summing to +9%, while topline CTR moved +2.5%.

**Step 1 — Diagnose with the hierarchy.** You map every team's practice against the five rungs. Findings: everyone has rung 2 (benchmarks of varying staleness — ads' is 26 months old); only fraud has rung 3 (backtests, but on labels its own policy generated — the feedback trap of section 8); rungs 1 and the offline-online link (rung 2↔4) exist nowhere. The claimed-wins-vs-holdback gap (+9% vs +2.5%) is the winner's curse plus no holdback — nobody can even measure cumulative effect. You present this one-page diagnosis at eng review. Diagnosis before policy: the standard you're about to write must be seen as the fix to *this*, not as process for its own sake.

**Step 2 — Write the standard (2 pages, ratified by the VP Eng).** Core clauses: (a) every production model has a versioned benchmark with a named owner and ≤12-month re-sample; (b) launch gate per section 6, tiered — fraud, ads, pricing, support LLM are Tier 1; (c) all experiments pre-register primary metric + MDE + duration in the experiment registry; surprising wins (>2× the historical effect size for that surface) replicate before entering the ledger; (d) a 2% quarterly holdback measures cumulative shipped lift; (e) 1% uniform-exploration slice on ranking surfaces; (f) the eval-card template (section 6 code block) is the review artifact. You explicitly do *not* mandate tooling — teams keep their scripts if outputs meet the standard. Standards travel; tool mandates get routed around.

**Step 3 — Fund the infrastructure.** Two engineers for two quarters build: golden-set tooling + judge pipeline for the support LLM (500-example golden set, $4k in expert grading; judge calibrated at κ=0.71 against human grades on 200 double-graded cases, anchor set pinned); a shared bootstrap-CI library so every dashboard delta ships with an interval; the experiment registry with surface declarations. Cost: ~$350k loaded. You defend it to the VP with the fraud incident alone (chargeback impact was $1.2M/quarter).

**Step 4 — The correlation study.** An analyst joins 31 past experiments: offline delta vs. A/B delta. Results: search's nDCG correlates with CTR at ρ=0.74 — publish and celebrate. Recs' AUC correlates at ρ=0.11 — the benchmark predates a major candidate-filter change; the metric has been a treadmill for a year. You kill it: recs re-samples the benchmark from post-filter served traffic and adopts slate-level nDCG. Two roadmap items die with the metric. This is the political moment: the recs lead escalates; the scatter plot wins the argument because it is *their own launches* on the axes.

**Step 5 — The contested launch.** Quarter 3: ads wants to ship a new bidder before the holiday freeze. Eval card shows: offline +1.9% AUC [+1.2, +2.6] — real; shadow shows a prediction-distribution PSI of 0.31 vs. incumbent concentrated in low-traffic advertiser segments; the A/B is +1.1% revenue but only 9 days in, pre-registered duration 14 days, and the treatment-effect time series is decaying (day 2: +2.4%, day 8: +0.6%). The team argues aggregate significance (p=0.03). You hold the gate: decaying time series + tail-segment distribution shift + 5 days short of pre-registration = not yet. The experiment completes at +0.3% [−0.4, +1.0] — a novelty artifact on the aggregate, and the tail-segment shift turned out to be underbidding on small advertisers, which the extended window surfaced as a complaint spike. The bidder ships one quarter later with a fix, at +0.9% [+0.4, +1.4]. The org learns two things: the gate is real, and the gate is *right* often enough to be worth it. That second lesson is the only durable source of a standard's authority.

**Outcome after four quarters.** Holdback shows +3.8% cumulative CTR-equivalent lift vs. +4.4% claimed — the ledger is now honest within noise. Zero shipped-then-reverted launches (previous year: three). Support LLM regression suite has blocked 11 prompt changes, each a would-have-been incident. The standard survives your absence — which is the actual success criterion for principal work.

## Exercise

**Task.** A company is about to launch **"PolicyPal"**: an LLM assistant that drafts responses to insurance-claim inquiries for human agents (2M inquiries/year, 300 agents, drafts accepted/edited/rejected by the agent; 12% of inquiries are regulated communications with legal wording requirements). The model is a fine-tuned 8B behind a RAG pipeline over policy documents. Current state: the team has a demo, a 40-example "vibes" test set, and a launch date in six weeks.

Produce two artifacts:

1. **An eval standard for this launch** (≤2 pages): the required rungs of the hierarchy and what each must show; golden-set design (size, stratification, sourcing, grading budget); judge design including human-agreement calibration plan and drift controls; the experiment design (primary metric — hint: draft-acceptance rate — MDE, power calculation with stated assumptions, duration, guardrails); the launch gate and rollback criteria; eval-set governance (versioning, sealed subset, re-sampling cadence).
2. **A completed eval card** for the launch, using the section 6 template, with plausible invented numbers — every field filled, every metric carrying an interval.

**You're done when:**
- The eval card has zero empty fields and zero numbers without a source or interval.
- Your power calculation is arithmetically checkable from stated assumptions (baseline acceptance rate, MDE, traffic) and yields the experiment duration.
- The regulated-communication slice appears explicitly in the golden set stratification, the guardrails, and the rollback triggers.
- A colleague can identify, from your standard alone, at which rung a hallucinated-policy-clause failure would be caught.

**Self-check questions:**
1. Which rung of the hierarchy catches a model that scores well on the golden set but produces subtly wrong deductible amounts on live long-tail policies — and what does that imply about your shadow-phase design?
2. Your judge agrees with humans at κ=0.55 on the "legal wording" dimension. What are your options, and which is cheapest?
3. Draft-acceptance rate can be gamed: how, and which guardrail catches it?
4. Eighteen months in, the team wants to remove 80 outdated examples from the golden set. What do you require before the number "acceptance-relevant quality +2% YoY" can appear on any dashboard afterward?
5. If agents' editing behavior feeds the next fine-tune, what feedback loop exists, and what logging do you mandate now to keep future bake-offs fair?
