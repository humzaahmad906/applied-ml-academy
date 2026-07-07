# Module 07 — Evaluation as an Org Discipline — Part 1 of 2: Foundations, Statistics & LLM Eval Infrastructure

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

$$n_\text{per arm} \approx \frac{16\,p(1-p)}{\text{MDE}^2} \qquad (\alpha=0.05,\ \text{power}=0.8,\ \text{two-sided})$$

Example: baseline CTR $p = 0.05$, minimum detectable effect = 2% relative, so $\text{MDE}_\text{abs} = 0.001$:

$$n \approx \frac{16 \times 0.0475}{(0.001)^2} = \frac{0.76}{0.000001} \approx 760{,}000 \text{ users per arm}$$

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

## You can now

- Place any evaluation activity on the five-rung hierarchy, state precisely what it proves and cannot prove, and use that framing to diagnose at which rung a given failure should have been caught cheaply.
- Run a pre-experiment power analysis, size the required sample for a given MDE and baseline rate, and recognize when a surface is too low-traffic to detect the effect being claimed — before the team celebrates the delta.
- Bootstrap a paired confidence interval on an offline metric delta using the shared-library pattern, and explain why "+0.8% [−0.3%, +1.1%]" and "+0.8% [+0.6%, +1.0%]" are different business situations requiring different decisions.
- Design and commission an offline-online correlation study with 25–40 launch data points, tag results by intervention type, and make a keep/recalibrate/kill call on a benchmark that teams have roadmaps and promotions built on.
- Onboard an LLM judge with the human-agreement-first protocol — calibrate on double-graded examples, run the bias battery, pin the judge version — before letting its score gate a single launch.
