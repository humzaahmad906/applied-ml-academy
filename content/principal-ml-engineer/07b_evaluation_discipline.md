# Module 07 — Evaluation as an Org Discipline — Part 2 of 2: Launch Gates, Eval-Set Governance & A/B Subtleties

This is part 2 of the Evaluation as an Org Discipline lesson. Here we cover the launch-gate policy that turns the eval standard into enforcement, the governance rules that keep eval sets trustworthy over time, and the A/B experiment subtleties a principal is expected to catch in review.

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

## You can now

- Place any evaluation activity on the five-rung hierarchy, state precisely what it proves and cannot prove, and use that framing to diagnose at which rung a given failure should have been caught cheaply.
- Run a pre-experiment power analysis, size the required sample for a given MDE and baseline rate, and recognize when a surface is too low-traffic to detect the effect being claimed — before the team celebrates the delta.
- Bootstrap a paired confidence interval on an offline metric delta using the shared-library pattern, and explain why "+0.8% [−0.3%, +1.1%]" and "+0.8% [+0.6%, +1.0%]" are different business situations requiring different decisions.
- Design and commission an offline-online correlation study with 25–40 launch data points, tag results by intervention type, and make a keep/recalibrate/kill call on a benchmark that teams have roadmaps and promotions built on.
- Write and enforce a launch gate policy with a filled eval card, three-verdict discipline (ship / ship-with-conditions / not yet), and a standing question list — and calibrate the gate's rigor by tracking first-pass rates toward the 70–85% healthy range.

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
