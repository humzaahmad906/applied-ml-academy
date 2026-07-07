# Module 12 — Governance, Model Risk & Responsible AI — Part 2 of 2: Documentation, Privacy & Velocity

This is part 2 of the Governance, Model Risk & Responsible AI lesson. Here we cover automating documentation and audit trails, resolving the three privacy-versus-ML tensions, and keeping governance from becoming a velocity tax.

## 6. Documentation and audit trails: automate or abandon

Documentation regimes fail in one specific way: templates get filled in once, by hand, under launch pressure, and never again. Six months later the model card describes a model two retrains ago. The fix is the same as for all metadata: **generate documentation from systems of record, and fail the pipeline when generation fails.**

The three artifacts worth institutionalizing, in descending order of automation potential:

- **Model cards** (Mitchell et al., 2019): intended use, training data summary, eval results overall and by subgroup, limitations. Roughly 80% is mechanically derivable — registry supplies version/owner/lineage, the eval harness supplies metrics tables, the training config supplies data references. Only "intended use" and "limitations" need human prose, and those change rarely. A card pipeline is ~2 engineer-weeks against a decent registry and pays for itself the first time anyone asks "what changed between v14 and v17?"
- **Data sheets** (Gebru et al., 2021): provenance, collection method, consent basis, known gaps and skews, per training dataset. Semi-automatable from catalog metadata; the consent-basis field must come from a human with legal input, once per source.
- **Decision logs**: the fairness-definition memo, the tier assignment, the build-vs-buy call, the launch approval — the Module 02 decision-document discipline applied to governance. These are inherently human-written; the system's job is to *index* them from the registry entry so the model's paper trail is one click, not one archaeology project.

The card pipeline in one sketch — the point is where each block *comes from*:

```python
def generate_model_card(model_id: str, version: str) -> str:
    reg   = registry.get(model_id, version)          # owner, tier, lineage
    evals = eval_store.latest_report(model_id, version)  # metrics + subgroups
    data  = catalog.describe(reg.training_data)      # sources, consent basis
    human = load_yaml(f"cards/{model_id}.yaml")      # intended_use, limitations
                                                     # -- versioned with the
                                                     # training config, written
                                                     # once, reviewed on change
    return render("model_card.md.j2",
                  registry=reg, evals=evals, data=data, human=human)

# Runs in the deploy pipeline. Generation failure (missing eval report,
# unregistered data source, absent human blocks) FAILS THE DEPLOY.
# That coupling — not the template — is what keeps cards current.
```

For consumer-affecting models add **adverse-action reasons**: US credit law (ECOA/Reg B) requires telling a declined applicant the principal reasons, which means your serving path must produce per-decision reason codes — typically top-k feature attributions mapped to human-readable reasons — and produce them *at decision time*, stored with the prediction. Retroactive explanation of a decision made by a since-retrained model is exactly the failure the next paragraph exists to prevent.

**Audit-trail architecture.** The compliance requirement underneath all of this is point-in-time reproducibility: for any consequential prediction, reconstruct *who trained what, on which data, with which config, and what the model saw and said*. Concretely, per prediction: `prediction_id, timestamp, model_name, model_version (immutable artifact hash), feature_vector as served, score, decision after policy layer, explanation/reason codes` — and per model version: `training code commit, config, data snapshot reference, eval report, approver`. Retention follows the domain: five years is a common floor in lending; the EU AI Act requires logs "appropriate to the intended purpose" for high-risk systems, in practice the system's lifetime. Note the happy alignment: log-and-wait, the training-serving-skew mitigation from the ML System Design course, produces this record as a side effect. The reliability system and the compliance system are the same system with two read paths — build it once, and let compliance help pay for it.

## 7. Privacy intersections: the deletion problem

Three collisions between privacy law and ML mechanics that governance must resolve in advance, because resolving them under a regulator's deadline is 10× the cost:

- **Right to deletion vs trained models.** GDPR Article 17 requires erasing personal data on request; a model trained on that data has, in some measurable sense, absorbed it (membership-inference attacks make this concrete). The pragmatic position most DPAs currently accept: delete from all training stores so the *next* retrain excludes the person, ensure the model does not regurgitate identifiable data, and document the retrain cadence as your effective deletion latency. This makes **retrain cadence a compliance parameter**: a model retrained annually has a 12-month deletion lag someone must sign off on. Machine unlearning is not yet a production-grade answer; say so honestly rather than gesturing at it.
- **Purpose limitation vs the data flywheel.** Module 06's flywheel assumes interaction data feeds training; GDPR assumes data is used for the purpose it was collected for. The engineering artifact is a consent/purpose tag on every dataset in the catalog, checked at training-pipeline admission — a join, not a meeting.
- **Retention vs audit.** Privacy law says minimize retention; Section 6 says keep prediction records for years. The resolution is field-level policy: retain the decision record (scores, reasons, model version) for the audit window while ageing out or tokenizing direct identifiers not needed to reconstruct the decision. Design the log schema with this split from day one; retrofitting field-level retention onto a monolithic JSON blob is a quarter of misery.

## 8. Governance without killing velocity

The failure mode that discredits governance forever is the two-month review queue. Velocity math: if your org ships 40 model changes a quarter and each waits two weeks for review, you've burned ~1.5 engineer-quarters of calendar time and taught every team that compliance is the enemy. The design principles that keep the tax proportional:

- **Tier the depth, not just the requirement.** T3 review is a 20-minute self-serve checklist with peer sign-off, audited by sampling. T2 adds an independent eval review, async, 3-business-day SLA. Only T1 sees a review board — and a 50-model org has maybe 5–8 T1 models, so the board meets monthly and is never the bottleneck.
- **Review the system once, not every change.** Approve the *pipeline* (data sources, eval gates, monitoring, rollback) at the tier's rigor; routine retrains within the approved envelope ship on green evals. Only envelope changes — new data source, new feature family, objective change — re-enter review. This single rule removes ~80% of review volume.
- **Put the gates in the pipeline.** Fairness metrics, doc generation, registry checks run in CI; the review meeting discusses exceptions, not checklists a machine can evaluate.
- **Give the review board teeth and a clock.** Named members, quorum rules, decisions in writing with conditions, a hard SLA (e.g., 10 business days for T1), and an escalation path when it's missed. A board that can only say "come back next month" is a velocity tax with no risk reduction.
- **Staff it as engineering.** One governance-minded ML engineer per ~20–30 production models to own the registry, card pipeline, and rubric beats a five-person committee producing memos. The Larson *Right Hand* mode fits: you're operating with the executive's authority to set standards, which is also why this cannot be done from a purely IC position without sponsorship — get the exec signature on the tiering rubric before enforcing it.

What "20-minute self-serve" means concretely — the entire T3 gate:

```text
T3 CHANGE CHECKLIST (self-serve; peer sign-off; ~10% sampled for audit)

[ ] Model is registered; inventory record fields current (owner, use, tier)
[ ] Tier re-check: does this change alter the decision affected, the
    population served, or a data source? If yes -> STOP, re-tier first
[ ] Eval gates green, compared against the pinned baseline (Module 07)
[ ] No new data source outside the approved envelope
[ ] No regulated data categories added to features
[ ] Rollback path verified (previous version deployable in <30 min)
[ ] Model card regenerated automatically (verify pipeline step passed)

Signed: author + one peer (not the author). Ship on completion — no meeting.
```

The checklist is deliberately boring. Its value is the second item: it is the tripwire that catches a T3 model drifting into T2/T1 territory, which is how "the ranking tweak" quietly becomes "the model that sets prices" without anyone re-tiering it.

## You can now

- Build a 9-field model inventory enforced by the deployment pipeline, enumerating your fleet from system artifacts rather than from memory and covering vendor and embedded models alongside internally trained ones.
- Apply the three-tier blast-radius rubric to any model portfolio, assign review depth proportional to risk, and write the tiering memo with a named approver so tier assignments do not drift silently as models change what they feed.
- Write a signed fairness-definition memo that names the chosen metric, explicitly accepts the residual disparity, states the conditions for reopening, and functions as the artifact a regulator or general counsel will actually ask for.
- Design a model-card pipeline that generates documentation from systems of record and fails the deploy when generation fails, so every card stays current without imposing manual effort on model teams.
- Resolve the three privacy-versus-ML tensions — deletion lag, purpose limitation, and retention-versus-audit — with field-level schema decisions made at design time, before a regulatory deadline forces the choices under pressure.

## Worked example

**Scenario.** You join LendFast, a mid-size online lender (~$800M/yr originations, 140 engineers, 9 ML engineers) as principal engineer. Week two, the head of risk forwards a state regulator's examination notice: fair-lending review in ~6 months, with a request for the model inventory and validation evidence. There is no inventory. Folklore says "about a dozen models." No CRO organization with MRM experience — you are it, with an exec sponsor (the CTO) and two engineers for a quarter.

**Step 1 — Inventory sprint (weeks 1–3).** You enumerate from systems, not from memory: scan deployment manifests for serving containers, the feature store's consumer list, the data warehouse for tables named `*_score*`/`*_pred*`, and vendor invoices (two third-party scores surface that no engineer mentioned). Result: **23 models**, not 12 — including a "temporary" income-verification heuristic-turned-XGBoost from 2023 with no owner (its author left; you assign one in week 2) and an LLM prompt that classifies hardship emails and routes collections treatment, which two teams insisted "isn't a model." Each gets the 9-field inventory record from Section 2. Deployment pipeline change ships in week 3: unregistered artifacts fail to deploy, with a 30-day grace period and a one-page registration form.

**Step 2 — Tiering (week 4).** Applying the Section 3 rubric with the CTO and head of risk in one two-hour session: **4 × T1** (credit underwriting model, pricing model, collections-treatment router — yes, the LLM one, it influences forbearance decisions — and the vendor fraud score, because it can block credit access), **7 × T2** (marketing response models, income verification, document-processing extractors feeding underwriting), **12 × T3** (internal forecasting, ops tooling). The tiering memo — one page, rubric plus assignments plus dissents — is signed by the CTO. Two teams sandbag ("our model is just advisory"); the rubric's decision-surface language settles both: if underwriters see the score, it materially influences the decision.

**Step 3 — T1 deep review of the underwriting model (weeks 5–9).** The main event, run as an SR 11-7-shaped validation by one of your two engineers (not the model's author) plus a fair-lending consultant (~$60k, worth it for the exam). *Conceptual soundness:* the model is a GBM on ~140 features; the review finds 12 features with no plausible causal story, including two device-fingerprint features that ablation shows contribute 0.3% AUC while correlating strongly with geography — cut, with the ablation table retained as less-discriminatory-alternative evidence. *Fairness analysis:* race proxies via BISG (the CFPB-standard method — documented as such, including its known error rates). AIR on approvals: 0.74 for one group at the current cutoff — below the 0.80 tripwire. Now the values decision: the team had never chosen a fairness definition. You write the Section 5 memo — candidate definitions were demographic parity (rejected: ignores real base-rate differences in the underwriting population and would require explicit group-conditional thresholds, legally fraught under ECOA), equalized odds (attractive but unstable to estimate on smaller subgroups at current volume), and **calibration within groups plus AIR monitoring with a less-discriminatory-alternative search** — recommended because calibrated scores are defensible to the regulator, and the LDA search is what fair-lending exams actually test for. The CEO and general counsel sign it. The LDA search (feature cuts above, plus a mild fairness-regularized retrain) lifts AIR to 0.83 at approximately equal portfolio default risk (−0.1% AUC, within noise). *Outcomes analysis:* backtest of score-band default rates finds the model materially miscalibrated on the thin file/no-file segment — a documented limitation now, with a monitoring alert on that segment's volume share.

The memo itself, in full — this is the artifact the exam will ask for and the template your exercise below should imitate:

```text
DECISION MEMO — Fairness definition, consumer underwriting model (T1)
Date: 2026-06-12   Owner: Principal Eng   Approvers: CEO, GC, Head of Risk

DECISION. We adopt (a) calibration within groups as the model's fairness
requirement, (b) adverse-impact-ratio monitoring with a 0.80 alert and
0.75 hard gate, and (c) a documented less-discriminatory-alternative
search on every envelope change.

WHY NOT THE ALTERNATIVES. Demographic parity would require group-
conditional thresholds (legally fraught under ECOA) and ignores real
base-rate differences in our applicant population. Equalized odds is
statistically unstable at our per-group volumes (<2k outcomes/quarter
for the smallest group); we will revisit at 2x volume.

WHAT WE ARE ACCEPTING. These properties cannot all hold at once
(Chouldechova 2017). Under our choice, false-positive rates WILL differ
across groups; current measured gap is 4.1pp. We accept this residual
disparity, monitor it quarterly, and re-open this memo if it exceeds 8pp.

MEASUREMENT. Race proxied via BISG (CFPB-standard; documented error
rates attached). Metrics computed per release in the eval pipeline;
deploy gate enforces (b).

REVIEW. Annually, on any envelope change, or on regulatory guidance
change — whichever comes first.
```

**Step 4 — The model-card pipeline (weeks 6–10, parallel).** Your second engineer builds card generation against the registry: a Jinja template pulling owner/version/lineage from the registry, metrics and subgroup tables (including AIR) from the eval harness's JSON output, and data-source descriptions from the catalog; humans write intended-use and limitations blocks once, stored as versioned YAML next to the training config. Card generation runs in the deploy pipeline; a failing generation blocks deploy. All 23 models have current cards by week 10 — total human writing effort ~15 hours across teams, because everything mechanical is generated. Adverse-action reason codes already existed (Reg B forced that years ago), but the review finds they were computed from a *surrogate* model that had drifted from production; fixing them to come from the served model's own attributions, stored per decision, closes the exam's most dangerous finding before it's found.

**Outcome.** Exam lands in month 7: the examiners get a queryable inventory, four T1 validation reports, a signed fairness-definition memo with an LDA search, and generated-from-source model cards. Two findings, both minor (retention documentation, one stale monitoring threshold), no enforcement action. Steady-state cost going forward: ~0.7 FTE on governance engineering, a monthly one-hour T1 board, and a 20-minute checklist per T3 change. The credit team's retrain cadence — the thing everyone feared governance would kill — goes from quarterly to monthly, because the approved-envelope rule (Step 3's pipeline approval) means routine retrains ship on green gates without any meeting.

## Exercise

**Task.** You are the principal at "MedSupply", a healthcare-adjacent B2B marketplace (hospital procurement, 4,000 hospital customers, EU + US operations). Risk-tier the following portfolio and write the governance policy one-pager.

The portfolio: (1) a search-ranking model for the product catalog; (2) a dynamic pricing model that sets per-customer discounts; (3) an LLM assistant that answers clinicians' product questions, including compatibility and sterile-use questions; (4) a credit-limit model for hospital net-30 terms; (5) a demand-forecasting model for warehouse stocking; (6) a resume-screening model HR bought from a vendor for the EU sales org; (7) an anomaly detector flagging suspicious orders for manual review; (8) a fine-tuned extraction model that reads procurement contracts and feeds the pricing model.

**Deliverables.**

1. A tier assignment table: model, tier, one-sentence justification citing blast radius and (where applicable) the EU AI Act class. Note aggregation effects (which models feed which decision surfaces) and at least one case where a model's tier is raised by what it feeds.
2. A one-page governance policy: the tier definitions, what each tier requires (review, docs, monitoring, revalidation), how the process avoids becoming a bottleneck (approved-envelope rule, SLAs, self-serve tiers), and who signs what. Target: readable by your CTO in five minutes, executable by a team lead without asking you questions.

**Acceptance criteria — you're done when:**

- Every model has a tier and a justification that names the decision affected and its blast radius, not just the model type.
- At least one model is tiered higher than its "technical" riskiness suggests because of what it feeds or where it operates (hint: items 6 and 8 both have traps — one is Annex III high-risk regardless of vendor origin; one inherits the tier of the pricing decision it feeds).
- The policy one-pager fits on one page and specifies review SLAs in business days, named approver roles, and the envelope rule for retrains.
- The LLM assistant's tier discussion addresses the clinical-safety question explicitly (what happens if it answers a sterile-use question wrong?) rather than treating "it's just a chatbot" as an answer.

**Self-check questions.**

1. For the vendor resume screener: which EU AI Act class does it fall into, and does "we bought it, we didn't build it" change your obligations as the deployer?
2. The credit-limit model shows AIR of 0.77 for one demographic group (proxied via firmographics — but hospital ownership correlates with community demographics). ECOA applies to business credit too. What are your next three actions, in order?
3. Which of the eight models can share a single approved-envelope pipeline review, and which need per-change review? What property of the change, not the model, drives that line?
4. Your CEO asks: "Can't we just not collect any demographic data, so we can't be accused of using it?" Give the two-sentence answer that explains why this makes the company *less* safe.
5. A hospital exercises GDPR deletion for a departed employee whose interactions trained the LLM assistant's fine-tune. What is your effective deletion latency, and where is it documented?
