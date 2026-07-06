# Module 12 — Governance, Model Risk & Responsible AI

## Why this module matters

Every failure in Module 11's pattern library had a governance-shaped hole behind it: nobody could say who approved the model, what data it trained on, or what evidence supported the launch. Principals inherit governance whether they want it or not — when the regulator letter arrives or the discrimination story breaks, the question comes to the most senior engineer in the room, not to legal. The senior engineer's mistake is treating governance as paperwork to minimize; the principal's insight is that governance done as an *engineering system* — registries, automated documentation, tiered review — is cheap, and governance done as manual paperwork is both expensive and ineffective. This module teaches you to build the system version. It is also increasingly a legal requirement, not a preference: the EU AI Act's high-risk obligations and decades of US banking model-risk supervision define a floor you must design to, not negotiate with.

## 1. Governance is an engineering system, not a review meeting

The naive implementation of ML governance is a committee: a monthly meeting where model owners present slide decks and a review board asks questions. This fails in a predictable way — the committee becomes a bottleneck at ~5 models/month of throughput, teams learn to route around it, and the artifacts it produces (slides, meeting minutes) are unqueryable and stale within a quarter. Two years in, the org has a governance *ritual* and no governance *state*: nobody can answer "which production models touch EU-resident data?" without a two-week email archaeology project.

The engineering framing inverts this. Governance is a set of invariants maintained by systems:

- **Every production model is registered** — enforced by the deployment pipeline refusing to serve unregistered artifacts, not by policy memo.
- **Every model has a risk tier** — assigned at registration, and the tier mechanically determines which gates apply.
- **Every high-tier model has current documentation** — generated from the registry and training metadata, and the deploy gate fails if generation fails.
- **Every prediction that affects a person is reconstructible** — model version, feature values, and explanation retrievable by prediction ID for the retention window.

Notice what this buys you: compliance questions become queries. "List all high-risk models, their last validation date, and their fairness metrics" is a SQL statement against the registry, not a quarter-long audit. The principal's job is to get the org to build the invariant-maintaining systems *before* the regulator or the incident forces it, which means selling it internally as what it actually is: the same metadata infrastructure you need for reliability (Module 10) and evaluation discipline (Module 07), with a compliance read API on top. Budget planning number: a registry-plus-docs-pipeline for a 50-model org is roughly 2 engineers for 2 quarters if you build on an existing MLOps stack — the mechanics of registries and lineage are covered in the MLOps course; here we care about what the org must be able to *answer*.

## 2. Model risk management: what banking figured out twenty years ago

The most mature model-governance regime in existence is US banking supervision, codified in **SR 11-7** (Federal Reserve/OCC, 2011). It predates deep learning and applies to any "quantitative method that processes input data into estimates" — which means your gradient-boosted credit model and your LLM-based document classifier both qualify. Even if you never work in banking, SR 11-7 is worth internalizing because every newer framework (NIST AI RMF, EU AI Act conformity assessment, insurance-regulator model laws) is a variation on its three pillars:

1. **Conceptual soundness.** Is the model built on a defensible design for its stated purpose? This is validated *before* deployment: are the features causally plausible or at least stable proxies, is the training population representative of the application population, are the assumptions documented and tested? A model that performs well for reasons nobody can articulate fails this pillar — Optum's cost-as-proxy-for-health-need model (Module 11) is precisely a conceptual-soundness failure that no amount of accuracy testing would have caught, because the model was accurate *at the wrong target*.
2. **Ongoing monitoring.** Does the model still work? Input drift, output drift, population stability, performance against outcomes as labels mature. Banking adds a discipline most ML teams lack: monitoring against the model's *documented limitations* — if the validation report said "not validated for loan amounts above $500k," there is an alert on the share of scoring requests above $500k.
3. **Outcomes analysis.** Did the decisions the model drove turn out well? Backtesting predicted-vs-realized default rates by score band, override analysis (how often do humans overrule the model, and were the overrides right?), and error analysis on the tails. This closes the loop the other two pillars leave open — a conceptually sound, well-monitored model can still be systematically miscalibrated on a subpopulation.

SR 11-7 also mandates **independent validation**: the person who built the model cannot be the person who validates it. In banks this is a separate department ("second line of defense"). In a tech org, the pragmatic version is validation by an engineer from a different team using a documented checklist — imperfect independence, but it catches the errors that author blindness protects. The eval-owner separation argued in Module 07 is the same principle.

### The model inventory: the foundation most orgs don't have

Here is an uncomfortable diagnostic you can run in any company: ask "how many ML models do we have in production?" and watch what happens. In most orgs the answers from three different leaders differ by 2–5×, because nobody counts the heuristic that quietly became a logistic regression, the vendor model embedded in a SaaS product, the fine-tuned classifier a team shipped inside a lambda, or the LLM prompt that is functionally a decision model. Industry surveys and supervisory findings consistently show the inventory problem is the number-one MRM finding — you cannot govern what you cannot enumerate.

A functioning inventory record is nine fields — resist the 60-field template that guarantees the inventory is never filled in:

```yaml
# inventory record — one per model, lives in the registry, not a spreadsheet
model_id: credit_underwriting_gbm
owner: j.alvarez            # a person, not a team alias — aliases go stale
business_use: >             # the decision affected, in decision language
  Approve/decline + credit-limit assignment for consumer installment loans
risk_tier: T1
inputs: [bureau_features_v3, bank_txn_features_v2, application_form]
training_data: snapshot://lending/train/2026-05-01   # immutable reference
production_version: sha256:9f2c...                    # deployed artifact hash
last_validation: 2026-03-15
monitoring: https://grafana/d/credit-uw               # dashboards + alerts
known_limitations: >
  Not validated for thin-file applicants (<6 months bureau history);
  miscalibrated above $75k requested amounts — hard policy cap applies.
``` The inventory is a table; the discipline is keeping it true, which is why registration must be enforced at the deployment pipeline (Section 1) rather than requested by memo. Include vendor and embedded models: the regulator does not care that you didn't train it; you deployed it against your customers. And include LLM systems — a prompt template plus a frontier model making eligibility decisions is a model under every definition that matters.

## 3. Risk tiering: not all models deserve the same rigor

The fastest way to kill governance is uniform rigor: if the feed-ranking tweak and the credit-underwriting model face the same review board, teams either drown or defect. Tier by **blast radius**, and let the tier mechanically determine the process. A workable three-tier scheme:

```text
Tier assignment — answer in order; first "yes" sets the tier:

T1 (high):    Does the model make or materially influence decisions about
              individuals' access to credit, employment, housing, insurance,
              education, healthcare, or legal outcomes? OR does it fall in an
              EU AI Act high-risk category? OR could a failure plausibly cost
              >$10M or a regulatory action?
T2 (medium):  Does it face customers directly (content they see, prices they
              pay, support they receive) OR move revenue/cost >$1M/yr OR
              process regulated data categories?
T3 (low):     Everything else — internal tools, offline analytics,
              non-customer-facing optimization.
```

What each tier buys, calibrated so T3 is nearly free:

| | T3 (low) | T2 (medium) | T1 (high) |
|---|---|---|---|
| Review | Self-serve checklist, peer sign-off | Checklist + independent eval review | Full review board + independent validation |
| Documentation | Auto-generated model card | Model card + data sheet | Card + data sheet + validation report + fairness analysis |
| Monitoring | Default drift dashboards | + performance-vs-outcome tracking | + subgroup metrics, override analysis, limitation alerts |
| Reproducibility | Registry entry | + pinned data snapshot | + full point-in-time reconstruction (Section 6) |
| Revalidation | On major change | Annual | Annual + on any retrain or data-source change |

Two design notes from experience. First, tier assignment itself needs review only at T1/T2 boundaries — publish the rubric and let teams self-assign T3, with a periodic audit sampling ~10% of self-assignments for sandbagging. Second, blast radius includes *aggregation*: fifty T3 models feeding one decision can constitute a T1 system. Tier the decision surface, not just the artifact.

## 4. The EU AI Act: risk classes as design constraints

The EU AI Act (in force August 2024, obligations phasing in through 2026–27) is the first comprehensive AI statute and the de facto global template — the "Brussels effect" that GDPR had on privacy. As a principal you need its risk taxonomy at the fluency level where it shapes architecture, not at lawyer level:

- **Prohibited** (Article 5, applicable since February 2025): social scoring by public authorities, emotion recognition in workplaces and schools, untargeted facial-image scraping, manipulative techniques causing harm. If a product idea lands here, the design review is over.
- **High-risk** (Annex III): AI in employment decisions, credit scoring, insurance pricing (life/health), essential-services eligibility, education scoring, biometrics, critical infrastructure, law enforcement. This is where most enterprise ML in regulated domains lands, and it carries the real obligations.
- **Limited risk**: transparency duties — chatbots must disclose they are AI; synthetic media must be labeled.
- **Minimal risk**: everything else (the large majority of systems); no new obligations.

The high-risk obligations read like a systems-design spec, which is exactly how to treat them: **risk management system** (documented, maintained through the lifecycle — your tiering and review process); **data governance** (training data relevance, representativeness, bias examination — your data sheets); **technical documentation and record-keeping** (automatically generated logs sufficient to trace each decision — your audit trail, Section 6); **transparency to deployers** (your model card); **human oversight** (a human must be able to understand, intervene, and override — which means your serving path needs an override mechanism and your UI needs to surface model uncertainty, a genuine architectural requirement, not a policy line); **accuracy and robustness evidence** (your eval suite, retained as evidence). Penalties top out at 7% of global revenue for prohibited-use violations and 3% for high-risk noncompliance — numbers that get a CFO's attention.

The design implication: if you build the SR 11-7-shaped system from Section 2, EU AI Act conformity is mostly a document-mapping exercise. If you build neither, you do the work twice under deadline. The map, explicitly — one engineering artifact serving both regimes:

| Engineering artifact | SR 11-7 pillar | EU AI Act high-risk obligation |
|---|---|---|
| Registry + tiering rubric | Model inventory, risk-based scope | Risk management system (Art. 9) |
| Data sheets + catalog consent tags | Conceptual soundness (data suitability) | Data governance (Art. 10) |
| Auto-generated model cards | Model documentation | Technical documentation, transparency (Arts. 11, 13) |
| Prediction-level audit trail | Ongoing monitoring evidence | Record-keeping / logging (Art. 12) |
| Override mechanism + uncertainty surfacing | Outcomes analysis (override review) | Human oversight (Art. 14) |
| Versioned eval suite + subgroup gates | Independent validation, outcomes analysis | Accuracy & robustness evidence (Art. 15) |

Module 01 of the ML System Design course covers the data-residency mechanics; here the point is that the *same registry* answers both regulators.

## 5. Fairness engineering: the part you cannot delegate to a library

Two facts about algorithmic fairness that every principal must be able to explain to a general counsel and a VP without notes.

**First: excluding protected attributes does not make a model fair, and disparate impact through proxies is illegal in regulated domains.** US fair-lending law (ECOA, Fair Housing Act) uses a disparate-*impact* standard: if outcomes differ significantly across protected groups and the practice isn't justified by business necessity achievable through less discriminatory means, it's actionable — regardless of whether race or gender was an input. ML models are proxy-finding machines; ZIP code, shopping patterns, device type, and name-derived features reconstruct protected attributes with high fidelity. The Apple Card episode (2019) is the canonical illustration of the *governance* failure mode: spouses with shared finances received credit limits differing by 10–20×, the public explanation was "the algorithm doesn't use gender," and the NYDFS opened a probe. The investigation ultimately found no unlawful discrimination — but the point for you is that "we don't use gender as an input" was legally and statistically vacuous as a defense, the issuer could not initially produce a better explanation, and the reputational cost was paid in full either way. The engineering consequence is paradoxical and worth stating plainly to leadership: **you often need protected-attribute data (collected or inferred, e.g., BISG for race in lending) to *test* for discrimination, even though you must not use it to *predict*.** Orgs that refuse to touch protected data "to be safe" have chosen to be unable to detect their own disparate impact.

The standard first-pass metric is the **four-fifths rule** from employment law, applied per decision threshold:

```python
def adverse_impact_ratio(decisions: pd.DataFrame, group_col: str,
                         favorable_col: str, reference_group: str) -> pd.Series:
    """AIR = selection_rate(group) / selection_rate(reference).
    AIR < 0.80 is the regulatory screening threshold (EEOC four-fifths
    rule; also used by CFPB in fair-lending exams). Not proof of
    discrimination -- a tripwire that triggers deeper analysis."""
    rates = decisions.groupby(group_col)[favorable_col].mean()
    return rates / rates[reference_group]
```

**Second: fairness definitions mathematically conflict, so choosing one is a values decision that must be made explicitly and above your pay grade.** Chouldechova (2017) and Kleinberg et al. (2016) proved that when base rates differ across groups, a model cannot simultaneously satisfy **calibration within groups** (a score of 0.7 means 70% risk for everyone), **equal false-positive rates**, and **equal false-negative rates**. This is not an engineering limitation to be optimized away; it is arithmetic. The COMPAS controversy (Module 11) was two parties correctly measuring different definitions — ProPublica showed unequal false-positive rates; Northpointe showed calibration — and both were right, because both cannot hold at once.

The principal's move here is the one this course keeps returning to: **surface the decision, don't bury it.** A team that silently picks demographic parity inside a training script has made a consequential values commitment on behalf of the company, invisibly. The correct artifact is a one-page decision memo: here are the two or three candidate definitions, here is what each means concretely for applicants ("equal approval rates" vs "equal error rates among qualified applicants" vs "scores mean the same thing for everyone"), here is our recommendation and why, here is the residual disparity we will accept and monitor — signed by the accountable executive. That memo is also, not coincidentally, exactly what a regulator asks for. Fairness *engineering* is then the tractable part: measure the chosen metric per subgroup in the eval suite, gate deployment on it, monitor it in production with the same rigor as latency, and run the less-discriminatory-alternative search (feature ablation, constraint-regularized retraining) that fair-lending law expects — modern tooling makes searching for less discriminatory model variants cheap enough that "we couldn't find one" no longer holds up if you never looked.

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
