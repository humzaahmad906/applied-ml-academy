# Module 15 — The Principal Interview Bank

## Why this module matters

Principal loops are not harder senior loops; they are a different exam. Senior interviews test whether you can design a system; principal interviews test whether you can design the *organization's relationship to its systems* — and whether you have actually done it, repeatedly, for years. The evidence bar shifts from "solved a hard problem" to "changed how a company operates," and the format shifts from whiteboard puzzles to strategy cases, ambiguous org scenarios, and behavioral interviews graded against explicit leadership rubrics. This module gives you the twelve system-design/strategy questions that cover ~90% of what L7+/E7+/Principal loops actually ask, with model answers, plus the behavioral pattern and the leveling mechanics that decide whether your offer says "Staff" or "Principal."

## 1. What these loops actually test

### The three assessment surfaces

**Org-scale system design.** The question sounds technical ("design the ML platform") but the rubric is organizational: Do you ask how many teams, not just how many QPS? Do you design adoption, funding, and migration alongside architecture? A senior candidate produces a correct component diagram; a principal candidate produces a component diagram *plus* the mechanism by which six teams end up using it. Interviewers at this level are usually principals themselves and are listening for scar tissue — the throwaway remarks ("shadow mode for a full seasonal cycle, because the Q4 distribution is different") that only come from having been burned.

**Strategy cases.** "You inherit X mess / the CFO asks Y / the board mandates Z." These have no architecture at all in the answer's first half. They test diagnosis (Module 02), sequencing (Module 08), and the discipline of quantifying before deciding. The trap: treating them as invitations to describe your favorite stack.

**The behavioral bar.** At Amazon, the Principal Engineering Tenets are the literal rubric: *Exemplary Practitioner, Technically Fearless, Ask Why / What If, Dive Deep, Deliver Results, Respect What Came Before, Are Right A Lot* — interviewers map your stories onto tenets and vote per tenet. The **bar raiser** (Amazon) or equivalent calibrated interviewer exists specifically to prevent the loop from hiring on likability or a single impressive project; they will pull one of your stories apart for 20 minutes looking for what *you* decided versus what happened around you. Meta E7/E8 loops weigh cross-functional influence and "org-shaped" impact; Google L7/L8 adds a hiring-committee stage where a packet is judged without you in the room.

### The packet and committee reality

At Google and increasingly elsewhere, the interviewers' notes, your self-summary, and references form a **packet** that a committee levels. Committees are conservative: the default outcome for an ambiguous packet is a down-level, not a rejection. What moves a committee to L7+ is evidence of **sustained** leadership — multiple years, multiple initiatives, compounding scope — not one great project. One heroic migration reads as strong senior. Three years of platform strategy where the second and third years show the first year's bets paying off reads as principal. Structure every answer so that the *duration and follow-through* are audible: "we shipped it in Q2, and eighteen months later it carried 90% of traffic and had absorbed two more teams" is a packet sentence; "we shipped it" is not.

Frontier labs (staff+ research/infra roles) run leaner loops but test the same delta with different vocabulary: instead of "influence without authority," expect deep technical interrogation of one system you built plus a strategy conversation about compute allocation, eval trustworthiness, or safety/velocity tradeoffs. The currency is the same: judgment with receipts.

## 2. System-design and strategy questions — the twelve

Format per question: the question, what separates a principal answer from a senior one, a condensed strong-answer sketch, and red flags. Each sketch follows the skeleton drilled throughout this course: requirements → metrics → architecture/plan → deep tradeoffs → cost math → risks. Assume 45–60 minutes each; budget the first 10 on requirements even when it feels slow.

### Q1. Design the ML platform for a 2,000-person company

**Principal vs senior:** Senior answers draw the standard boxes (feature store, registry, training orchestration, serving, monitoring). Principal answers start with the *demand side* — how many teams, what model archetypes, what's already built — and treat adoption and funding as first-class design components. The platform is a product with internal customers, not an infrastructure diagram (Module 03).

**Strong answer sketch:**
- Requirements: ~2,000 employees → ~600–800 eng → expect 6–12 ML-adjacent teams, 20–60 models, mixed archetypes (recsys, risk, forecasting, GenAI). Ask what exists: greenfield platforms for brownfield orgs fail. Ask the funding model: platform team headcount (12–20 eng, $3–5M/yr loaded) must be justified against duplication cost.
- Metrics: time-to-first-model-in-prod (target: weeks → days), % models on paved road, platform NPS, incident rate per model, GPU utilization, cost per model per month. State that adoption % is the metric that matters — an unused platform is a cost center.
- Architecture: thin paved road, not a mandate — data access layer with point-in-time feature store; training orchestration on K8s or managed equivalent; a model registry as the single source of truth with ownership metadata; two serving tiers (batch-to-KV, real-time containers) plus a gateway for LLM APIs; eval infra and monitoring as defaults, not add-ons.
- Tradeoffs: mandate vs paved-road (paved road + make it genuinely better; mandate only the registry and security boundaries); build vs buy per layer (buy commodity — orchestration, tracking; build the integration glue and anything touching your data semantics); central team vs embedded platform engineers (start central, embed liaisons into the two biggest teams).
- Cost math: 15-eng platform team ≈ $4M/yr vs the counterfactual — 8 teams each spending 2 engineers on redundant infra ≈ $4.5M/yr *plus* 5 serving stacks' worth of incident risk. Platform breaks even on deduplication alone if adoption exceeds ~60%.
- Risks: platform becomes a bottleneck (mitigate: self-serve everything, SLAs on platform-team response); second-system effect (mitigate: migrate the ugliest existing system first, not the easiest); building for imagined scale (mitigate: no component before two teams need it).

**Red flags:** naming vendor products as the answer; designing for Google scale at a 2,000-person company; no adoption mechanism; no funding story; "we'll mandate it."

### Q2. You inherit 40 models and 9 serving stacks — walk me through your first 90 days

**Principal vs senior:** Senior instinct is to start consolidating. Principal instinct is to start *counting* — inventory, business-value ranking, risk triage — and to earn credibility before spending it. The consolidation itself takes 18 months; the 90 days buy the right to run it (Modules 02, 08).

**Strong answer sketch:**
- Weeks 1–4, diagnose: build the model inventory (owner, business metric, revenue exposure, last retrain, monitoring status, stack). Expect to find 5–8 models nobody owns and 2–3 carrying most of the revenue. Interview every team lead; listen for what they'd fix. Deliverable: a one-page findings memo with numbers, no recommendations yet.
- Weeks 5–8, stop the bleeding: eval + output monitoring on the top-5 revenue-exposure models regardless of stack (this is cheap and stack-agnostic); named on-call ownership for every tier-1 model; kill or archive the orphans (typically 15–25% of the inventory).
- Weeks 9–12, strategy: pick the target serving stack from *data* (which stack hosts the most tier-1 traffic with the fewest incidents — usually you keep 2: one batch, one real-time, not 1); write the strategy doc with migration sequencing (strangler pattern, riskiest-value-first vs easiest-first tradeoff made explicit); land one visible quick win (e.g., decommission the worst stack's last two models) to make the plan credible.
- Cost math: 9 stacks ≈ 9 on-call rotations, 9 upgrade treadmills; consolidating to 2 saves an estimated 4–6 eng-years/yr in maintenance — that's the headline number for leadership.
- Risks: teams defend their stacks (pre-wire, per Module 13); migration stalls after the easy wins (put the second-hardest migration second, while energy is high).

**Red flags:** proposing the target architecture in week 1; promising 9→1 in a quarter; no business-value ranking; treating it as purely technical rather than organizational archaeology.

### Q3. API vs self-host at 10M requests/day — show me the math

**Principal vs senior:** Senior answers pick a side. Principal answers segment the traffic, run the numbers with stated assumptions out loud, include the ops-team cost line that juniors omit, and land on a cascade with a crossover condition (Modules 05, 09).

**Strong answer sketch:**
- Assumptions (state them): 1,500 input / 300 output tokens per request, frontier API at $2.50/$10 per 1M in/out, quality requirement varies by request type.
- API path: 10M × (1,500 × $2.50/1M + 300 × $10/1M) = 10M × $0.00675 ≈ **$67.5k/day ≈ $2M/month**. At this spend you negotiate: committed-use discounts of 20–40% are standard, so call it $1.2–1.6M/month effective.
- Self-host path: output throughput needed = 10M × 300 / 86,400 ≈ 35k tok/s sustained, ~2.5× peak → plan for ~90k tok/s. A well-tuned 8×H100 node runs an 8B model at ~40k decode tok/s → 3 nodes serving + 1 for headroom/canary ≈ 4 nodes × $30k/month ≈ $120k/month compute. Add the honest lines: a 4–6 eng serving team ($1–1.5M/yr ≈ $100k/month), eval/quality infra, and the capability question — does an 8B fine-tune match frontier quality *on your distribution*? Only evals answer that.
- The principal move — segment: typically 70–85% of traffic is routine (classification, extraction, templated generation) where a distilled 8B matches frontier after fine-tuning; the rest needs the API. Cascade: 80% self-hosted at ~$0.3/1M-token-equivalent + 20% API ≈ **$350–500k/month all-in vs $1.2–2M all-API** — a $10–18M/yr swing, which pays for the serving team many times over.
- Risks: utilization collapse if traffic is spiky (self-host math dies below ~60% util); model-upgrade treadmill; single-vendor API risk on the other side. Mitigation: keep the API as fallback and for the long tail permanently.

**Red flags:** comparing API list price to raw GPU cost with 100% utilization and zero headcount; not asking what the requests *are*; a binary answer; no crossover condition.

### Q4. Design org-wide LLM evaluation infrastructure

**Principal vs senior:** Senior answers describe an eval harness. Principal answers design an *eval standard* — the artifact schema, the adoption mechanism, and the calibration discipline that keeps LLM-judges honest (Module 07).

**Strong answer sketch:**
- Requirements: N teams shipping LLM features, no shared definition of "works," incidents discovered by users. Goal: every model/feature has a versioned eval card; regressions caught pre-deploy.
- Metrics for the eval infra itself: coverage (% of production LLM surfaces with eval cards), judge–human agreement rate (target >85% on calibration sets), time-to-add-an-eval (<1 day), regression escape rate.
- Architecture: central eval *service* (versioned golden sets, run history, dashboards) consumed as a library in each team's CI; eval cards as the unit — task definition, golden set provenance, metrics with thresholds, judge prompt + calibration evidence, known failure modes; three eval layers — deterministic assertions (cheapest, run always), LLM-as-judge (calibrated quarterly against human labels, with position-bias and verbosity-bias checks), human review (sampled, for calibration and high-stakes surfaces).
- Tradeoffs: central mandate vs federated adoption (mandate the card schema and the CI gate for tier-1 surfaces only; make the tooling good enough that tier-2/3 adopt voluntarily); blocking gates vs advisory (blocking gates teams route around are worse than advisory gates teams trust — start advisory, graduate to blocking per-surface once flake rate <2%).
- Cost math: 3-eng eval platform team + ~$10–30k/month judge inference ≈ $1.5M/yr vs a single eval-escape incident (the $2M class is common; see Module 11).
- Risks: golden sets go stale (accrete from production incidents — every postmortem adds cases); judge drift when the judge model upgrades (pin versions, re-calibrate on upgrade); Goodharting the eval (rotate held-out sets).

**Red flags:** one universal benchmark for all teams; LLM-judge with no human calibration story; eval infra with no adoption mechanism; treating evals as a launch gate only rather than a continuous monitor.

### Q5. A critical model degraded three weeks ago and nobody noticed. Diagnose and fix — the org, not just the model

**Principal vs senior:** Senior answers fix the model and add an alert. Principal answers treat "nobody noticed for three weeks" as the actual incident and redesign the ownership and observability system that allowed it (Modules 10, 11).

**Strong answer sketch:**
- Immediate (days 1–3): quantify blast radius (which decisions, how many, $ impact); mitigate — rollback to last-known-good, or threshold override with human review if rollback is impossible; communicate the number to leadership yourself, early, with the fix plan attached.
- Diagnosis of the org failure, asked as five whys: Why no alert? No output-distribution monitoring. Why not? No monitoring standard. Why did no one own it? No ownership registry; the builder left. Why didn't downstream metrics catch it? Business metrics reviewed monthly, attribution unclear. This is Sculley's hidden-technical-debt made flesh: the model was an unowned dependency of a revenue process.
- The org fix: (1) model ownership registry — every production model has a named owner and an on-call path, enforced at deploy time by the registry; (2) tiered monitoring standard — tier-1 models get input-drift, output-distribution, and business-metric monitors with paging; tier-3 get weekly digests; (3) a weekly model-health review ritual (30 min, dashboards, rotating chair) so humans look at the numbers; (4) blameless postmortem published org-wide with the systemic actions, not the individual error.
- Metrics: mean-time-to-detect for model regressions (3 weeks → target <24h for tier-1), % models with named owners (→100%), monitor coverage by tier.
- Risks: monitoring theater (100 dashboards nobody reads — the review ritual is the fix); over-alerting causing fatigue (page only on tier-1 with business-metric confirmation).

**Red flags:** stopping at "add drift detection"; naming a culprit; no dollar quantification; fixing this model without the registry/tiering that fixes the *class*.

### Q6. Build vs buy a feature store — defend your answer to the CFO

**Principal vs senior:** Senior answers compare features. Principal answers present a 3-year TCO in the CFO's language — NPV of headcount, opportunity cost, exit costs — and apply the differentiation test before any spreadsheet (Module 09).

**Strong answer sketch:**
- The differentiation test first: does a feature store differentiate our product? No — it's plumbing. Default to buy unless the numbers refuse.
- Build TCO (3-yr): 4 eng × 15 months to parity ≈ $2.5M build + 2 eng ongoing ≈ $2.5M maintenance → **~$5M**, plus 15 months of delay during which teams keep duplicating pipelines.
- Buy TCO (3-yr): vendor license $350–500k/yr → $1.3M + integration (2 eng-quarters, $250k) + 1 eng ongoing ownership ($1.35M) → **~$2.9M**, live in one quarter.
- The CFO framing: buying frees ~3 engineers/yr for revenue work — the real argument is opportunity cost, not license fees. State the exit cost honestly: vendor lock-in mitigated by keeping feature *definitions* in our repo in an open format; switching cost estimated at 2 eng-quarters.
- When build wins anyway: extreme scale where vendor pricing scales with rows (compute the crossover), data-residency constraints no vendor meets, or feature semantics so entangled with proprietary systems that integration exceeds build. Check each; show your work.
- Risks: vendor viability (escrow, open formats), price escalation at renewal (multi-year lock with caps), the hidden buy-side cost of adapting your workflow to the vendor's opinions.

**Red flags:** engineering-pride arguments ("we can build it better"); comparing license fee to zero instead of to loaded eng cost; omitting integration and maintenance from the buy side; no exit story.

### Q7. Migrate a batch recsys to real-time without a revenue dip

**Principal vs senior:** Senior answers describe the target architecture. Principal answers describe the *transition states* — shadow, holdback, staged ramp, seasonal coverage — because the revenue risk lives in the transition, not the destination (Module 08).

**Strong answer sketch:**
- Requirements: quantify why real-time first — what revenue does in-session reactivity add? If the answer is "maybe 2%," the migration must cost less than 2% of revenue in risk-adjusted terms. Latency budget, feature freshness needs, current batch cadence.
- Sequence: (1) dual-write features — stream pipeline runs alongside batch, parity-checked for weeks (feature parity is where these migrations die: point-in-time semantics differ between the batch join and the stream); (2) shadow mode — real-time system scores live traffic, predictions logged not served, offline comparison; (3) interleaving/small-slice A/B at 1% → 5% → 25% with **revenue and guardrail metrics wired to auto-rollback**; (4) staged ramp with a 5–10% long-term holdback on batch for a full seasonal cycle — Q4 traffic does not resemble June traffic, and the batch system is your control and your fallback; (5) decommission batch only after the holdback confirms no seasonal regression.
- Metrics: primary — revenue/session, conversion; guardrails — p99 latency, cost/request (real-time serving is 5–20× batch cost per prediction: state it), coverage (% requests real-time scored vs falling back).
- Cost math: batch precompute for 20M users daily might cost $3k/day; real-time scoring at 5k QPS with feature retrieval might cost $15–25k/day plus the stream infra — the lift must beat the delta.
- Risks: feedback-loop shift (real-time reacts to itself faster — monitor for popularity spirals); training-serving skew reappearing in the stream path; team burnout from running both systems (time-box the dual-run, put an end date in the plan).

**Red flags:** big-bang cutover; offline metrics as the go/no-go; no holdback; no seasonal argument; not asking whether real-time is even worth it.

### Q8. Design the data flywheel for a new AI product

**Principal vs senior:** Senior answers say "log user feedback and retrain." Principal answers design the event schema, the label-acquisition economics, the eval accretion path, and the consent boundary — before the first model ships (Module 06).

**Strong answer sketch:**
- Requirements: what user action constitutes signal? Explicit (thumbs, edits, accepts) is sparse but clean; implicit (accepted-without-edit, task-completed, escalated-to-human) is dense but noisy. Design the product UI so that the highest-value signal is a natural user action — e.g., "edit the draft" produces a correction pair for free.
- Architecture: full event record from day one (input, context, model version, output, user action, outcome) — schemas can be refined later, missing columns cannot; a labeling path that converts events → training pairs and events → eval cases (the flywheel feeds *both*, and the eval branch is the one teams forget); retrain cadence driven by data accrual and drift, not calendar; a curation gate — raw feedback is biased (dissatisfied users respond more) and needs sampling correction before training.
- Cold start: expert-labeled seed set (buy 5–10k labels, ~$20–50k) + synthetic augmentation; the flywheel replaces this within 1–2 quarters if instrumentation is right.
- Metrics: data accrual rate (usable pairs/week), label quality (agreement with expert sample), model improvement per 10k interactions, and the moat metric — how much better is our fine-tune than base on our distribution (this is the number that justifies the whole loop).
- Consent and privacy: user data → training requires explicit terms coverage, a per-user opt-out that propagates to training sets (deletion from a trained model is hard — design for exclusion at dataset construction), and region pinning.
- Risks: feedback loops amplifying model biases (Tay is the canonical adversarial case — Module 11); training on your own outputs (mark generated content, exclude); the flywheel that spins but doesn't compound because nobody closes the eval loop.

**Red flags:** "collect everything, figure it out later"; no consent design; flywheel feeds training but not evals; no cold-start plan; no curation/sampling-bias step.

### Q9. When would you NOT use ML?

**Principal vs senior:** Senior answers list textbook criteria. Principal answers reveal a portfolio discipline — they have personally killed ML projects and can name the decision rule and the dollar amounts (Modules 02, 09).

**Strong answer sketch:**
- Rules suffice: if 50 lines of deterministic logic hit 95% of the value, ML adds a training pipeline, drift risk, and an on-call burden to chase the last 5%. Example: shipping-fee anomaly detection — a threshold rule caught 92% of cases; the proposed model's incremental recall was worth ~$40k/yr against ~$300k/yr of ML lifecycle cost. Killed.
- No feedback loop exists: if you cannot measure outcomes, you cannot maintain the model — you're deploying a decaying asset with no gauge.
- Label economics fail: when labels cost more than the decisions are worth (rare, expensive-to-adjudicate events), or labels encode a bias you can't correct (proxy-label traps — cross-reference Module 11).
- Error costs are catastrophic and unreviewable: ML behind a human-review gate is fine; autonomous ML where a single error is irreversible and unbounded is not — the system design must cap the blast radius or the answer is no.
- The problem is a policy problem: if stakeholders disagree about what the *right answer is*, a model launders the disagreement instead of resolving it. Fix the policy first.
- Regulatory explainability: where adverse-action reasons are legally required, an uninterpretable model may be net-negative even if more accurate.
- The principal close: ML is a maintenance liability with a capability attached — Sculley's high-interest credit card. The question is never "can a model do this?" but "is the lifecycle cost of the model beaten by the lifecycle value?"

**Red flags:** treating it as a trick question; abstract criteria with no example where the candidate personally said no; not mentioning maintenance cost as the dominant term.

### Q10. Convince six teams to adopt one platform — you have no authority over any of them

**Principal vs senior:** Senior answers say "show them it's better." Principal answers name the mechanics: lighthouse adoption, pre-wiring, coalition sequencing, making the paved road genuinely superior, and reserving exec mandate as the last 10% (Module 13).

**Strong answer sketch:**
- Diagnose the resistance first — it is usually rational: migration cost is real, their stack works for them, and they've watched platform promises die before. Respect what came before; the 9 stacks exist because each solved a real problem at the time.
- Sequence: (1) pick the lighthouse — the team with the worst current pain (not the friendliest), give them white-glove migration support, and make them measurably faster; (2) publish the win with their numbers in their words — peer testimony beats platform-team slides; (3) pre-wire the next two adopters 1:1 before any group forum — collect objections privately, fix the top two in the roadmap, so the public RFC lands pre-socialized; (4) make the paved road better on the dimensions teams actually feel: onboarding time, on-call load, deploy speed — not architectural purity; (5) accept 5-of-6 for quarters; for the last holdout, either their edge case is real (platform gap — fix it) or it's inertia (now, and only now, exec air cover for a deadline).
- Metrics: adoption %, migration cost per team (platform team absorbs most of it — teams adopt what's cheap to adopt), post-migration incident and velocity deltas.
- Risks: lighthouse fails publicly (choose a migration you're 90% sure of; over-invest in it); platform team credibility spent on a mandate too early (a forced adoption that goes badly poisons the next three).

**Red flags:** "get the VP to mandate it" as the first move; assuming resistance is irrational; no lighthouse; no mechanism beyond "documentation and demos."

### Q11. Tell me about your biggest bet that failed — and the system-level lesson

**Principal vs senior:** Senior answers pick a safe failure and a personal lesson ("I learned to communicate more"). Principal answers pick a bet with real cost, own the decision rather than the circumstances, identify the signal they ignored, and show the *organizational* control that now exists because of it (Modules 11, 14).

**Strong answer sketch:**
- The bet, with numbers: "I championed building a real-time feature platform in 2023 — 5 engineers, 3 quarters, ~$1.8M — on the thesis that four teams would need sub-second features within a year."
- The failure, owned: "Demand didn't materialize: one team needed it, two found batch adequate, one built their own anyway. I had extrapolated from the most enthusiastic team's roadmap and treated polite interest as commitment. The signal I ignored: no team would commit engineers to co-develop — revealed preference was telling me the demand was soft."
- The unwind: "We descoped to serve the one real customer, cut the team to 2, and I wrote the postmortem myself and presented it to the org."
- The system-level lesson — the part that separates principal: "Every infrastructure bet I sponsor now requires a committed design partner who stakes engineering time before we staff it, and carries kill criteria set at inception — a demand milestone at the 1/3 mark that triggers an explicit continue/descope/kill review. That rule has since killed one bet early (~$400k spent instead of $2M) and validated two others."
- Close on Are-Right-A-Lot honesty: the goal isn't zero failed bets — a portfolio with no failures is under-ambitious — it's failing at 1/3 cost with the learning captured.

**Red flags:** a fake failure ("we succeeded but too slowly"); blaming leadership, market, or another team; a personal-habit lesson with no organizational control; no numbers.

### Q12. Design governance for ML at a bank — without killing velocity

**Principal vs senior:** Senior answers add review gates. Principal answers design risk *tiering*, automate evidence generation inside the platform, and embed the second line early — governance as a property of the paved road, not a checkpoint at the end (Module 12 has the full treatment; this is the interview cut).

**Strong answer sketch:**
- Requirements: banks already have Model Risk Management under SR 11-7 / ECB TRIM equivalents — do not design as if greenfield; the job is making the existing second line fast, and extending it to ML/GenAI sensibly.
- Risk tiering is the whole game: tier 1 (credit decisioning, AML, capital models) — full independent validation, documented conceptual soundness, ongoing monitoring, adverse-action explainability; tier 2 (fraud scoring, collections optimization) — standardized validation template, sampled review; tier 3 (internal tooling, marketing ranking) — self-serve checklist, registered but not gated. Most orgs fail by running tier-1 process on tier-3 models, which teams then evade — creating *shadow ML*, which is the worst governance outcome.
- Velocity mechanisms: (1) the platform auto-generates the evidence — lineage, training-data snapshots, eval results, monitoring configs are captured by the registry and pipeline as side effects of normal work, so "compliance documentation" is a export button, not a quarter of writing; (2) model-risk partners embedded at design time (an hour at kickoff beats a rejection at month 6); (3) pre-approved pattern library — a fine-tuned classifier on the standard stack with standard monitoring is pre-cleared as a template, and only deviations need review; (4) SLA on validation itself (tier-2 review in 10 business days, tracked and reported like any service).
- Metrics: time-from-model-ready-to-production by tier (the velocity number), % models registered (the shadow-ML detector), validation findings per model (quality), audit findings (the board number).
- Risks: GenAI doesn't fit classic MRM validation (define eval-card-based validation for LLM surfaces now, before the regulator does it for you); governance theater (documents nobody reads — the monitoring requirement is the part with teeth).

**Red flags:** one process for all models; ignoring that SR 11-7 exists; governance bolted on after deployment; no shadow-ML story; framing compliance and velocity as zero-sum.

## 3. Behavioral questions — the principal answer pattern

Every principal behavioral answer has five beats. Interviewers are trained to probe for whichever beat is missing:

1. **Situation at org scope** — numbers that establish the blast radius: teams, dollars, users, timelines. "Our recsys" is senior; "the ranking systems carrying $140M of GMV across three teams" is principal.
2. **Decision under ambiguity** — the moment where the data was incomplete, the experts disagreed, and *you* made the call. Name what you didn't know at decision time.
3. **Influence mechanics** — how agreement actually happened: pre-wiring, RFCs, lighthouse projects, coalition order. "I convinced them" is a claim; the mechanism is the evidence.
4. **Measurable business outcome** — a number, with the counterfactual stated honestly ("about half the lift is attributable to the migration; the rest was seasonal").
5. **Systemic lesson** — the process, standard, or artifact that now exists so the org doesn't depend on you repeating the heroics.

### The ten questions to prepare

1. Tell me about a time you influenced a major decision you had no authority over.
2. Tell me about a time you strongly disagreed with leadership. What happened? (Have both variants: one where you were right, one where you committed and were wrong.)
3. What's the most complex technical problem you've solved? (They're listening for whether the complexity was partly organizational.)
4. Tell me about a project you killed — especially one you had championed.
5. Describe a conflict between two teams you resolved.
6. Tell me about a time you raised the engineering bar beyond your own team.
7. Tell me about a failure with real cost. What changed because of it?
8. How have you grown senior engineers into staff-level engineers?
9. Tell me about a high-stakes decision you made with incomplete data under time pressure.
10. What's the biggest technical bet your current org should make in the next two years, and why isn't it happening? (Forward-looking judgment — answer with a thesis, options, and a kill criterion, per Module 14.)

Prepare 6–8 stories that cover all ten (stories are reusable across questions); write them out; rehearse the numbers until they're automatic. A hesitation on your own metrics reads as fabrication.

### Worked example 1 — influence without authority (Q1 pattern)

**S:** "In 2024 I was the senior-most IC across a 60-engineer product org running four separate model-serving setups; duplicate infra cost roughly five engineer-years annually and tier-1 incidents were taking 4+ hours to resolve because each stack had different observability."
**T:** "No one asked me to fix it — serving wasn't 'mine.' I decided consolidation was the highest-leverage problem in the org and that I was the only person positioned to drive it across team lines."
**A:** "I wrote a one-page problem memo with the cost numbers before proposing anything. I pre-wired the four team leads individually — two were supportive, one neutral, one strongly opposed because their latency requirements were real. I incorporated their P99 constraint into the target design as a named requirement, which converted the opposition into co-authorship — the final RFC had their lead as a reviewer. We migrated the team with the worst on-call pain first, cut their incident resolution time from 4 hours to 40 minutes, and published their numbers. The remaining teams adopted over two quarters; no mandate was ever needed."
**R:** "Consolidation to one stack plus a batch path saved ~4 engineer-years/yr, cut tier-1 MTTR by 70%, and the RFC-plus-prewire pattern became the org's default for cross-team changes."
**Systemic lesson:** "I learned that opposition usually encodes a real requirement — the fastest path through resistance is to find the requirement and design for it. I now treat 'who is most opposed' as the first stakeholder interview, not the last."

### Worked example 2 — the failed bet (Q7/Q4 pattern)

**S:** "In 2023, I sponsored an internal LLM fine-tuning platform — 4 engineers, two quarters, roughly $1.2M loaded — on the thesis that six product teams would fine-tune within the year."
**T:** "I set the roadmap and made the staffing case to the VP personally."
**A:** "Three months post-launch, one team had fine-tuned; the others found prompted frontier APIs sufficient for their quality bars, which had improved faster than my thesis assumed. The failure signal I had ignored: no team committed engineers as design partners — I took survey enthusiasm as demand. I wrote the postmortem myself, presented it at the eng all-hands, descoped the platform to a thin wrapper for the one real user, and reassigned three engineers to the eval infrastructure that teams *were* asking for."
**R:** "We spent $1.2M to learn a $300k lesson — the delta is on me. The eval infra those engineers built became the most-adopted platform component within two quarters."
**Systemic lesson:** "Every platform bet I've sponsored since requires a committed design partner staking real engineering time before staffing, and a demand checkpoint at one-third budget with explicit continue/kill criteria. That rule killed one subsequent bet at $400k instead of $2M. Public ownership of the postmortem also turned out to be the cheapest credibility purchase available — the VP cited it in my next review as the reason she trusted me with a bigger bet."

### Worked example 3 — decision under ambiguity with incomplete data (Q9 pattern)

**S:** "Eleven days before Black Friday, our fraud model's precision dropped 6 points — false positives were blocking ~$120k/day of legitimate orders, and the retrain pipeline was producing models that failed offline checks for reasons we didn't yet understand."
**T:** "As the principal on the incident, I had to choose between shipping a hastily-fixed model into peak season, loosening thresholds and eating fraud, or freezing on the degraded model."
**A:** "We had maybe 60% of the diagnostic picture: an upstream data change was implicated but not confirmed. I ruled out shipping an unvalidated model into our highest-traffic week — irreversible downside, peak blast radius. Instead I split the decision: threshold override with a manual-review queue for the gray zone (staffed by borrowing 6 people from support, pre-cleared with their director in one conversation), which capped both the false-positive losses and fraud exposure, while the team ran root-cause under less pressure. I set an explicit revisit date: if root cause landed before Black Friday minus 3 days, we'd ship the fix behind a 5% canary; otherwise we'd ride the override through the peak."
**R:** "Root cause landed in 6 days — a feature pipeline schema drift — the canary validated, and we shipped 4 days before peak. Blocked-revenue losses were held to ~$300k versus a projected $1.3M for freezing, with no measurable fraud increase."
**Systemic lesson:** "Two changes: schema contracts with breaking-change alerts on every tier-1 feature pipeline, and a standing 'peak-season change freeze plus override playbook' so the next incident decision is a lookup, not an improvisation. The meta-lesson I offer when asked: under time pressure, decompose the irreversible decision into a reversible one plus a deadline."

## 4. Negotiation and leveling notes

**Down-leveling is the default failure mode, not rejection.** Committees resolve ambiguity downward: a strong-but-unclear principal packet becomes a staff/senior offer. This is usually framed as "join at L6 and you'll be promoted quickly" — treat that promise as worth roughly zero unless it comes with written scope commitments; internal promotion to L7/E7 is often *harder* than interviewing in at that level, because now you need sustained internal evidence plus a promo cycle plus org headroom.

**How packet evidence maps to level.** Committees pattern-match on three axes: **scope** (team → org → company: whose roadmap changed because of you?), **duration** (a single 6-month project caps at senior evidence; 2–3 years of compounding initiatives reads staff/principal), and **artifact trail** (strategy docs, standards, RFCs, postmortems that outlived you — things other people still use). Audit your own stories against these axes before the loop: if every story is 1 team × 2 quarters × heroic execution, you will be leveled senior no matter how well you interview. The capstone in Module 16 exists precisely to make this artifact trail tangible.

**Interview-day tactics that affect leveling.** Say the org-scale things out loud even when unprompted — interviewers can only write down what you said. Attach numbers and durations to every claim. When asked a system-design question, spend visible time on adoption, cost, and org mechanics; that's what distinguishes the write-up. Ask each interviewer "what does principal look like on your team?" — it signals you're calibrating for the level and gives you material for the next session.

**Negotiating the level, not just the comp.** If the offer comes in a level low: ask what specific evidence was missing (recruiters will often tell you), offer supplementary evidence (a strategy doc you can share sanitized, an extra reference who saw your org-scope work), and be willing to interview again in 6–12 months rather than accept a level that takes 3 years to fix internally. Comp negotiation at these levels moves mostly on equity and on competing offers; level negotiation moves on evidence. They are separate conversations — have them separately.

## How to drill

**Mock cadence.** Six weeks out: one system-design/strategy mock per week from the twelve above, rotating genres (platform design → strategy case → math-heavy → governance), with a partner who has sat on senior loops if at all possible. Three weeks out: add one behavioral mock per week; deliver your worked stories against a timer (4 minutes per story, then survive 10 minutes of probing). Solo weeks: written skeleton drills — 25 minutes to produce the requirements→metrics→architecture→tradeoffs→cost→risks skeleton for one question, no prose, then compare against the sketches above.

**Self-grading rubric — score every mock 0–2 on each:**

1. Did I spend the first 10 minutes on requirements and state my assumptions as assumptions?
2. Did I define metrics — including for the *platform/org*, not just the model — before architecture?
3. Did I do cost math out loud with stated planning numbers, including headcount?
4. Did I name at least two organizational mechanisms (adoption, funding, pre-wiring, ownership) — not just components?
5. Did every claim carry a number or a duration?
6. Did I state risks with mitigations and at least one kill/rollback criterion?
7. (Behavioral) Did the story hit all five beats, and did the systemic lesson name an artifact that outlived me?

Below 10/14, drill the specific missing beats — do not just run more full mocks. The failure pattern to watch in yourself: under pressure, everyone regresses to the level they actually operate at. The only durable fix is doing the work — which is what Module 16 is for.
