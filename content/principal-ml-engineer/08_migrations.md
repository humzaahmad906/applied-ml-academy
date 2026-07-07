# Module 08 — Migrations & System Evolution

## Why this module matters

Greenfield design is the easy mode of engineering: no users, no legacy behavior, no one depending on your bugs. The defining principal-level project is the other thing — changing the engine while the plane is flying, without the passengers noticing, and then actually removing the old engine instead of bolting the new one alongside it forever. Every ML org more than three years old is mid-migration somewhere: monolith scoring to platform serving, homegrown features to the feature store, one model family to another, one cloud to another. Migrations are where principal judgment is most visible because they are long, unglamorous, statistically subtle, and politically loaded — and because a failed migration costs more than most failed launches. This module gives you the playbook, the parity-metrics discipline that makes ML migrations different, the economics, and the org mechanics that determine whether the old system ever actually dies.

## 1. Migrations are the job

Will Larson's observation holds: at large companies, migrations are effectively the only mechanism for making cross-cutting technical change, and running them well is a defining Staff/Principal skill. In ML orgs the stakes are higher because the systems being migrated hold *behavioral* state — a fraud model in production embodies years of threshold-tuning, downstream rule accretion, and consumer adaptation that no design doc records. The senior engineer sees a migration as a technical project; the principal sees it as three projects stacked: a technical project (build the new path), a statistical project (prove the new path is equivalent-or-better where it must be), and an organizational project (get every consumer to move and every stakeholder to keep funding it through the boring middle). Most migrations fail on the second or third, not the first.

A field guide to the migrations you will actually be handed, with honest planning shapes (team sizes assume the owning platform/product split stays intact):

| Migration | Typical shape | Where it dies |
|---|---|---|
| Batch scoring → real-time serving | 4–6 eng, 9–15 mo | consumers of the batch output tables |
| Homegrown features → feature store | 2–4 eng, 2–4 quarters *per team migrated* | point-in-time semantics mismatches |
| Model family swap (e.g., GBM → NN, or LLM vendor A → B) | 2–3 eng, 1–2 quarters | behavioral long tail; prompt/threshold folklore |
| Serving stack consolidation (N stacks → 1) | 5–8 eng, 12–24 mo | the stack owned by the team that "just needs one more quarter" |
| Cloud/region move | infra-led, ML rides along, 12–18 mo | data gravity: training sets, feature history |
| Monolith training pipeline → orchestrated platform | 3–5 eng, 6–12 mo | undocumented preprocessing steps |

The right column is the planning input seniors skip: every migration type has a characteristic kill zone, and the plan should name it on page one.

The other reason migrations belong to principals: they are the moments of maximum optionality. A serving migration is your once-per-five-years chance to fix the logging schema, kill three undocumented side channels, and install the eval gates from Module 07. Migrating a system "bug-for-bug compatible" and *then* improving it is usually the right sequencing (one variable at a time), but the improvements must be in the plan from day one or the funding evaporates the moment parity is reached.

## 2. Why ML migrations are harder than software migrations

A software migration ends with a proof: the test suite passes on the new stack; behavior is equivalent by construction. An ML migration cannot end that way, because the system's behavior is statistical. Move a model from batch Spark scoring to real-time serving and *everything shifts slightly*: feature values are fresher (different, not wrong), float behavior differs across hardware and libraries, preprocessing is reimplemented with subtly different tokenization or null handling, and traffic arrives in a different order. There is no unit test that proves the new system is "the same" — there is only a *distributional* argument: shadow-score real traffic on both stacks, compare the prediction distributions, agreement rates, and downstream metrics, and define in advance how much divergence is acceptable and where.

This inverts a software instinct. In software migrations, any behavioral diff is a bug. In ML migrations, some diffs are bugs (a feature computed wrong), some are improvements (fresher features), and some are neutral noise (float nondeterminism) — and telling them apart is analytical work that must be budgeted. Three consequences for the plan:

1. **Shadow phase is mandatory, not optional.** You cannot reason your way to parity; you must measure it on production traffic, at production scale, across enough time to cover weekly seasonality (minimum two weeks, longer for systems with monthly cycles like billing fraud).
2. **Parity gates are statistical gates.** "Predictions match" becomes "prediction-level agreement ≥ X on slice S, score-distribution PSI ≤ Y, and downstream metric neutral within a pre-registered CI" — numbers chosen *before* the shadow phase, or the team will rationalize whatever divergence it finds.
3. **Divergence triage is a workstream.** Plan for a standing "diff court": a weekly session where the largest divergence buckets are root-caused into bug / improvement / noise. In practice this consumes 20–30% of the migration's engineering time and produces most of its bug discoveries — including, routinely, bugs in the *old* system that everyone had adapted to.

### The diff court

Because divergence triage is where the migration's truth gets decided, run it as a standing ritual with a written record, not an ad hoc Slack thread. Weekly, one hour, fixed attendees (migration DRI, one engineer from each stack, the model owner). Agenda: the top divergence buckets by traffic contribution, each ending in a signed disposition:

```text
DIFF DISPOSITION — bucket #23                       date / owner
POPULATION   requests where category_affinity differs > 0.2  (1.8% of traffic)
ROOT CAUSE   new path computes 24h window; old path 7d (spec transcription error)
CLASS        BUG (new stack)
ACTION       fix window to 7d; re-shadow bucket; verify τ recovers
STATUS       fixed 2026-03-14; bucket agreement 99.1% post-fix
```

The disposition log becomes the migration's evidence base: at cutover review, "every divergence bucket above 0.1% of traffic has a written disposition" is a sentence that ends arguments. It is also the artifact that protects the team a year later when someone asks why the new system behaves differently on some slice — the answer is on file, with a date and a reason.

That last point deserves emphasis: the old system is not ground truth, it is merely incumbent. Every long-lived ML system contains "load-bearing bugs" — a feature that has been silently null for two years, a threshold tuned to compensate for a preprocessing error. The parity decision for each is genuinely hard: replicate the bug (safe, preserves downstream adaptations, embarrassing) or fix it (correct, but now you're changing behavior and parity gates will fire). Default: replicate first, fix in a controlled follow-up experiment after cutover. One variable at a time.

## 3. The migration playbook

Six phases. Each has an entry criterion, an exit criterion, and a deliverable. Skipping a phase is how migrations become multi-year zombies.

| Phase | Core activity | Exit criterion | Deliverable |
|---|---|---|---|
| 0 Inventory | Find every consumer, incl. undeclared | Graph reviewed by all named teams | Dependency graph + disposition per consumer |
| 1 Sequencing | Order traffic slices by risk/representativeness | First slice + owner committed | Ordered slice plan with risk notes |
| 2 Shadow | Dual-score, compare distributions | Parity gates green ≥14 consecutive days | Parity report |
| 3 Canary | Ramp with automated rollback | 100% of slice, guardrails neutral, 1 clean weekly cycle | Wired rollback triggers + canary log |
| 4 Cutover | Flip default, repeat per slice | All slices on new stack | Per-slice cutover records |
| 5 Deprecation | Remove access in stages, delete | Old system's bill is $0, code archived | Tombstone doc (graph + parity report) |

**Phase 0 — Inventory and dependency graph.** Before anything: enumerate every model, pipeline, feature, and — hardest — every *consumer* of the system being replaced. Sculley et al. (NeurIPS 2015) named the pattern: undeclared consumers. Someone's dashboard reads your scoring table; a downstream team trained a model on your model's outputs; an analyst's weekly report joins against your predictions. None of them appear in any config. Methods that actually find them: query logs on the output tables (90 days minimum — monthly consumers exist), access logs on the API, org-wide code search for the table/endpoint names, and a loud deprecation-warning header that logs callers. Deliverable: a dependency graph with every consumer classified as *migrate / notify / break deliberately*. Exit criterion: the graph reviewed by every team that appears on it. Expect the inventory to take 2–4× longer than the optimistic estimate; it always does, and it is the highest-information phase.

**Phase 1 — Sequencing (strangler fig for ML).** The strangler-fig pattern — stand up the new system around the edges of the old and grow it until the old is enclosed — applies to ML with one crucial adaptation: **route by traffic slice, not by code path.** Software stranglers carve by endpoint or module; ML systems can't be carved that way because a model is indivisible. Instead, carve the *traffic*: one country, one product category, one customer tier at a time. Each slice gets the full new path end-to-end, gets validated with the parity gates, and becomes a stable beachhead. Sequencing principles: start with a slice that is (a) low blast-radius, (b) *representative enough* to exercise the hard parts — a slice so easy it proves nothing is a common trap — and (c) owned by a friendly team. Deliverable: an ordered slice plan with per-slice risk notes. Exit: first slice chosen and its owning team signed up.

**Phase 2 — Shadow (dual-score everything).** Both stacks score all traffic (or the target slice); only the old stack's outputs are served. Log both, joined by request ID, into a comparison table. Compare *distributions, not just point metrics*: score histograms and PSI/KL by slice, prediction-level agreement, rank correlation for ranking systems, latency p50/p99/p999, error and timeout rates, feature-value deltas for the top-50 features. Run the diff court weekly. Exit criterion: all pre-registered parity gates green for N consecutive days (N ≥ 14), all divergence buckets above the triage threshold dispositioned in writing. Deliverable: the parity report — this document is what you'll point to when someone questions the cutover in month nine.

**Phase 3 — Canary with automatic rollback.** Serve the new stack's outputs to 1% → 5% → 25% of the slice, with *automated* rollback triggers wired before the first request: "if guardrail metric X regresses more than Y% over Z hours vs. concurrent control, route back to old stack, page the team." Automated matters — human-in-the-loop rollback during an incident adds 30–90 minutes of debate exactly when you can't afford it, and the person on-call at 3 a.m. shouldn't have to make a judgment call the team could have made in advance. Each ramp step holds long enough to detect the effects you care about (power math from Module 07 applies — 1% canaries only detect large regressions; that's what they're for). Trigger-design rules that separate useful automation from pager noise:

- **Compare against concurrent control, never against last week.** Absolute thresholds fire on seasonality; relative-to-control thresholds fire on the migration.
- **Two speeds:** a fast trigger on operational metrics (error rate, latency — minutes) and a slow trigger on behavioral metrics (approval rate, CTR — hours, needs volume). Most teams wire only the fast one and discover the behavioral regression from a stakeholder email.
- **Rollback must be cheaper than deliberation.** If routing back takes one config flip and 60 seconds, you can afford sensitive triggers; if rollback is itself a deploy, every trigger becomes a debate. Invest in the flip before the ramp.
- **Every firing is logged and reviewed** — including false positives, which are tuning data, not embarrassments.

Exit: 100% of slice served by new stack, guardrails neutral, one full weekly cycle clean.

**Phase 4 — Cutover and repeat.** Flip the slice's default, keep the old stack warm as instant rollback for one more cycle, then move to the next slice. Later slices go faster — the playbook is proven — but resist the urge to skip shadow on "similar" slices; the third slice is where the null-handling bug that only manifests on international addresses lives. Exit: all slices cut over.

**Phase 5 — Deprecation: actually delete the old thing.** The hardest phase, and where most migrations silently fail: the org declares victory at cutover, the team disbands, and the old stack runs for three more years "just in case," costing infra money and — worse — remaining a thing every future change must stay compatible with. Running both stacks indefinitely is the worst outcome: you paid for the migration *and* kept all the old costs. Discipline: deprecation is in the definition of done from day one, with a date leadership has committed to; the undeclared consumers found in Phase 0 each get a migration path and a break date; access is removed in stages (write-freeze → read-alerts → 4xx with a grace override → gone); and the final deletion is celebrated as loudly as a launch, because the org needs to learn that deletion is a deliverable. Exit criterion: the old system's compute bill is zero and its code is archived.

## 4. Parity metrics design

The parity gate is a designed artifact. Three layers, all pre-registered:

**Prediction-level agreement.** For classifiers: decision agreement rate at the production threshold (not just score correlation — scores can correlate at 0.98 while decisions flip on 4% of traffic if mass sits near the threshold, and threshold-adjacent traffic is precisely the interesting traffic). For rankers: rank correlation (Kendall's τ) on the top-k, and top-1/top-3 overlap, since users only see the head. For regressors/scores: distribution PSI ≤ 0.10 overall, plus quantile-by-quantile deltas. Set gates per slice, not just globally — global agreement of 97% can hide a slice at 70%.

**Downstream business-metric neutrality.** Agreement gates catch behavior change; they don't tell you whether the change matters. The canary phase's real gate is business-metric neutrality within a pre-registered interval: e.g., "approval rate within ±0.3pp of concurrent control, chargeback rate ratio in [0.95, 1.08] at 95% confidence." Neutrality intervals must be honest about power — proving a null needs more data than detecting an effect; state the interval you can actually resolve at canary traffic volumes.

**Tail behavior.** Averages hide the failures that page you. Explicitly gate: p999 latency, the agreement rate on the 1% most extreme scores (both tails), behavior on the rare-but-critical slices from the Phase 0 inventory (the fraud model's behavior on $10k+ transactions matters more than its average), and null/degenerate-input handling (empty carts, brand-new users, malformed requests — the new stack *will* handle these differently until proven otherwise).

The gates are pre-registered as a short spec, signed before shadow starts. A minimal example for a binary classifier migration:

```text
PARITY GATE SPEC — fraud-scorer v2 stack migration        signed: <DRI, date>
Shadow window: ≥21 days (covers monthly billing cycle)
GLOBAL
  decision agreement @ prod threshold          ≥ 98.5%
  score distribution PSI                       ≤ 0.10
  score quantile deltas (p1..p99)              |Δ| ≤ 0.02 at every decile
PER-SLICE (each of: top-5 countries, new-users, tx > $10k, mobile)
  decision agreement                           ≥ 97.0%
  PSI                                          ≤ 0.15
OPERATIONAL
  p99 latency                                  ≤ 45 ms   (old stack: 38 ms)
  p999 latency                                 ≤ 120 ms
  error/timeout rate                           ≤ 0.2%
TRIAGE RULE
  any bucket contributing > 0.1% of disagreements → root-caused in diff
  court, dispositioned in writing as bug / improvement / noise
CANARY NEUTRALITY (business gate, per ramp step)
  approval rate within ±0.3pp of concurrent control
  chargeback-rate ratio in [0.95, 1.08] at 95% CI (powered at 25% step only)
ROLLBACK (automated)
  approval rate deviates > 1pp for 4h, OR p99 > 60 ms for 30 min,
  OR error rate > 1% for 10 min → route to old stack + page
```

Two properties make this document valuable: it is *falsifiable* (every line is a number a dashboard can evaluate), and it is *signed before data arrives* — which is what prevents the month-4 conversation where a team under deadline pressure argues that 96% agreement is "basically fine."

## 5. The long tail: the last 5% takes 50% of the calendar

Every migration's traffic chart looks the same: 80% migrated in the first 40% of the timeline, then an agonizing crawl. The tail-consumer triage, made explicit:

```text
for each remaining consumer:
  can they move with < 1 eng-week of their time?   → MIGRATE (you supply the guide)
  is their requirement real but new-stack-hostile? → ACCOMMODATE (shim; price its
                                                     permanent cost; get it approved)
  is the consumer low-value or unowned?            → BREAK (2 warnings + date + VP air cover)
  none of the above (big, blocked, political)      → ESCALATE with options + costs;
                                                     do not let it silently anchor the calendar
``` The last slices are last for reasons — the customer with a contractual SLA and a custom integration, the country with a data-residency constraint the new stack doesn't support yet, the internal consumer whose owning team was disbanded. Plan for it explicitly: budget the calendar asymmetrically (if the plan says 12 months, slices 1–8 get months 1–6 and the last two slices get months 7–12); triage the tail into *migrate / accommodate / break* — for some consumers the correct answer is a negotiated breakage with notice, and for one or two it may be a permanent shim (a thin compatibility adapter on the new stack) whose ongoing cost you accept and record; and watch for the tail's morale problem — the team that heroically shipped the first 80% is bored and poachable during the last 20%, which argues for rotating the closing crew and making tail-slice completion visibly valued in perf.

## 6. Migration economics

A migration is an investment with a payback period, and the principal owns the honest math:

$$\text{total\_cost} = \text{build\_cost} + \text{dual\_running\_cost} \times \text{overlap\_months} + \text{divergence\_triage\_cost} + \text{consumer\_migration\_cost}$$

$$\text{payback} = \frac{\text{total\_cost}}{\text{monthly\_benefit}} \qquad \text{(infra savings + velocity + risk)}$$

*Dual-running is the term teams most often forget; divergence triage consumes 20–30% of engineering time and belongs on its own line; consumer-migration cost is real money from other teams' budgets.*

Worked instance, spelled out:

```text
COSTS
  build             5 eng × 12 mo × $30k/mo loaded            $1 800k
  dual-running      (old $70k + new $50k)/mo × 9 mo overlap   $1 080k
  divergence triage (already inside eng time, but name it:
                     ~25% of the 5-eng year)                   (~$450k of the build)
  consumer teams    14 consumers × avg 0.5 eng-mo × $30k      $  210k
  contingency       15% on build (migrations slip)            $  270k
  TOTAL                                                       ≈ $3 360k

BENEFITS
  infra retirement       $70k/mo → $840k/yr
  incident class retired 9 stale-recs incidents/yr × est $25k  $225k/yr
  velocity on new stack  conservatively booked at $0 (real,
                         but don't let the spreadsheet depend on it)
  PAYBACK  ≈ $3.36M / ($1.07M/yr) ≈ 3.1 years
```

Payback in the 2–3 year range is typical and fine; what is not fine is discovering the dual-running line in month eight. Note the deliberate conservatism: velocity benefits are real but unquantifiable, so book them at zero and let them be upside — a migration justified *only* by soft velocity claims is a migration that gets defunded at the first budget review.

Two disciplines follow. **Time-boxing:** every phase gets a calendar budget, and blowing a budget by >50% triggers a go/no-go review, not a silent extension — zombie migrations (year three, 60% done, team of two, no end date) cost more than either finishing or killing, because they consume the org's change budget while delivering neither state. **Declaring bankruptcy:** sometimes the right decision is to stop — freeze the legacy system (no new features, security patches only, a named owner, an explicit "this is permanent until conditions X change" doc) and route only *new* use cases to the new stack. This is a legitimate outcome, strictly better than a zombie, and it takes principal-level standing to say out loud, because it means writing down sunk cost in public. The Phase 0 economics should name the bankruptcy conditions in advance ("if consumer migration cost exceeds $X or slice 4 slips past date Y, we freeze instead").

## 7. Org mechanics: migration as a product

The technical playbook fails without the org playbook:

- **The migration has users, so run it as a product.** Every consumer team is a customer; their migration cost is your product's price. Drive the price down: migration guides, codemods/adapters, office hours, and doing the first consumer's migration *for them* to debug the path. Larson's rule of thumb: if you want teams to move, make moving nearly free.
- **Comms cadence.** A visible dashboard (slices migrated, consumers remaining, gate status) updated weekly, a monthly stakeholder note, and a single named DRI. Migrations die of silence: six quiet weeks and leadership assumes it stalled, and the funding conversation starts.
- **Forcing functions leadership actually backs.** A deprecation date only works if, when the date arrives and two teams haven't moved, leadership holds it. Get the commitment *in writing before starting* — "when we hit the date, laggards get broken after two warnings" — because asking for enforcement for the first time at the deadline is asking to be overruled. Softer forcing functions to layer first: freeze feature development on the old stack (change requests answered with "that ships on the new stack"), charge old-stack infra costs back to the holdout teams, put remaining consumers on a list reviewed in the VP's staff meeting.
- **Credit flows to closers.** Perf systems naturally reward the greenfield 80%; the principal makes sure the engineers who ground through the tail and deleted the old system get equal-or-better recognition. This is not sentiment — it is incentive design for the next migration.

All of it condenses into a one-page charter that leadership signs before engineering starts. If you cannot fill this page, you are not ready to begin:

```text
MIGRATION CHARTER — <from X to Y>                DRI: ____   sponsor: ____
WHY NOW          the cost of not migrating, in $/mo and incidents/yr
DEFINITION OF DONE   old system deleted (bill = $0), not "new system live"
SCOPE            systems in / systems explicitly out
CONSUMERS        N total; migrate ___ / accommodate ___ / break ___
SLICE PLAN       ordered list, first slice + owning team named
PARITY GATES     link to signed gate spec (pre-registered)
ECONOMICS        build $___ + dual-running $___/mo × ___ mo; payback ___ mo
CALENDAR         per-phase budgets; tail slices get ≥40% of the calendar
BANKRUPTCY CONDITIONS   freeze-instead-of-finish triggers, named in advance
LEADERSHIP COMMITMENTS  deprecation date ____ will be enforced after 2
                 warnings; old-stack feature freeze from date ____ (signed)
COMMS            weekly dashboard link; monthly stakeholder note; escalation path
```

The two lines that earn their place: *definition of done* (cutover is not done; deletion is done) and *leadership commitments* (extracted in writing while enthusiasm is high, spent later when a laggard team escalates).

## You can now

- Identify the characteristic kill zone for the migration type you've been handed and name it on page one of the plan, before any code is written.
- Design shadow parity gates as pre-registered statistical specs — prediction-level agreement, score distribution PSI, tail-behavior thresholds, and operational SLOs — signed before shadow data arrives so no team can rationalize divergence after the fact.
- Run the diff court weekly and produce signed written dispositions (bug / improvement / noise) that make the cutover review a presentation of evidence rather than an argument, and put every load-bearing bug decision in writing at the moment it's made.
- Size migration economics honestly with build cost, dual-running overlap, divergence-triage cost, and consumer-migration cost — then name bankruptcy conditions in advance so a zombie migration cannot form silently.
- Extract leadership commitments in writing (deprecation date enforcement, old-stack feature freeze, VP air cover for breaking laggards) before engineering starts, and use layered forcing functions to close the tail without relying on goodwill at the deadline.

## Worked example

**Setting.** "Cartwheel," a marketplace at ~30M MAU. The recommendations system scores nightly on a 7-year-old Spark monolith: 40M user embeddings refreshed nightly, top-500 candidates per user written to a key-value store, served as-is all next day. The company has built a real-time serving platform (Module 03/05 work) with a feature store and streaming features. The prize: in-session responsiveness (react to the last 3 clicks), projected +2–4% conversion on recs-driven GMV of $800M/yr, plus retiring the monolith ($95k/mo infra, 1.5 SRE-equivalents of toil, and a 6-hour nightly job whose failure means stale recs company-wide — it failed 9 times last year). Team: 6 engineers, DRI: you. Plan: 12 months. Leadership signs the economics: build $2.2M, dual-running $85k/mo × ~8 months, payback ~26 months counting infra + a conservative 1% conversion lift.

**Phase 0 — Inventory (6 weeks; planned 3, took 6 — as predicted).** Query logs on the recs KV store and output tables over 90 days find **14 consumers**, of which 5 were undeclared: the email-marketing pipeline reads the top-500 table to build daily digest emails; the ads team trains a lookalike model on rec scores; a BI dashboard computes "catalog coverage" from the table; the mobile team caches top-50 on-device from a bulk endpoint; and a fraud heuristic uses "item was in user's recs" as a feature. Classification: email and mobile *migrate* (they get new API endpoints), BI *migrate* (new metrics table), ads *notify + break deliberately* (training on another model's scores is the Sculley correction-cascade anti-pattern; they agree to move to raw engagement features, 1 quarter of their time, negotiated at the VP level), fraud *accommodate* (a shim endpoint, permanent, documented, $800/mo). Deliverable: dependency graph reviewed with all 6 owning teams. Entry to Phase 1: graph signed off.

**Phase 1 — Sequencing (2 weeks).** Traffic slices by product surface and geography. Order: (1) "similar items" shelf in New Zealand + Ireland (0.9% of traffic, full stack exercised, English-language, friendly regional team), (2) same shelf globally, (3) homepage feed AU/CA, (4) homepage feed US/EU (68% of GMV), (5) email digest, (6) mobile bulk endpoint. Rationale recorded per slice. Homepage-US is deliberately late but not last — email and mobile are lower-risk and can trail while the team is still assembled.

**Phase 2 — Shadow on slices 1–2 (10 weeks).** Dual-score: nightly batch keeps serving; real-time path scores the same requests and logs to a comparison table keyed by request ID. Pre-registered gates: Kendall's τ ≥ 0.80 on top-50, top-10 overlap ≥ 70%, score PSI ≤ 0.10 per slice, p99 latency ≤ 120ms, error rate ≤ 0.3%. First-week reality: τ = 0.66, top-10 overlap 58%. Diff court triage over 4 weeks buckets the divergence: **(a) bug** — the real-time path's category-affinity feature used a 24h window vs. the batch 7d window, a spec transcription error (fixed, τ → 0.74); **(b) old-system bug** — batch had been silently dropping items added to catalog after 11 p.m. from candidate generation for years (~2% of items; decision: do *not* replicate; document as an intended improvement and exclude affected requests from the parity gate, since parity against a bug is not the goal — this is the replicate-vs-fix call made explicitly, in writing); **(c) intended freshness** — recent-click features genuinely differ; requests with in-session activity are gated separately with looser τ ≥ 0.65, since divergence there is the point. Gates green for 16 consecutive days in week 9. Parity report published. The headline table from that report:

| Gate | Threshold | Week 1 | Week 5 (post-fix) | Week 9 (final) |
|---|---|---|---|---|
| Kendall's τ, top-50 (no session activity) | ≥ 0.80 | 0.66 | 0.78 | 0.84 |
| Top-10 overlap | ≥ 70% | 58% | 69% | 76% |
| Kendall's τ (in-session requests) | ≥ 0.65 | 0.41 | 0.63 | 0.68 |
| Score PSI (worst slice) | ≤ 0.10 | 0.24 | 0.11 | 0.07 |
| p99 latency | ≤ 120 ms | 96 ms | 101 ms | 98 ms |
| Error rate | ≤ 0.3% | 0.9% | 0.2% | 0.1% |

The week-1 column is the point: shadow *always* starts red. A team that expects green on day one either has trivially easy parity gates or hasn't instrumented enough to see the divergence it has.

**Phase 3 — Canary on slice 1 (4 weeks).** Ramp 1% → 5% → 25% → 100% of NZ+IE with automated rollback: "add-to-cart rate from recs modules < −2% vs. concurrent control over any 6h window, OR p99 > 150ms for 30 min, OR error rate > 1% → auto-route to batch + page." **The incident:** at the 25% step, day 3, 02:40 local, a feature-store cache-node failure caused feature-fetch timeouts; the serving path fell back to default feature values, recs quality visibly degraded (generic popular items), and add-to-cart from recs dropped 9% over the next 6 hours. The rollback trigger fired at 08:55 — automatically, no human decision — routed the slice to batch, paged the team, and the on-call's job was root cause, not a 4 a.m. judgment call. Fixes: feature-fetch fallback changed from "defaults" to "yesterday's batch scores" (degrade to stale, not to generic — the degradation-ladder pattern from Module 10), plus a cache-health precondition added to the router. Ramp restarted a week later; clean. The incident *increased* leadership confidence — the safety net demonstrably works — which is worth saying out loud in the monthly note rather than burying.

**Phase 4 — Cutover waves (5 months).** Slices 2–4 repeat shadow-then-canary with shortened shadows (2 weeks each, gates unchanged). Slice 4 (homepage US/EU) runs a proper A/B alongside the parity canary: real-time recs show **+2.6% add-to-cart, +1.7% recs-attributed conversion [+0.9, +2.5]** — the freshness prize, measured, pre-registered. Email (slice 5) surfaces the long-tail grind: the digest pipeline needs top-500 per user in bulk, which the real-time path can't serve economically, so the team builds a daily bulk-scoring job *on the new stack* (same models, batch harness) — the correct accommodation, 6 weeks nobody planned. Mobile bulk endpoint follows the same pattern. Slices 5–6 plus stragglers consume months 8–12: the last 5% of traffic takes ~40% of the calendar, on plan only because the plan budgeted for it.

**Phase 5 — Deprecation (2 months).** Write-freeze on the monolith at month 11 (announced at month 3; leadership holds it when one internal team asks for an extension — they get two weeks, in writing, not a quarter). Read-alerts identify two stragglers the inventory missed (a quarterly finance report; a batch export a partner team had cron'd). Both migrated in 3 weeks. Old tables 4xx behind a grace flag for 2 weeks, then dropped; Spark cluster returned; code archived with the parity report and the dependency graph as the tombstone README. Final accounting: 13.5 months (12 planned), $2.4M build, $85k/mo × 9 months dual-running; monolith bill now $0; conversion lift alone pays back in ~14 months, better than the plan because the lift came in at the high end. The team demo at all-hands ends with the cluster-teardown screenshot, and the two engineers who closed the tail present it.

**What made this a principal-level project:** the pre-registered statistical gates (nobody argued about divergence after the fact), the explicit replicate-vs-fix rulings in writing, the automated rollback that turned an incident into a confidence builder, the ads-team break negotiated at VP level, the deprecation date leadership actually held, and the bankruptcy conditions that never fired but existed. None of these is code.

## Exercise

**Task.** Write the migration plan for the following system. "Lumen," a B2B SaaS company (4k enterprise customers, ~200 eng), scores **lead quality** for customers' sales teams with a 5-year-old system: a per-customer logistic-regression fleet (one model per customer, ~3 800 models) trained weekly on a cron box, scores written directly into the production Postgres of the CRM product, served by the CRM reading the column. The ML platform team has a modern stack: centralized training orchestration, a feature store, a scoring API, and a single multi-tenant gradient-boosted model that beats the per-customer fleet offline by +4.1 AUC points on 70% of customers (and *loses* on 11%, mostly the largest accounts with the most historical data). Known consumers: the CRM UI, a Zapier integration exposing the score field, customer-built reports on the Postgres column (unknown count), and the customer-success team's churn model, which uses lead-score volatility as a feature. Contractually, 40 enterprise customers have "model behavior change" notification clauses (30 days notice).

Deliverable: a migration plan document (2–4 pages) containing: (1) Phase 0 inventory approach including how you'll find customer-built consumers you cannot query-log; (2) slice sequencing with rationale — including what you do about the 11% of customers where the new model is *worse*; (3) shadow design with pre-registered parity gates (name the metrics and thresholds, per-slice); (4) canary design with automatic rollback triggers; (5) the deprecation plan for the Postgres column and the cron fleet, including the contractual-notice choreography; (6) economics — build cost, dual-running, payback, and your bankruptcy conditions; (7) the org mechanics — comms cadence, forcing functions, and what you need leadership to commit to in writing before you start.

**You're done when:**
- Every parity gate has a number, a slice, and a pre-registration rationale; none is decided "during shadow."
- The 11%-worse-off customers have an explicit disposition (hint: "migrate everyone anyway" and "keep the fleet forever" are both wrong; consider a per-customer champion gate, and price its permanent cost).
- The plan names at least one consumer class you will *deliberately break*, with the notice mechanism.
- The economics section states the dual-running monthly cost and the calendar month at which bankruptcy conditions trigger a go/no-go review.
- A reader can trace any single customer from "on the fleet" to "fleet deleted" through named phases with entry/exit criteria.

**Self-check questions:**
1. Why is "the new model wins offline by +4.1 AUC" insufficient evidence to begin canary, even with shadow parity gates green? (Two distinct reasons — one statistical, one from the consumer inventory.)
2. The churn model consumes lead-score *volatility*. The new model's scores are smoother. Is this a parity failure, an improvement, or a break — and who decides?
3. Scores are written into the product's Postgres. What does the strangler-fig "route by traffic slice" pattern mean when the interface is a database column, and what intermediate artifact do you introduce?
4. What rollback trigger protects a customer whose sales team sorts their entire pipeline by this score, and at what granularity must rollback operate?
5. If month 9 arrives with 60% of customers migrated and the two largest accounts refusing, which of your pre-written conditions fires, and what are the three options on the table?
