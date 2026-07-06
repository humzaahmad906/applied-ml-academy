# Module 09 — Build-vs-Buy & Unit Economics

## Why this module matters

No single decision a principal makes more often, or with more compounding consequence, than build-vs-buy: the feature store, the experiment platform, the labeling pipeline, the vector database, the LLM itself. Each call moves seven figures over a three-year horizon, and — worse than the money — a wrong call installs either a vendor you cannot leave or an internal system you must staff forever. Seniors argue these decisions on capability ("the vendor doesn't support X"); principals argue them on total cost of ownership, risk, and optionality, in language a CFO can act on. This module gives you the full framework, the LLM-specific crossover math, a worked TCO template, the negotiation leverage that follows from doing the analysis, and the decision-review hygiene that keeps a good 2026 decision from silently becoming a bad 2028 one.

## 1. The shape of the decision

Three framing corrections before any spreadsheet:

**It is a portfolio decision, not a point decision.** Every "build" consumes scarce senior-engineer attention that has an opportunity cost measured in *not-shipped product*. The question is never "could we build a better feature store than the vendor's?" (usually yes) but "is a feature store where our 15 best infra engineers create the most differentiated value?" (almost never). The strategic filter, applied before any cost math: **build what differentiates you, buy what doesn't.** Your ranking model is a moat; your feature store is plumbing. Companies that invert this — buying the differentiator, building the plumbing — end up with a commodity product and a bespoke maintenance burden.

**"Build" is never one-time; "buy" is never turnkey.** The two hidden-cost structures are mirror images (section 3), and honest analysis prices both.

**The decision has an expiry date.** Vendor pricing, open-source maturity, your volume, and model capabilities all move fast enough that any build-vs-buy answer is a snapshot. The decision memo therefore ships with revisit triggers (section 8) — this is the cheapest clause you will ever write.

## 2. The framework: five axes

Score every serious build-vs-buy decision on all five. The spreadsheet (axis 2) gets the attention; the other four decide the close calls and cause the disasters when skipped.

**Axis 1 — Capability fit.** Does the vendor product actually do the job, on *your* workload, verified by a time-boxed proof-of-concept on your data — not the demo dataset? Score the gap in three buckets: gaps the vendor roadmap closes in ≤2 quarters (acceptable with a contractual commitment), gaps you can shim with glue code (price the glue — section 3), and structural gaps (per-tenant models, point-in-time semantics your scale breaks, an SLA they won't sign). One structural gap outweighs any price advantage.

**Axis 2 — 3-year TCO.** Three years, not one: year-1 comparisons flatter vendors (build costs front-load) and flatter builds equally (maintenance hasn't hit yet). Four line-item families on each side: license/usage fees; infrastructure (the vendor's compute is *in* their price; your build's is not); **headcount, fully loaded** — salary × 1.25–1.4 for benefits/tax/equity/overhead, $350–450k per senior infra engineer in a US-hub market as the 2026 planning number; and **opportunity cost** — what the build headcount would otherwise produce. Opportunity cost is the line CFOs respect most and engineers omit most; even a conservative proxy (the fully-loaded cost again, doubled for high-leverage teams) beats omitting it.

**Axis 3 — Vendor risk.** Three sub-risks. *Lock-in:* what does exit cost in engineer-months? Data egress is the visible cost; the invisible one is workflow entanglement — 200 pipelines written against the vendor's SDK is a migration project (Module 08) priced in the millions. Prefer vendors wrapping open standards/formats; wrap vendor SDKs behind a thin internal interface from day one (~5% integration overhead, buys an order of magnitude on exit cost). *Pricing power:* per-seat and per-call pricing at renewal, after lock-in, moves 20–50% upward if you have no alternative; your negotiating position at renewal is set by decisions you make at signing (section 6). *Viability:* an ML-infra vendor with 18 months of runway is a risk you price — ask about funding, customer count, and insist on source-code escrow or an open-core license for anything load-bearing.

**Axis 4 — Optionality value.** Building creates option value (you can extend it in any direction) and destroys it (the team is committed for years). Buying creates option value (you can leave — if you kept exit cheap) and destroys it (roadmap capture). The honest question: which decision keeps more doors open *given what you know you don't know*? Early-stage products with unclear requirements should bias buy — you don't yet know what to build; at-scale products with stable, unusual requirements bias build.

**Axis 5 — Data gravity and residency.** Where the data must live sometimes decides the whole question before economics start. If training data cannot leave your VPC/region (regulated industries, EU residency, defense), the vendor list shrinks to those offering in-VPC deployment — often at 2–3× the SaaS price, which changes the TCO. Conversely, if the vendor's platform is where your data already lives (warehouse-native vendors), integration cost collapses and buy gets a structural discount. Check this axis first; it is binary more often than the others.

## 3. Hidden costs, both directions

**Hidden costs of build:**
- **The forever team.** The build estimate ("3 engineers, 2 quarters") is the *down payment*. A production internal platform needs 2–4 engineers permanently: upgrades, security patches, on-call, user support, and feature requests from every internal team that adopts it. Over 3 years the maintenance tail is typically 1.5–2.5× the initial build. Any build proposal without a permanent staffing line is fiction.
- **The roadmap tax.** Vendors ship features from the field's collective demand; your internal tool ships what your 3-engineer team gets to. Three years in, the internal feature store lacks the five capabilities the vendor added from other customers' needs — and your product teams pay that gap in velocity, invisibly, forever.
- **Key-person risk and the hiring premium.** Internal platforms concentrate knowledge in 1–2 heads; backfilling a departed platform author costs 6–9 months of ramp.

**Hidden costs of buy:**
- **Integration glue.** The vendor does 80% of the job; the 20% adapter layer — authentication plumbing, schema mapping, the sync job between the vendor and your warehouse, the wrapper SDK, monitoring — routinely runs **30–50% of what the build would have cost**, and it is *your* code, staffed by *your* on-call. Price it explicitly (a realistic PoC surfaces it).
- **Pricing that scales badly.** Per-seat pricing punishes growth; per-call/per-row pricing punishes success. Model the vendor bill at 1×, 3×, and 10× current volume before signing — many vendor deals are cheap at today's volume and catastrophic at the volume your own success projections claim. Negotiate volume-tier caps *now*, while you still have alternatives.
- **Roadmap capture and forced migrations.** Vendors deprecate APIs, sunset products, get acquired. Each event is a forced mini-migration on their schedule, not yours.

## 4. The LLM-specific decision

The highest-frequency 2026 instance of build-vs-buy is *API vs. self-hosted model*, and its cousin *prompt vs. RAG vs. fine-tune*. Both are economics questions wearing fashion clothing.

### API vs. self-host crossover

Both sides reduce to $/1M tokens; the ML System Design course derives the mechanics — here is the decision-grade version:

```text
API side:
  $/req = in_tok × in_rate + out_tok × out_rate
  (output rates run 3–5× input on all major providers; decompose or be 2–3× wrong)
  effective_$/req = $/req × (1 − cache_hit × prefix_share)   ← prompt caching

Self-host side:
  $/1M tok = GPU_$/hr × 1e6 / (tok_per_s × 3600 × utilization)
  all-in    = raw_compute × (1.3–1.6)      ← serving eng, eval infra, upgrades, idle
              + fine-tune/distill program cost amortized over volume

Planning numbers (state your own):
  H100 ≈ $2–4/hr reserved/spot; tuned vLLM, 7–8B FP8 ≈ 3 000–8 000 decode tok/s/GPU
  → raw compute ≈ $0.10–0.40 / 1M output tokens at ≥60% utilization
```

A worked crossover, end to end. A document-extraction path: 25k requests/day, 2 500 input + 300 output tokens/request, frontier API at $2.50/$10 per 1M in/out, 70% cache hit on a 1 500-token stable prefix:

```text
API SIDE
  raw $/req      = 2 500×$2.50/1M + 300×$10/1M          = $0.00925
  cached $/req   = $0.00925 − 0.70 × 1 500×$2.50/1M     ≈ $0.00663
  monthly bill   = $0.00663 × 25k/day × 30              ≈ $5.0k/mo

SELF-HOST SIDE (fine-tuned 8B, H100 @ $3.00/hr, 5 000 decode tok/s/GPU)
  headline $/1M  = $3.00 × 1e6 / (5 000 × 3 600 × 0.8)  ≈ $0.21/1M  ← the
                   number in every self-host pitch deck
  fleet reality  : 25k req/day × 300 out-tok = 7.5M tok/day
                   average decode load = 7.5M / 86 400   ≈ 87 tok/s
                   peak (2.5× diurnal factor)            ≈ 217 tok/s
                   minimum HA fleet = 2 GPUs (zones)     = 10 000 tok/s capacity
                   → real utilization ≈ 1–2%, not 80%
  real $/1M      = $3.00×1e6 / (5 000×3 600×0.015) ≈ $11/1M — 50× the headline
  monthly cost   = 2 GPUs × $3.00 × 730h ≈ $4.4k/mo compute
                   + 0.5–1.0 FTE serving/eval ownership  ≈ $17–34k/mo
  → self-host TOTAL ≈ $21–38k/mo  vs. API $5k/mo: API wins by 4–8×
CROSSOVER: at ~20× the volume (500k req/day), the same fleet math gives
  ~35% utilization on 2 GPUs, API bill ≈ $100k/mo vs. self-host ≈ $30–45k/mo
  all-in → self-host wins. The crossover is a VOLUME, and you should know yours.
```

The lesson generalizes: below a few hundred sustained tokens/second, the *headcount* line dominates the GPU line, and the API wins even when the per-token slide says otherwise. Run the fleet arithmetic — average load, peak factor, HA minimum, real utilization — before quoting any $/1M figure in a decision meeting; the gap between headline and real utilization is the single most common error in self-host proposals.

The two facts that decide most cases: **utilization sensitivity** — at 10% utilization your self-host cost is 8× the headline number, and bursty/diurnal traffic makes sustained 60%+ genuinely hard below ~5–10M requests/day on a narrow path; and **the capability question** — self-host math only applies where a fine-tuned 4–8B model matches the frontier API *on your narrow distribution*, which is true for high-volume extraction/classification/routing/drafting paths and false for open-ended reasoning. Hence the stable 2026 pattern: **API for product-market fit** (zero infra, latest capability, pay-as-you-go while volume and interaction data accumulate), then **distill + self-host the high-volume narrow paths** once (a) volume sustains utilization, (b) you have interaction data to fine-tune on, and (c) the eval infrastructure from Module 07 exists to prove parity. Keep the API for the long tail — the answer is a portfolio, not a side.

### Fine-tune vs. RAG vs. prompt — as economics

Strip the fashion; each option is a different cost curve:

- **Prompting** is all marginal cost: every request re-pays for the instructions and examples in input tokens. Cheapest to start, most expensive per request at volume, zero maintenance. Caching flattens the curve substantially.
- **RAG** trades a fixed infra cost (vector store, embedding pipeline, reranker — commonly 20–40% of total system cost at scale) for grounding on *changing* knowledge. It is the only correct answer when the knowledge updates faster than you could retrain, and it is auditable (you can show the source). Its marginal cost is retrieval + longer prompts.
- **Fine-tuning** is capex: a training program ($10k–200k for a small model done properly, including the eval work) that buys lower marginal cost (shorter prompts, smaller model) and *behavioral* reliability (format adherence, tone, task specialization). It cannot keep up with fast-changing facts, and it locks the behavior in until the next run.

| | Cost shape | Buys you | Fails when | Maintenance |
|---|---|---|---|---|
| Prompt | all marginal (per-request tokens) | speed to ship, easy iteration | volume makes instructions expensive | near zero |
| RAG | fixed infra + moderate marginal | grounding on changing knowledge, auditability | knowledge is stable (paying infra for nothing) | pipeline + index ops |
| Fine-tune | capex ($10k–200k/run) + low marginal | behavior, format, small-model economics | facts change faster than retrain cadence | eval + retrain program |

Decision rule: **changing knowledge → RAG; stable behavior at volume → fine-tune; low volume or still exploring → prompt.** Real systems compose all three (fine-tuned small model + RAG + a slim prompt); the composition is chosen by asking, line by line, "is this token spend buying knowledge, behavior, or instructions?" and moving each to its cheapest home.

## 5. The worked TCO template

The artifact that carries the decision. Spreadsheet-style; every number gets a stated source or assumption:

```text
BUILD-vs-BUY TCO — <decision> — 3-year horizon        owner / date / v
ASSUMPTIONS (each with source)
  volume now / yr3:        e.g. 20M pred/day → 60M    (product plan §3)
  eng fully-loaded:        $400k/yr senior infra      (finance planning rate)
  discount/growth applied: none (keep it simple, note it)

                                   BUILD              BUY
Year 0-1
  license / usage                  —                  $______  (quote, tier X)
  build / integration eng          ___ eng-mo × $33k  ___ eng-mo × $33k  ← glue!
  infra (compute+storage)          $______            included? $______
  one-time (PoC, security review)  $______            $______
Year 2
  maintenance / license            ___ FTE × $400k    $______ (renewal +__%?)
  infra                            $______            $______
Year 3                             (same lines)       (same lines)
  ------------------------------------------------------------------
  SUBTOTAL (cash)                  $______            $______
  Opportunity cost of build FTEs   $______            —
  Exit cost (amortized risk)       —                  $______ (eng-mo × prob)
  ------------------------------------------------------------------
  3-YEAR TCO                       $______            $______
  $/unit at yr-3 volume            $______/1M pred    $______/1M pred

SENSITIVITY (recompute TCO under each; a decision that flips is a fragile one)
  volume ±50%  |  vendor renewal +30%  |  build slips 2 quarters  |
  maintenance needs 3 FTE not 2  |  key vendor feature slips a year

NON-COST AXES (one line each): capability gaps | lock-in / exit | viability |
  optionality | data residency
RECOMMENDATION + REVISIT TRIGGERS (see §8)
```

Discipline notes: the glue-code line under BUY is mandatory (its systematic omission is why buy decisions look 40% cheaper than they turn out); the sensitivity block is mandatory (a recommendation that survives all five perturbations is robust — one that flips on "volume +50%" is really a bet on the volume forecast, and the memo must say so).

## 6. Negotiation leverage

The TCO analysis is not just decision support — it is negotiating capital. Vendor list prices for ML infrastructure are opening bids with 20–40% of headroom, and the discount you get is a function of the alternatives you can credibly hold:

- **Benchmark 2–3 vendors, always,** even when you're 90% sure of the winner. Parallel PoCs cost 2–3 engineer-weeks and reliably pay five-to-six figures at signing. Vendors price single-source deals accordingly.
- **A credible build alternative moves price 20–40%.** "We've costed the internal build at $1.9M over 3 years" — said with a real spreadsheet behind it — changes the conversation. The build analysis you did to make the decision honestly is the same artifact that makes the buy cheaper. (Credibility requires the vendor to believe you *could*: a platform team with shipped systems makes the threat real; a two-person team does not.)
- **Negotiate the year-3 terms at year 0:** renewal caps (e.g., ≤7%/yr), volume-tier pricing at 3× and 10× current usage, data-egress terms and exit assistance, and SLA credits with teeth. Everything is negotiable before signature and almost nothing after entanglement.
- **Time your signature.** Vendor quarter-ends and fiscal year-ends are real; a deal that stalls in week 6 closes in week 13 with an extra 10% off. Undignified, effective.

## 7. The CFO memo

The decision is not made in the spreadsheet; it is made when a budget owner approves it. Principals write the memo in the reader's units — **dollars per unit of business, time-to-market, and risk** — not model accuracy or architectural elegance. One page:

```text
DECISION MEMO — <title>                          ask: approve option __ / $__
RECOMMENDATION  One sentence. ("Buy VendorX at $310k/yr with a 3-yr renewal
                cap; revisit if volume triples.")
THE NUMBERS     3-yr TCO both paths; $/unit both paths at yr-3 volume;
                cash-flow shape (build is back-loaded pain, buy is flat).
TIME-TO-MARKET  Buy: live in Q_. Build: live in Q_. What the delta costs
                in delayed product terms (the CFO's favorite line).
RISK            Top risk of the recommendation + its mitigation; top risk
                of the alternative (why we're not doing it).
WHAT WE'RE NOT DECIDING   Scope fence — adjacent decisions left open.
REVISIT TRIGGERS          The conditions that reopen this (see §8).
```

Anti-patterns that kill memos: leading with architecture diagrams; ten options (bring two, maximally three); precision theater ($1,847,332 — use $1.8M; false precision reads as naivety); and hiding the losing option's genuine advantages (a memo that strawmans the alternative gets re-litigated by the first exec who notices).

## 8. Decision review hygiene

A build-vs-buy call is made under assumptions with a half-life of about a year. Mature orgs record and revisit; immature orgs re-litigate ad hoc (wasteful) or never (dangerous).

- **Record the assumptions with the decision** — a short decision record: options, numbers, the 3–5 load-bearing assumptions. Ten minutes of writing; its absence costs the org the same debate annually, from memory, with turnover:

```text
DECISION RECORD — <decision>            date / decider / status: ACTIVE
CHOSE      option __ over __ (3-yr TCO $__ vs $__; memo linked)
BECAUSE    the 2–3 decisive factors, one line each
ASSUMED    1. volume grows ≤2×/yr          3. no in-VPC mandate lands
           2. vendor renewal ≤7% (capped)  4. OSS alternative stays immature
REVISIT IF volume > __ | price change > __% | vendor acquired/roadmap slip
           > 2Q | residency mandate | OSS alt reaches capability __
NEXT       triggers wired to: <dashboard/alert>; record reviewed on any fire
```

- **Set explicit revisit triggers, not calendar reviews.** "Revisit if: volume exceeds X, vendor price changes >Y%, the vendor is acquired or misses the committed roadmap item, open-source alternative reaches capability Z, or a residency requirement lands." Triggers wired to a metric someone already watches actually fire; annual calendar reviews get skipped.
- **Reversal is not failure.** The 2026 decision was right on 2026 facts; the 2028 reversal is right on 2028 facts. A principal who makes reversal cheap (exit terms, wrapper SDKs, decision records) and un-shameful (the trigger fired; we planned for this) has built an org that updates — rarer and more valuable than an org that decides correctly the first time.

## Worked example

**Setting.** "Meridian," a fintech, ~300 engineers, ~40 in ML across 5 teams (fraud, credit risk, collections, marketing, support LLM). Pain: every team hand-rolls features; two production incidents last year traced to training-serving skew; fraud and credit compute near-identical features twice. The platform lead proposes building a feature store; a vendor ("FeatherStore," representative of the managed category) quotes $280k/yr list. You own the decision. Planning assumptions, stated: senior infra engineer fully loaded **$420k/yr** ($300k comp × 1.4); volume 15M online lookups/day → 45M by year 3; residency: SOC 2 required, in-VPC preferred but not mandated by regulators today.

**Capability fit (PoC, 3 weeks, two engineers).** FeatherStore handles Meridian's point-in-time joins and online serving at p99 = 9ms on replayed production traffic. Two gaps: no native support for fraud's 200ms-freshness streaming counters (vendor roadmap says two quarters; fraud says the current homegrown Redis path can persist for exactly that path — a shim, priced below), and the credit team's model-governance requirement (feature lineage reports for regulators) needs custom export glue. No structural gaps. A second vendor PoC (warehouse-native) fails the online-latency requirement at p99 = 140ms — but its existence is now negotiating capital.

**3-year TCO — BUILD.** Initial build: 4 engineers × 5 quarters (offline store + online store + point-in-time engine + SDK + monitoring; the estimate already includes the historical 1.5× slip factor) = **$2.10M**. Permanent maintenance from year 2: 2.5 FTE = **$1.05M/yr × 2 = $2.10M**. Infra: $18k/mo growing to $30k/mo ≈ **$0.86M**. Cash subtotal ≈ **$5.06M**. Opportunity cost: those 4 engineers are the platform team's strongest; the displaced roadmap item is the real-time serving consolidation (Module 05) with an estimated seven-figure annual infra saving — booked conservatively at **$1.0M**. **Build, 3-yr: ≈ $6.1M** ($5.1M cash).

**3-year TCO — BUY.** License: negotiated (two-vendor benchmark + the credible build spreadsheet on the table) from $280k to **$225k/yr with a 7% renewal cap and 3×-volume tier pricing locked** — the build analysis paid for itself before the decision was even made. 3-yr license ≈ **$0.72M**. Integration glue: SDK wrapper (mandatory, for exit cheapness), warehouse sync, lineage-export glue for credit, the fraud Redis shim, migration of the two largest existing pipelines = 2 engineers × 3 quarters = **$0.63M** — right in the predicted 30–50%-of-build band for year-0 build cost. Ongoing ownership: 1 FTE (vendor management, wrapper, upgrades, internal support) = **$0.42M/yr × 2.5 = $1.05M**. Infra riders (private-link, egress) ≈ **$0.12M**. Exit-cost risk: estimated 8 engineer-months if exercised, × 25% probability ≈ **$0.07M** amortized. **Buy, 3-yr: ≈ $2.6M.**

The completed template, condensed:

```text
FEATURE STORE TCO — Meridian — 3yr            v3, final     owner: <you>
ASSUMPTIONS  15M→45M lookups/day | eng $420k loaded | SOC 2 req'd, in-VPC pref
                                        BUILD           BUY (FeatherStore)
Yr 0-1   license                        —               $225k
         build / integration eng        $1 680k (4 FTE) $630k glue (2×3Q)
         ongoing ownership              —               $210k (0.5 FTE, mid-yr on)
         infra                          $216k           $40k riders
Yr 2     maintenance / license          $1 470k (2.5 FTE + $420k build tail)
                                                        $241k + $420k (1 FTE)
         infra                          $300k           $40k
Yr 3     maintenance / license          $1 050k         $258k + $420k
         infra                          $360k           $40k
         --------------------------------------------------------------
         CASH SUBTOTAL                  $5 076k         $2 524k
         opportunity cost (build FTEs)  $1 000k         —
         exit-risk (8 eng-mo × 25%)     —               $70k
         --------------------------------------------------------------
         3-YEAR TCO                     ≈ $6.1M         ≈ $2.6M   (2.4×)
         $/1k lookups (3-yr TCO ÷ ~33B) $0.19           $0.08
SENSITIVITY  no perturbation tested (±50% vol, +30% renewal, 2Q build slip,
             +1 maint FTE, vendor death) flips the decision
```

**Sensitivity.** Volume +50%: buy grows ~$60k/yr under the locked tiers — decision holds. Vendor renewal +30%: capped at 7% by contract — held, *because it was negotiated at signing*. Build slips 2 quarters (historical base rate: likely): build → $6.9M — gap widens. Maintenance needs 3.5 FTE not 2.5 (base rate: also likely): build → $6.9M. Vendor dies: exit to the warehouse-native runner-up or to an open-source deployment behind the wrapper SDK ≈ 8 engineer-months — survivable by construction. **No perturbation flips the decision; the recommendation is robust, and the memo says so.**

**The memo (as sent, condensed).** *Recommendation:* Buy FeatherStore at $225k/yr (3-yr, 7% renewal cap, 3× volume tiers locked). *Numbers:* 3-yr TCO $2.6M vs. $6.1M build; over the horizon's ~33B lookups, $0.08 vs. $0.19 per 1k. *Time-to-market:* first production use case in ~1 quarter vs. ~5; the skew-incident class that cost us two production incidents last year gets addressed 15 months sooner. *Risk:* vendor dependency — mitigated by wrapper SDK, exit terms, and a tested runner-up; the build path's top risk (permanent 2.5-FTE commitment against our strongest platform engineers, displacing the serving consolidation) is why we're not doing it. *Not deciding:* the streaming-counter path stays on the fraud team's Redis pending the vendor's Q3 delivery; revisit then. *Revisit triggers:* volume >100M lookups/day; renewal proposal >7%; vendor acquired or Q3 streaming feature slips >2 quarters; a regulator mandates in-VPC. Approved in one meeting; the CFO's only question was about the renewal cap — which is the question the memo was built to survive.

**Eighteen months later, a trigger fires.** FeatherStore is acquired by a major cloud vendor; six months post-acquisition, the renewal proposal arrives at +40% with the volume tiers "restructured" — contractually blocked for one more cycle by the 7% cap, but the writing is on the wall for year 4. The decision record is pulled up (ten minutes, not a re-litigated quarter): assumptions checked — volume is at 70M/day and the open-source alternative flagged in the record has matured materially. The wrapper SDK means the exit estimate is re-validated at ~9 engineer-months. Outcome: Meridian runs the renewal negotiation holding a now-*tested* exit plan, lands +9% (cap + goodwill), and green-lights a two-quarter migration evaluation for year 4 — executed, if it proceeds, with the Module 08 playbook. The 2026 buy decision and the potential 2028 exit are *both* right: that is what the triggers were for, and why reversal was cheap.

## Exercise

**Task.** You are the principal at "Atlas Logistics" (900 employees, 140 engineers, 18 in ML). The support organization (120 agents, 1.4M tickets/yr, growing 25%/yr) runs an LLM ticket-drafting assistant on a frontier API: currently 60k requests/day at ~3 000 input + 400 output tokens/request, on a model priced at $2.50/1M input, $10/1M output, with prompt caching available (the 1 800-token system-prompt-plus-examples prefix is stable; observed cache hit rate 65%). The CTO asks: should Atlas keep the API, or fine-tune and self-host an 8B model for the drafting path? Constraints and facts to use: an internal PoC shows a fine-tuned 8B matches the frontier model on draft-acceptance rate for the top 6 ticket categories (74% of volume) but clearly loses on the long tail; Atlas has one strong infra engineer with GPU-serving experience and can hire (fully loaded $400k); reserved H100s available at $2.60/hr; assume a tuned 8B serves ~5 000 decode tok/s/GPU; the CISO would *prefer* ticket text stay in-VPC but has signed the current API DPA.

Produce: (1) a **full build-vs-buy analysis** using the section 5 TCO template — API-with-caching vs. self-host-the-top-6-categories-keep-API-for-the-tail (analyze the *portfolio* option, not a false binary), with 3-year math at 25%/yr volume growth, an explicit utilization calculation for the GPU fleet you'd need (peak-vs-average matters: assume peak = 2.5× average), fine-tuning program costs, and the ongoing eval/serving headcount; (2) a **sensitivity block** covering at minimum: growth 0% and 50%, API price cut of 40% (frontier prices have historically fallen — treat this as likely), acceptance-rate parity degrading on tail-shift, and the serving engineer quitting; (3) a **one-page CFO memo** per section 7, with revisit triggers.

**You're done when:**
- Every dollar figure traces to a stated assumption, and the API-side math correctly applies the cache formula (effective input cost, not list price).
- Your GPU fleet size comes from an arithmetic chain the reader can check: req/day → tok/s average → × peak factor → ÷ per-GPU throughput → utilization %, and the utilization number visibly drives the $/1M-token result.
- The portfolio option (self-host head, API tail) is costed as its own column, not described in prose.
- The memo fits one page, leads with the recommendation, and its risk section names the *strongest* argument against your recommendation honestly.
- At least three revisit triggers are wired to observable quantities.

**Self-check questions:**
1. At Atlas's current volume, what utilization does a minimum viable HA deployment (2 GPUs across zones) actually achieve, and what does that do to the $/1M-token figure versus the headline "5 000 tok/s at 80%" math?
2. A 40% API price cut arrives twelve months in. Which side of your analysis does it hit, does it flip your recommendation, and how did your revisit triggers anticipate it?
3. The 8B matches the frontier model *today*, on today's ticket mix. What Module 07 infrastructure must exist before — and after — cutover for that claim to stay trustworthy, and where did you book its cost?
4. The CISO's in-VPC preference is not a mandate. How does your framework price a preference (as opposed to a requirement), and what would convert it into a decision-flipping constraint?
5. If Atlas had 10× the ticket volume, which specific lines in your TCO change, which utilization threshold gets crossed, and at approximately what request volume is the true crossover for the drafting path?
