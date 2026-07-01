# 08 — The ML / AI Solutions Architect Track

The IC architect track covers the staff/principal/platform architect path — operating inside a single company and owning the long-term technical health of one platform. This chapter covers a different role with overlapping skills: **Solutions Architect (SA)**.

The distinction matters because the work, comp model, hiring pipeline, and day-to-day rhythm are different. Some engineers are happier in SA than IC architect; some the reverse. Knowing what you're optimizing for saves years.

**Who this section is for:** Senior ML engineers (3–7+ years) considering a Solutions Architect role — at a cloud vendor (AWS, GCP, Azure, Databricks, Snowflake, NVIDIA), at an ML platform vendor (Tecton, Weights & Biases, Anyscale, Hugging Face), at a systems integrator/consultancy (Accenture, Deloitte, Slalom, Capgemini), or as an internal SA at a large F50 (the "architecture office" function).

---

## Phase 1 — What an ML Solutions Architect Actually Does

The honest version.

### Two Flavors of the Role

There are two distinct populations both called "Solutions Architect":

| Flavor | Where | Primary stakeholders | What you produce |
|---|---|---|---|
| **External / vendor SA** | AWS, GCP, Azure, Databricks, NVIDIA, Snowflake, Tecton, Hugging Face, Anyscale | Customers (their engineers + their execs) | Reference architectures, POCs, workshops, design reviews; you help close deals |
| **Internal / enterprise SA** | F50 architecture office; consultancies (Accenture, Deloitte, ThoughtWorks) | Internal business units + engineering teams | Design reviews, standards, vendor selection, capability roadmaps |

Both share core skills. Different comp model, different rhythm, different exit options.

### Core Output: Customer Outcomes, Not Code

Where an IC architect optimizes for the long-term health of one platform, an SA optimizes for **customer adoption and success**. Daily output:

- Reference architectures matched to specific customer contexts
- Workshops and enablement (you teach a lot)
- Proof-of-concept code (throwaway, but real)
- Design review feedback for customer teams
- Pre-sales support (vendor SAs): qualifying deals, demoing, helping close
- Post-sales support (vendor SAs): unblocking customers, deepening usage
- Written content: blog posts, conference talks, white papers
- Vertical expertise: financial services, healthcare, retail, manufacturing, public sector — deep enough to talk to industry-specific buyers

You touch code, but not as the main artifact. The main artifact is **the customer's correct decision** — they picked the right architecture, adopted the right platform, avoided the wrong path.

### Vendor SA Specifics

At a cloud / platform vendor:

- **Quota and territory.** You're assigned customers (accounts) or a vertical (financial services, healthcare). Performance partly measured by their adoption.
- **Comp model.** Base + variable. Variable tied to account consumption / new logos / strategic deal wins. For Senior SAs at major clouds, total comp is in the same range as Senior SWE total comp, sometimes higher; variable is real money (20–40% of total).
- **Tied to sales motion.** You partner with an Account Executive (AE). AE owns the deal; SA owns technical credibility. Bad AEs make SA life miserable; great AEs make SAs look great.
- **Travel.** Pre-pandemic: 50–80% travel. Post-pandemic: 20–50% typical. Still more than IC roles.
- **Quarterly intensity.** End of quarter = sprint to close deals. Plan vacations around it.

### Internal SA Specifics

In a F50 architecture office:

- **Cross-business-unit scope.** You set standards across many internal teams. No external customers.
- **No variable comp tied to deals.** Salary + bonus like other senior engineering roles.
- **Less travel.** More meetings, fewer flights.
- **Power comes from credibility, not authority.** You usually can't *force* teams to follow your guidance. You influence.

### Consultancy SA Specifics

At Accenture / Deloitte / ThoughtWorks / Slalom / Capgemini:

- **Project-billed.** You're sold to clients at $X/hour. Utilization (billable hours) is a core metric.
- **Diverse engagements.** One quarter you're at a bank, next at a retailer, next a hospital.
- **Vertical specialization** still emerges; the best consultants build a "go-to" pattern.
- **Travel.** Historically heavy; varies post-2020.
- **Promotion ladder.** Senior Consultant → Manager → Senior Manager → Director → Partner. Different beast than engineering.

### What You Stop Doing

- Building a single system over years
- Owning a production runbook
- Long-term technical debt cleanup
- The deep "month-long debugging" engineer experience

### What You Start Doing

- Talking to 5–15 customers/teams per week (vendor SA at scale)
- Writing reference architectures for situations you're seeing repeatedly
- Conference talks, blog posts, webinars
- POCs you build in 2 weeks and abandon
- Knowing 5 vertical markets well enough to be credible in their language

### The SA's North Star

Different from the IC architect's "long-term platform health":

- **Vendor SA:** customer success with your product/platform. Adoption, growth, retention.
- **Internal SA:** capability uplift across the org. Teams ship faster, with fewer mistakes, on consistent foundations.
- **Consultancy SA:** repeat business + reference customers + a track record of delivery.

In all three, **trust is the currency**. Bad SAs over-sell, get caught, and lose the room forever. Strong SAs are deliberately honest about trade-offs, including when their product/platform is the wrong answer.

---

## Phase 2 — The SA Mindset

Specific shifts.

### Sell the Trade-off, Not the Tool

A vendor SA who can only say "our product is great" loses fast. Senior customers see through it. The SA who says:

> "For this workload, our managed serving is the right answer because of A and B. But if your team grows past 50 engineers and your inference cost goes over $200K/month, you'd save money by self-hosting on K8s. We can revisit then."

…gets the deal *and* keeps the customer's trust for the next deal. Honesty about edges is the SA's most valuable trait.

### Discovery Over Pitching

Junior SAs walk into a meeting and pitch. Senior SAs ask:

- "What problem are you trying to solve?"
- "What have you tried?"
- "What's worked? What hasn't?"
- "Who's the user? What does success look like for them?"
- "What's the timeline? What's the budget?"
- "Who else is involved in this decision?"

By the end of discovery you know if your platform is the right fit. If it is, you propose. If it isn't, you say so. Saying so is what builds the trust that gets you the next opportunity.

### Customer Context First

Each customer has a unique combination of:

- Current stack (legacy systems, recent investments)
- Team skills (Python-heavy vs Java vs no engineers)
- Risk tolerance (regulated bank vs SaaS startup)
- Time pressure (next quarter vs next year)
- Budget (enterprise license vs $0)
- Politics (the VP loves vendor X; the CTO hates vendor X)

The same architecture problem has different correct answers for different customers. Senior SAs internalize this; junior SAs apply the same playbook everywhere and fail.

### The "What Would I Actually Do?" Test

A useful self-check before recommending something: would you yourself build this way if it were your problem and your bill? If not, don't recommend it. Recommending against your own conviction destroys trust quickly.

### Three Time Horizons (Different from IC Architect)

- **0–3 months:** the current customer engagement. POC, design review, workshop.
- **3–12 months:** repeating patterns across customers. Write reference architecture; build internal enablement.
- **12+ months:** vertical / domain expertise. Deep knowledge of one or two industries that compounds over years.

The shape: short-term concrete, long-term pattern-recognition across customers.

---

## Phase 3 — Reference Architectures: The SA's Core Artifact

A reference architecture is a documented pattern for solving a class of problem. The SA's library of them is their leverage.

### What a Good Reference Architecture Includes

1. **Problem statement.** One sentence: "Customer X has problem Y, with these constraints."
2. **Context applicability.** When this pattern fits and when it doesn't.
3. **Architecture diagram.** C4 Container level usually.
4. **Component choices** with one sentence on each *why*.
5. **Variants.** "If your team is K8s-experienced, use this. If not, use this managed alternative."
6. **Cost model.** Rough order of magnitude, parameterized by volume.
7. **Failure modes.** What breaks first. How to detect.
8. **Operational notes.** SLOs, scaling, monitoring.
9. **Customer success stories** (anonymized) if available.

The discipline: write these *after* you've helped 3+ customers with the same shape of problem. Patterns harvested from real engagements beat patterns invented in a vacuum.

### A Library to Build

ML SAs typically maintain a personal collection (and a team-shared one) covering at minimum:

#### Classical ML serving

- Real-time prediction service for sub-100ms use cases
- Batch prediction for nightly scoring of millions of entities
- Streaming prediction (Kafka → Flink → online store → service)
- Multi-model serving (many small models, one shared infrastructure)
- Edge / on-device inference

#### LLM / GenAI patterns

- RAG on enterprise documents (the canonical 2026 ask)
- Multi-tenant LLM platform for an internal "AI gateway"
- LLM fine-tuning factory (org-wide pipeline for LoRA-SFT + DPO + serving)
- Agentic systems for internal automation
- LLM-powered customer support (chat + voice + escalation handoff)
- Document understanding / extraction
- Code generation / developer productivity
- Search relevance with embeddings

#### Data + feature patterns

- CDC from operational DB → lakehouse → features → models
- Feature store deployment (Feast, Tecton, or platform-native)
- Multi-region feature serving
- ML data quality framework

#### ML platform patterns

- Training cluster (GPU pool, scheduling, multi-tenancy)
- Model registry + promotion workflow
- Monitoring + drift detection stack
- ML CI/CD with model contracts
- GitOps for ML
- Federated / mesh ML platform

#### Vertical-specific

- Fraud detection (financial services)
- Risk modeling (insurance, lending)
- Recommendation (retail, media, marketplace)
- Demand forecasting (retail, logistics, manufacturing)
- Clinical decision support (healthcare; HIPAA-aware)
- Predictive maintenance (manufacturing, energy)
- Content moderation (social, marketplaces)
- Personalization (any consumer)

Aim for 20–30 reference architectures in your personal library. That's the SA equivalent of a senior engineer's design patterns.

### Reference Architectures Are *Living*

The mistake juniors make: write a reference architecture, treat it as truth, recommend it for years. Reality:

- The field changes every quarter (especially LLM-adjacent)
- Your platform's capabilities change every quarter
- Customer feedback reveals patterns that didn't work

Maintain. Refresh quarterly. Mark superseded versions.

### Where to See Good Examples

Public references worth studying:

- **AWS Well-Architected Framework — Machine Learning Lens**
- **GCP Architecture Center — Machine Learning**
- **Azure Architecture Center — AI + Machine Learning**
- **Databricks Solution Accelerators**
- **NVIDIA NIM and NeMo reference architectures**
- **The TWIML AI Solution Architecture series**

Read 20+ of these. Note what makes the good ones good: specific decisions, named trade-offs, concrete cost numbers, when-not-to-use.

---

## Phase 4 — The Discovery Conversation

The single most important conversation in the SA role. Done well, it makes everything downstream easier.

### The Standard Arc (90–120 minutes)

**Minutes 0–10 — Introductions and framing.**

- Who's in the room and their role
- What we're hoping to achieve in the hour
- The customer's hopes (set their expectations early)

**Minutes 10–40 — Current state.**

- What's the business goal driving this?
- What's the user-facing experience today?
- What's the current technical state? Walk me through.
- What's gone well? What hasn't?
- What's the team's skill profile?

**Minutes 40–80 — Constraints and desired state.**

- What's the timeline?
- Budget order of magnitude?
- Compliance constraints (data residency, audit, fairness, AI Act)?
- Existing vendor commitments?
- What does "great" look like?
- What does "good enough for this quarter" look like?

**Minutes 80–110 — Pattern matching and options.**

- "Based on what you've said, this pattern fits…"
- 2–3 candidate approaches with explicit trade-offs
- The next 30/60/90-day path

**Minutes 110–120 — Next steps.**

- Concrete actions and owners
- What I'll send (reference architecture link, sample code)
- When we'll reconvene

### What Strong Discovery Sounds Like

- "Help me understand…" (asking, not telling)
- "I want to make sure I'm pointing you to the right pattern. Can you tell me more about…"
- "What's the actual cost of getting this wrong?"
- "Who's the executive sponsor?" (you need to know early)
- "What happens if you do nothing?" (often more revealing than the customer expects)

### What Weak Discovery Sounds Like

- "Our product does X, Y, and Z" (pitching, not listening)
- "Most of our customers do…" (anchoring to a pattern that may not fit)
- Closed yes/no questions when open questions would reveal more
- Pretending to know the customer's industry better than the customer

### The "Customer's Customer" Question

A pattern that separates senior from junior SAs: always understand the customer's customer.

- "Your team will use this model. Who does your team's output serve?"
- "Your bank uses fraud detection. What does the actual cardholder experience look like?"

Knowing the customer's customer makes you a better advisor because you can spot when the proposed solution wins technically but loses for the end user.

### Active Listening

You'll learn this is genuinely a skill. Counterintuitive moves:

- **Silence after a question.** Give the customer 7–8 seconds. They often start with a polished answer; the *real* answer comes after.
- **Reflective summarization.** "So if I understand right, the issue is X driven by Y, and you've tried Z but it didn't work because…" Lets them correct your understanding.
- **The "tell me more" follow-up.** When they say something interesting, don't move on. Stay there.

---

## Phase 5 — POCs and the Demo Trap

### What a POC Should Do

A Proof of Concept is meant to **answer a specific question**, not to be a working product.

Good POC questions:

- "Can our platform handle 2000 RPS with sub-50ms P99 on the customer's actual data shape?"
- "Does our LLM produce acceptable answers on their corpus with their preferred prompt style?"
- "Can their data team build a pipeline using our SDK in a week?"

Bad POC scopes:

- "Build a complete fraud detection system" (too big; ships nothing in 4 weeks)
- "Show what's possible" (too vague; never ends)
- "Just kick the tires" (no exit criteria; drifts forever)

### The POC Contract

Before starting, write a one-pager:

- **Question being answered.** One sentence.
- **Success criteria.** Specific, measurable.
- **Scope.** What's in and explicitly out.
- **Timeline.** 2–4 weeks for most ML POCs. Anything longer is a project, not a POC.
- **Responsibilities.** What the SA provides; what the customer provides (data, access, time).
- **Decision.** "On Date X we'll review and decide go/no-go on Y."

Both sides sign. This kills the POC-that-never-ends pathology.

### POC Anti-Patterns

1. **The POC that becomes production.** Throwaway code becomes production system, lives forever, embarrasses everyone. Fix: write "DO NOT USE IN PRODUCTION" in the README and mean it.
2. **The POC with shifting goalposts.** "It works, but can you also show…" Discipline: original criteria, period.
3. **The POC the customer doesn't engage with.** SA does all the work alone. Worthless — adoption proof requires the customer's team to participate.
4. **The unfunded POC.** No clear path to a deal. POCs are expensive; gate them on qualified opportunity.

### Demo Skills

Live demos are an art. Patterns that work:

- **Pre-built happy path** that always works. Practice 5+ times.
- **One unscripted moment.** Type a custom query the customer suggests. Shows the product is real, not a screenshot.
- **Show the surprising thing.** Whatever your product does that *no one else's* does. Lead with that.
- **Stop after 15 minutes.** Demos longer than that lose the room. Invite questions; stop talking.
- **Have a backup.** If the live demo dies (network, login, dependency), have a 60-second recorded version on file. Switch fast.

### The Most Common Demo Failure

Going deep on features instead of outcomes. Customers don't care that you have 47 connectors; they care whether *their* connector works in *their* situation. Keep refocusing on the customer's specific job.

---

## Phase 6 — Workshops and Enablement

A core SA artifact: hands-on workshops that teach customers how to use the platform.

### Workshop Design

**Time budget:** 90 minutes is short; 4 hours is long; 1 day is rare and expensive.

**Pre-work:** never trust the customer to install software live. Send a setup guide a week ahead. Provide a hosted alternative for those who skip pre-work.

**Pace:** account for the slowest person. 30% slack. Easier to add bonus content than to skip planned content.

**Structure:** introduction (10 min) → demo of finished state (5 min) → guided build (60 min) → free exploration (15 min) → Q&A and what's next.

**Materials:** a Git repo, a hosted notebook (Colab / SageMaker Studio Lab), or a fully-managed sandbox (vendor SAs often have these). Ideally no local installs.

**Live error handling:** something always breaks. Have a backup: "if your setup isn't working, paste your output into chat and I'll help — meanwhile here's the answer."

### What Makes a Workshop Good

1. **Solves a real problem the audience cares about.** Generic "intro to our SDK" workshops are forgettable. "Fine-tune a Llama-3.1-8B to extract structured data from invoices" sticks.
2. **Ships an artifact.** The customer leaves with something working they can show to their boss.
3. **Time-boxed.** Stays within the announced budget.
4. **Reusable.** A good workshop runs many times with minimal updates. Bake in low-maintenance design.

### Enablement at Scale

Vendor SAs at scale teach 100+ customers per year. You can't repeat the same workshop 100 times. Patterns:

- **Train-the-trainer.** Teach the customer's lead engineer to run the workshop for their team.
- **On-demand video courses.** Record once; learn forever. Vendor SAs often partner with marketing/dev-rel for this.
- **Office hours.** Weekly drop-in slot for any customer to bring questions. Scales surprisingly well.
- **Community-driven enablement.** Customers helping customers in a vendor-hosted forum. The SA seeds it.

---

## Phase 7 — Vertical Specialization

The single highest-leverage move in the SA career.

### Why Verticals Matter

A generalist SA at a major cloud knows ML well. A specialist SA at the same company knows ML *and* financial services regulation *and* the actual language a Chief Risk Officer uses. The specialist closes a 7-figure deal because they sound like an insider.

Most senior SAs end up specialized in 1–2 verticals. Picking yours is one of the most consequential career moves.

### The Major ML Verticals

#### Financial services

- Banking (retail, commercial, investment)
- Insurance (P&C, life, health)
- Capital markets
- Payments

Core ML use cases: fraud, credit risk, anti-money-laundering (AML), customer churn, KYC, market prediction, document automation, claims processing.

Key knowledge:

- Model risk management (SR 11-7, the canonical US bank framework)
- Fair lending (ECOA, FCRA)
- Anti-discrimination (DOJ enforcement, NYC LL144)
- SOX, GLBA, NYDFS Part 500
- Markets context: Federal Reserve, OCC, FDIC, CFPB, FINRA, SEC
- Data residency, customer data restrictions
- Stress testing (CCAR / DFAST analogs)
- Explainability requirements (counterfactuals for credit denials)

#### Healthcare and life sciences

- Providers (hospitals, clinics)
- Payers (insurance)
- Pharma
- Medical devices

Core ML use cases: clinical decision support, medical imaging, drug discovery, patient stratification, fraud-waste-abuse detection, real-world evidence, claims automation, voice transcription.

Key knowledge:

- HIPAA, HITECH, BAA agreements
- FDA SaMD (Software as Medical Device) classification
- 510(k) and PMA pathways
- GxP (GMP, GLP, GCP) for pharma
- 21 CFR Part 11 (electronic records)
- De-identification, HIPAA Safe Harbor vs Expert Determination
- The HL7 / FHIR data model
- EHR ecosystems (Epic, Cerner, Athenahealth)

#### Retail and consumer

- E-commerce
- Brick-and-mortar
- Marketplaces
- D2C brands

Core ML use cases: recommendation, search relevance, demand forecasting, dynamic pricing, inventory optimization, fraud, personalization, supply chain, returns prediction.

Key knowledge:

- Unit economics — gross margin, CAC, LTV
- Order/promo/inventory data shape
- POS systems
- Tax and SKU complexity
- Peak-season scaling (Black Friday is real)
- A/B testing maturity (often the most sophisticated experimentation orgs)

#### Manufacturing and industrials

- Discrete (automotive, electronics)
- Process (chemicals, pharma)
- Energy
- Heavy industry

Core ML use cases: predictive maintenance, quality control (vision-based), supply chain, demand sensing, yield optimization, anomaly detection, digital twins.

Key knowledge:

- OT (operational technology) vs IT
- ISA-95 reference model
- PLC, SCADA, historians (OSIsoft PI is huge)
- Edge computing constraints
- Functional safety (IEC 61508, ISO 26262 for automotive)
- Plant economics (uptime is everything)

#### Public sector and defense

- Federal, state, local
- Defense, intelligence
- Public health

Core ML use cases: case prioritization, fraud detection, document understanding, predictive analytics, geospatial intelligence, language services.

Key knowledge:

- FedRAMP (Moderate, High, IL5/IL6 for defense)
- StateRAMP
- ITAR, EAR, CUI handling
- ATO process
- FISMA
- Procurement cycles (slow; multi-year)
- The Section 508 accessibility requirement

#### Media, entertainment, and gaming

- Streaming services
- Music
- Publishing
- Gaming

Core ML use cases: recommendation, content moderation, ad targeting, search, generative content, matchmaking (gaming).

Key knowledge:

- Content licensing
- Rights management
- Live-streaming infrastructure
- Player telemetry shape

#### Telco and edge

Core ML use cases: network optimization, customer churn, fraud (SIM swap, IRSF), capacity planning, RAN intelligence.

Key knowledge:

- The 5G core, MEC
- Carrier-grade SLA expectations
- Regulatory: FCC, ITU, GDPR + ePrivacy

### How to Pick a Vertical

- **Where can you tolerate the long ramp?** 6–18 months to sound credible to insiders.
- **Where is the talent gap?** Healthcare and financial services pay the largest premium for the few who know both ML and the regulatory shape.
- **Where does the deal flow concentrate?** Look at your potential employer's customer base.
- **Where do you have an unfair advantage?** A prior background in any of these verticals is gold; never waste it.

### How to Build Vertical Credibility

1. **Read the regulators' actual documents.** SR 11-7 cover-to-cover. FDA AI/ML SaMD guidance. They're short and revealing.
2. **Subscribe to the trade press.** American Banker, Modern Healthcare, Risk.net, Insurance Journal, RetailDive, Industry Week. Skim daily.
3. **Attend the conferences your vertical attends.** Money 20/20 (fintech), HIMSS (healthcare), NRF Big Show (retail), Hannover Messe (manufacturing).
4. **Build vertical-specific demos.** A generic "fraud demo" is forgettable. A "fraud detection demo using SR 11-7-compliant model documentation and OCC-aligned validation evidence" stops the room.
5. **Get a relevant cert if your vertical has one.** AAFM CFA (finance), AHIMA RHIA (healthcare information management — niche but real for healthcare-leaning SAs), AWS Industry-specific specialty certs.

---

## Phase 8 — Certifications That Matter

Certifications are signals, not skills. They open doors; they don't substitute for substance. For SA roles specifically, they're more valued than for IC roles — buyers and managers screen for them.

### The "Universally Useful" Certs

- **AWS Solutions Architect Professional** — the de facto F500 ML/data architecture cert
- **AWS Machine Learning Specialty** — narrower, useful
- **Google Cloud Professional ML Engineer** — for GCP-leaning shops
- **Google Cloud Professional Cloud Architect** — broader
- **Databricks Generative AI Engineer Associate / Machine Learning Professional** — for Databricks-heavy markets
- **Snowflake SnowPro Advanced: Data Scientist / Architect** — narrower; useful in finance/CPG
- **NVIDIA-Certified Associate / Professional (Generative AI / LLMs)** — useful for inference-infrastructure roles
- **HashiCorp Terraform Associate** — infrastructure baseline
- **Kubernetes — CKA / CKAD / CKS** — broadly useful

### The Industry-Specific Worth-Doing

- **AWS Certified Advanced Networking — Specialty** — relevant if you target large enterprise
- **AWS Certified Security — Specialty** — financial / government
- **HIPAA-specific from cloud vendors** (HIPAA-aware solution training)
- **FedRAMP-aware training** (specific to federal SA roles)

### The Generally-Skip

- Vendor product certs of small vendors with no market gravity
- "AI Foundations" certs from anyone — too vague to signal
- Pay-to-issue "AI strategist" certs

### How to Use Certs

- Get 2–3 that match your target role
- Put them on LinkedIn and your resume
- Mention them in cover letters when the JD mentions them
- Do not lead with them in interviews — lead with concrete work

A senior SA with three AWS certs and no real customer stories is weaker than a senior SA with no certs and three landed deals. Substance always wins; certs accelerate the door-opening.

---

## Phase 9 — The Pre-Sales Motion (Vendor SAs)

The single most distinctive part of the vendor SA job. Most engineering tracks never see this.

### The Stages of a Deal

A typical enterprise ML platform deal:

1. **Lead generation.** Marketing brings interest. Inbound or outbound.
2. **Qualification (BANT / MEDDIC).** Is this a real opportunity? Budget, authority, need, timeline.
3. **Discovery.** The conversation in Phase 4 above. SA usually leads.
4. **Solution design.** SA designs an architecture matched to the customer's situation.
5. **Demo / proof.** Live demo or POC.
6. **Proposal and negotiation.** AE leads; SA confirms technical scope.
7. **Procurement / legal.** Security review (the SA fields most questions). Contracts, MSAs, DPAs.
8. **Close.** Signed.
9. **Onboarding / land.** Customer's first 90 days. SA-heavy.
10. **Expansion.** Renewal + grow. SA partners with Customer Success.

The SA shows up in 3, 4, 5, 7, 9, 10 most heavily. The AE owns 2, 6, 8.

### Common Qualification Frameworks

- **BANT** — Budget, Authority, Need, Timeline. Simplest.
- **MEDDIC** — Metrics, Economic buyer, Decision criteria, Decision process, Identify pain, Champion. The enterprise standard.
- **CHAMP** — Challenges, Authority, Money, Prioritization. Modern variant.

You don't need to memorize them. You need to know: **what's the customer's actual pain, who decides, who controls budget, what's the timeline, and what's the criteria for "yes."** If you can't answer these, the deal isn't qualified.

### Champions

The single most important relationship in any deal: the **champion**. The customer-side person who advocates for your solution internally. Without one, deals die quietly.

How you cultivate a champion:

- Make them look good. Their boss should hear that their idea worked.
- Send them ammunition. Custom reference architectures. Data points for their internal pitch.
- Be reachable. Answer messages fast.
- Be honest. The fastest way to destroy a champion is to oversell something they then have to walk back internally.

### Security and Procurement Review

The longest, most-frustrating phase of most enterprise deals. SA's role: shepherd the customer's security team through your platform's controls. You'll fill out:

- SIG / CAIQ questionnaires
- Vendor security questionnaires (often 200+ questions)
- Privacy impact assessments
- Penetration test results requests
- SOC 2 / ISO 27001 / HIPAA / FedRAMP attestations
- DPA negotiations

A good SA has these documents memorized. The customer should never wait 3 weeks for an answer they should get in 3 hours.

### Negotiation Patterns You'll See

You generally won't negotiate price (AE owns that). But you'll be in rooms where:

- The customer demands a feature that doesn't exist. SA must say no honestly while pointing to the roadmap.
- The customer wants a custom SLA. SA evaluates feasibility.
- The customer asks "what if we use 10x more than projected." SA models the cost.
- The customer asks "what's your data training policy." SA confirms the contract terms.

Never commit to a feature that doesn't exist. Never imply a roadmap date. These are how SAs lose careers.

---

## Phase 10 — Internal SA (Enterprise Architecture Office)

A different role with overlapping skills.

### What the Architecture Office Does

At a large F50 (typically 5K+ engineers), there's usually an Enterprise Architecture (EA) function. Inside, ML/AI architects:

- Set technical standards across the org
- Review proposed architectures from business units
- Run a vendor selection process for shared platforms
- Define reference architectures for internal teams
- Coordinate cross-BU ML capability builds
- Govern the AI risk framework
- Liaise with vendors as the central buyer

This is the closest IC-equivalent of a vendor SA — same skills, internal customers.

### Power and Influence

A common trap: internal SAs assume they can mandate. They can't. Engineering leaders, when pushed, find ways to bypass.

The art is **influence without authority**:

- Be the place teams *want* to come for advice
- Make the standard path the easy path (paved roads)
- Document why; let the why convince
- Pick the battles that matter; cede the small ones

### The Paved Road

The most powerful internal-SA pattern. Build (or sponsor) the easiest path for teams to do the right thing:

- A starter template repo with the org's standard ML pipeline
- A managed feature store everyone can use
- A shared LLM gateway with the right enterprise controls
- A model registry the security team has already blessed

Teams will use the paved road because it's faster than building their own, *and* it passes the compliance reviews automatically. You've made the right thing the default.

### Standards You'll Likely Write

- ML platform reference architecture (what's "blessed")
- Approved vendor list
- Model documentation standard (model cards for everyone)
- ML CI/CD standard
- LLM usage policy (which providers approved for which data)
- Data classification + handling for ML
- AI risk assessment template
- Deprecation policy for production ML

### Stakeholder Map for Internal SA

Same stakeholder categories as in the IC architect track, but rotated:

- **Executive sponsors** — CIO, CTO, Chief AI Officer
- **Business unit CTOs / VPs of Engineering** — your peers; you need their buy-in
- **Application architects** in each BU — your equivalents
- **Compliance, legal, risk** — heavy collaboration
- **Vendor account teams** — they sell to you; you set the strategy
- **Outside consultants** — Accenture / Deloitte / etc., often parallel to you

---

## Phase 11 — Consultancy SA Track

The consultancy variant: Accenture, Deloitte, EY, PwC, KPMG, IBM Consulting, Capgemini, ThoughtWorks, Slalom, BCG/McKinsey/Bain digital practices.

### What's Different

- **Project-billed.** You charge a client $X/hour or $Y/day.
- **Utilization metric.** Your billable percentage matters. Too low → fired. Too high → burnout.
- **Diverse contexts.** Bank in Q1, retailer in Q2, hospital in Q3.
- **The pyramid.** Many juniors, fewer seniors, few partners. Climb or stagnate.
- **Travel.** Historically very heavy; somewhat reduced post-2020 but still significant.

### What's the Same

- Reference architectures matter
- Vertical expertise matters
- Discovery, design review, POC, workshop skills all the same
- Customer outcomes are the unit of success

### When Consultancy Makes Sense

- You want exposure to many industries before specializing
- You value the on-the-job training (Big-4 consultancies invest heavily in junior development)
- You're early in career and want the brand on your resume
- You like project-based variety more than long-term ownership

### When It Doesn't

- You hate travel
- You want to live with the consequences of your architecture
- You're optimizing for technical depth over breadth
- You want to build (not advise)

### The Promotion Ladder (Approximate)

| Title | Years | Role |
|---|---|---|
| Analyst / Associate | 0–2 | Junior IC; learning the work |
| Senior Associate / Consultant | 2–4 | IC owning deliverables |
| Manager | 4–7 | Leading project teams |
| Senior Manager | 7–10 | Leading multi-project programs |
| Director / Principal | 10+ | Selling + delivering |
| Partner | 12+ | Equity; primarily sells |

The "up or out" reality is real at the Big-4 levels. Exit options are excellent (industry roles at senior IC or director levels), so it's a reasonable 4–8 year sojourn.

### The Consultancy SA's Edge

The genuine skill consultancies build that internal teams don't: **structured problem solving under time pressure with executive audiences**. You'll be in the room with the CIO 3 weeks into a project. You'll deliver a board-ready presentation in 48 hours. These are skills that travel — and exit options to industry "Director of ML" roles are excellent.

---

## Phase 12 — The SA's Toolkit

What SAs actually use day-to-day.

### Customer-Facing

- **A great laptop dev environment** that demos cleanly
- **A backup demo recording** for when the live one breaks
- **A "warm starter" cloud account** with credits and pre-built sandboxes
- **A diagramming tool fluently** — Mermaid, Excalidraw, Lucidchart, Miro
- **A presentation template** that looks like your company's brand
- **A note-taking system** — every customer call gets notes that survive months later
- **A CRM** — Salesforce, HubSpot, vendor-internal. Update it.

### Internal

- **A library of reference architectures** — your personal Git repo of reference documents
- **A library of customer case studies** (anonymized) you can quote
- **A FAQ doc** of the questions you've been asked 5+ times
- **A network of internal experts** (product managers, support engineers) you can pull in fast

### Communication

- **Async-first writing skill.** Most of your influence is in writing. PR descriptions, follow-up emails, design review comments, internal Slack threads.
- **Presentation skill.** You'll present to 5–500-person audiences. Comfortable, conversational, on-message.
- **Time management.** SA calendars are dense. Calendar-block thinking time, or it disappears.

### Staying Technical

- **A personal lab.** A side project that keeps your hands in code.
- **Hands-on with every major new release** of your platform. You can't recommend what you haven't touched.
- **Read open-source code.** vLLM, KServe, Kubeflow, Airflow — read the source. You'll be a more credible advisor.

### Staying Current

- **Vendor docs** for adjacent platforms — you need to be able to compare honestly
- **Industry analyst reports** — Gartner, Forrester, IDC. Customers read these; you should know what they say.
- **Customer-facing podcasts and webinars** — your customers listen to these
- **Sales / pre-sales training** — yes, attend it. Even technical SAs benefit.

---

## Phase 13 — Career Paths and Trade-offs

How SA careers compare to IC engineering careers.

### Compensation (Approximate, US, 2026)

#### Vendor SAs at major cloud / platform companies

- **Associate / Junior SA:** $180K–$260K total comp
- **Senior SA:** $280K–$450K
- **Principal SA:** $400K–$650K
- **Distinguished SA:** $600K+
- **SA Manager / Director:** $400K–$800K (people management)

Variable comp is 20–40% of total for individual SAs. Tied to account performance and consumption.

#### Internal SAs (F50 enterprise architecture)

- **Senior Architect:** $250K–$400K
- **Principal / Distinguished Architect:** $400K–$650K
- **Chief Architect / VP-IC:** $600K+

Lower variable; closer to senior engineering comp.

#### Consultancies

- **Manager:** $200K–$350K
- **Senior Manager:** $300K–$500K
- **Director:** $400K–$700K
- **Partner:** $700K+ (with significant variability; partnership is equity)

### Compared to Senior IC at the Same Company

Vendor SA total comp is usually similar to senior IC total comp at the same company, with a different mix (more variable, less RSU). Some years it beats senior IC; some years it lags.

Internal SA total comp typically matches staff-engineer comp at the same company.

Consultancy partner comp at top firms exceeds senior IC at most tech companies but takes 12–18 years and isn't guaranteed.

### Exit Options

Strong from any of the three:

- Vendor SA → Director of Platform / VP of Engineering at customer-side companies
- Vendor SA → another vendor (senior SAs are highly poachable across cloud / platform vendors)
- Internal SA → Director of Architecture / CTO at smaller companies
- Consultancy → industry roles at director levels; or → vendor SA; or → CIO / CDAO trajectories

The breadth of customer / industry exposure makes SAs unusually flexible.

### The SA-vs-IC Question

A useful frame for which to pick:

- **Pick IC architect** if you want to live with your own decisions, value deep ownership, dislike sales, prefer steady cadence, want to optimize for technical depth.
- **Pick SA** if you like variety, get energy from people, are comfortable with presentation and persuasion, want to see many industries, value flexibility over deep ownership.

Both can be deeply technical. The difference is the audience and the artifact.

### The Reverse Move

It is common to do both at different career stages. A typical pattern:

- 4–8 years IC building things
- 3–5 years vendor SA learning the market, building customer relationships
- Back to IC at director/VP level with much better business judgment

Or:

- 3 years consultancy after college, get the brand and the breadth
- 5–10 years IC at a great company, build the depth
- Director / VP role combining both

Most senior leaders in ML platform roles have done both sides. Plan for the long game.

---

## Phase 14 — Common SA Failure Modes

What to watch for in yourself.

### The Over-Promiser

Says yes to capabilities that don't quite exist. Customer adopts; reality bites; trust evaporates.

**Fix:** Discipline. Never commit to a feature or date without product confirmation. Default to "let me check and get back to you within 24 hours."

### The Vendor Apologist

Defends the platform against every criticism. Loses customer trust because customers know the platform has flaws.

**Fix:** Acknowledge weaknesses honestly. "You're right, our cold-start latency is a known issue. Here's the roadmap; here's the workaround."

### The Solo SA

Tries to handle everything alone. Burns out. Misses leverage from product, engineering, customer success, partner ecosystem.

**Fix:** Build internal network. Know who to call for what. Bring in specialists.

### The Technical-Only SA

Refuses to engage with business framing. Falls back to architecture diagrams when the customer's actual question is "what's the ROI."

**Fix:** Learn business vocabulary. Read the customer's annual report. Translate technical capability into customer-business language.

### The Demo-Driven SA

Lives demo-to-demo. Doesn't build durable artifacts (reference architectures, case studies, FAQs).

**Fix:** Invest 20% of time in artifact creation. Compounds.

### The Politically Naive SA

Walks into a meeting without knowing who's pro / con / indifferent. Says the wrong thing to the wrong person.

**Fix:** Map the room before the room. Ask the champion: "who's there, what's their angle?"

### The Career-Stalled SA

Stays at the same level for 5+ years. Hasn't picked a vertical. Knows everything generically; nothing deeply.

**Fix:** Pick a vertical. Pick a sub-specialty. Own it visibly.

---

## Phase 15 — Interview Preparation for SA Roles

How SA interviews differ from engineer interviews.

### The Typical Loop

For a senior SA role at a major cloud / platform vendor:

1. **Recruiter screen.** Career story, motivation for SA over IC.
2. **Hiring manager.** Customer scenarios, ML breadth.
3. **Technical / architecture round.** Design a system in front of the panel.
4. **Customer presentation.** Present a solution to a fictional customer scenario.
5. **Behavioral / leadership.** Influence, conflict, customer stories.
6. **Cross-functional.** AE peer or product peer; how you collaborate.
7. **Executive interview.** Director / VP; strategic fit.
8. **References.**

What's missing: LeetCode (mostly), narrow tool deep-dives. What's added: presentation, customer empathy, business framing.

### The Customer Presentation Round

The most distinctive round. Format: you're given a scenario (a fictional customer's situation) 1–3 days in advance. You prepare a 30–45 minute presentation. The panel role-plays as the customer.

What's tested:

- Discovery (do you ask questions or pitch?)
- Architecture (is your design appropriate for their context?)
- Communication (can you present clearly?)
- Trade-off articulation
- Handling pushback ("our security team won't approve this — what now?")
- Knowing when to say "I don't know, let me get back to you"

Preparation: research the company you're interviewing at. Their reference architectures, their customer base, their competitors. Practice the discovery conversation out loud. Run a mock.

### Common Behavioral Prompts

- "Tell me about a time you helped a customer pick between competing solutions."
- "Describe an architecture you recommended that turned out wrong."
- "Walk me through a difficult conversation with a customer's technical lead."
- "Tell me about a deal you helped close and what you contributed."
- "How do you handle pushback when your platform isn't the right fit?"
- "Tell me about a workshop that didn't go well."

Prepare 6–8 STAR-format stories. Tag each by theme.

### Technical Architecture Round

Like the IC architect technical interview round but with two differences:

1. **Customer context first.** The interviewer will often play a customer; your first 5–10 minutes should be discovery, not whiteboarding.
2. **Multi-vendor honesty.** You should be willing to say "for this scenario, AWS X is the right tool; for that one, GCP Y." A vendor SA who only thinks in their own platform is an immediate red flag.

### Vertical Round (Vendor SA)

If you're targeting a vertical-aligned role (financial services SA, healthcare SA), expect a vertical-specific round. Test your knowledge of:

- The vertical's regulatory environment
- Common use cases and their constraints
- Customer org structures and decision-making
- The major players (competitors, customers, partners)

Skim the trade press for 4 weeks before this interview. Walk in talking like an insider.

### What Hiring Managers Actually Filter For

- **Curiosity.** Did you ask good questions during discovery?
- **Customer empathy.** Did you center the customer or the platform?
- **Technical credibility.** Could you defend your architecture choices?
- **Storytelling.** Could you make abstract trade-offs concrete and memorable?
- **Composure.** Did pushback rattle you?
- **Coachability.** Did you take their challenge gracefully and adjust?

The last one is underestimated. Vendor SA orgs invest in their people; they want someone who'll grow.

---

## Phase 16 — Reading List for SAs Specifically

Beyond the reading lists in the advanced topics and architect tracks, books that are SA-specific:

### Sales and customer skills

1. **The Trusted Advisor** (Maister, Green, Galford) — the canonical book on advisory relationships
2. **Solution Selling** (Bosworth) — old but foundational
3. **The Challenger Sale** (Dixon, Adamson) — modern enterprise sales theory
4. **SPIN Selling** (Rackham) — discovery question methodology
5. **Mastering the Complex Sale** (Thull) — for complex, multi-stakeholder enterprise deals

### Influence and communication

6. **Influence: The Psychology of Persuasion** (Cialdini)
7. **Made to Stick** (Heath & Heath) — making ideas memorable
8. **Difficult Conversations** (Stone, Patton, Heen)
9. **The Pyramid Principle** (Minto) — structuring written and spoken arguments. Required reading at consultancies.
10. **Resonate** (Duarte) — presentation design

### Industry / business

11. **The Innovator's Dilemma** (Christensen) — vocabulary every customer exec uses
12. **Crossing the Chasm** (Moore) — the technology adoption lifecycle
13. **Good Strategy / Bad Strategy** (Rumelt) — distinguishing the two
14. **Competing on Analytics** (Davenport & Harris) — analytics maturity model
15. **The Phoenix Project** (Kim) and **The Unicorn Project** (Kim) — IT transformation in narrative form

### ML and AI specifically for SAs

16. **AI 2041** (Kai-Fu Lee, Chen Qiufan) — for the "what's coming" vocabulary execs use
17. **The Coming Wave** (Mustafa Suleyman) — same purpose, different lens
18. **The AI-First Company** (Carbone, Carbone) — strategic positioning of AI in enterprises
19. **Prediction Machines** (Agrawal, Gans, Goldfarb) — economics of AI
20. **Designing Machine Learning Systems** (Chip Huyen) — your technical baseline (already on previous lists)

### The two SA-must-reads

The most-cited books in the SA community:

- **The Trusted Advisor** (Maister, Green, Galford) — the role you're actually playing
- **The Pyramid Principle** (Minto) — how to write and present like a senior consultant

Read both twice. They change how you operate.

---

## Phase 17 — Hybrid Paths

Many of the most successful ML platform leaders zigzag between SA and IC architect. Worth knowing the common patterns.

### IC → SA → IC

The most common high-impact path:

1. 4–8 years IC engineer, build deep technical credibility
2. 3–5 years vendor SA, learn the market and customer empathy
3. Return as Director / VP / Chief Architect with both technical depth and business judgment

The IC time builds substance; the SA time builds breadth and communication. Together they're rare.

### SA → Founder

A surprisingly common exit. SAs at cloud vendors see hundreds of customer problems. Patterns repeat. The SA who notices a $200M problem solved badly across 30 customers has a startup idea worth pursuing.

Examples: many of the founders of ML observability companies (Arize, Fiddler, WhyLabs) came from cloud / platform SA roles.

### IC → Consultancy → CIO/CDO Track

For those who want the executive trajectory rather than the deepest-IC trajectory:

1. 4–6 years IC for substance
2. 4–6 years consultancy for breadth + executive exposure
3. Director / VP of Engineering at a smaller F500 for ownership
4. CIO / CDO / CTO eventually

The consultancy years compress executive-track learning that takes much longer in industry alone.

### SA → Customer Success Engineering Leadership

Some SAs migrate from pre-sales to post-sales (Customer Success / Solutions Engineering for retained customers). Often a calmer life with similar comp at the senior levels. Strong fit for SAs who like the relationships but tire of the quarterly intensity.

### Pure SA Lifer

Some SAs stay SA for 20+ years and reach Distinguished / Chief Architect status at the major clouds. The work stays intellectually rich, the comp scales, and the variety doesn't get boring.

The honest signal: if you've been an SA for 8+ years and you still enjoy the new-customer-each-week rhythm, you're probably built for it.

---

## A Closing Thought

The SA path is **as serious as the IC path**, just shaped differently. Both end at the same total compensation tier; both require world-class technical credibility; both require the surface area beyond pure technical skill (writing, communicating, deciding under ambiguity).

The right question isn't "is SA less prestigious than IC." It's "do I get more energy from designing systems alone or from helping customers design theirs?" Both are legitimate sources of energy. Picking the one that matches you is the most important career decision in the next decade.

If you're not sure, the answer is to do a 1–2 year tour in the role you've never done. IC engineers benefit from a stint near the customer; SAs benefit from a stint owning a production system. Most of the strongest senior ML platform leaders have done both.

---

## Where This Curriculum Ends

The full curriculum covers:

- MLOps foundations through specialization
- Portfolio projects for F50 roles
- Advanced technical topics
- The IC architect track
- The Solutions Architect track (this chapter)

Work through this seriously and you have credible paths to:

- Senior MLOps / ML Platform Engineer at any F50 (foundations through specialization)
- Staff / Principal / IC Architect (add the advanced topics and architect track, plus practiced application)
- Senior / Principal Solutions Architect at any major cloud / platform vendor (foundations plus this chapter and customer-facing experience)
- Director / VP of ML Platform anywhere (most of the above plus 5–8 years)

The compound interest on this work over 3–5 years is enormous. Few engineers build this surface area. The ones who do become indispensable — whether they call themselves architects, SAs, principals, or directors. The work is what compounds; the title is just signaling.

Now go pick one path and commit. Or — if you're early enough in career — pick a portfolio project and let the work tell you which path you want.
