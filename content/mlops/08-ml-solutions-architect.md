# 08 — The ML / AI Solutions Architect Track — Part 1 of 2: Role, Mindset, Reference Architectures, Discovery, POCs, and Verticals


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

## You can now

- Distinguish the three SA populations — vendor, internal/enterprise, and consultancy — by their comp model, stakeholders, rhythm, and exit options, and reason about which fits you.
- Apply the SA mindset — selling the trade-off not the tool, discovery over pitching, customer context first — to keep a technically correct recommendation grounded in what a specific customer actually needs.
- Write a reference architecture that names its context applicability, component trade-offs, variants, cost model, and failure modes — and know to harvest patterns from 3+ real engagements before writing one.
- Run a structured 90-120 minute discovery conversation, using open questions, silence, and reflective summarization to reach the customer's real problem before proposing anything.
- Scope a POC with a signed one-pager (question, success criteria, in/out scope, timeline, decision date), run a workshop that ships an artifact, and know which certs and vertical are worth investing in.
