# 08 — The ML / AI Solutions Architect Track — Part 2 of 2: Pre-Sales, Internal and Consultancy SA, Career Paths, and Interviews

This is part 2 of the ML / AI Solutions Architect Track lesson. Here we cover the vendor pre-sales motion, the internal enterprise-architecture-office SA and consultancy SA tracks, the SA's toolkit, career paths and trade-offs, common failure modes, interview preparation, the SA-specific reading list, and hybrid paths between SA and IC architect.

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

---

## You can now

- Locate the SA-specific stages of an enterprise deal (qualification, discovery, solution design, security/procurement review), cultivate a champion, and know where the SA vs. the AE owns each stage.
- Distinguish the internal/enterprise-architecture-office SA (paved roads, influence without authority) and the consultancy SA (billable utilization, the promotion pyramid) from the vendor SA track, and compare their compensation and exit options.
- Recognize the common SA failure modes (over-promiser, vendor apologist, technical-only SA, demo-driven SA) in yourself and apply the corresponding fix.
- Prepare for an SA interview loop — especially the customer-presentation round — with discovery-first framing, multi-vendor honesty, and 6-8 STAR-format stories.
- Plan a multi-year path across IC and SA (or consultancy) roles, using the reading list (The Trusted Advisor, The Pyramid Principle) to build the advisory and communication skills that compound across both.

## Try it

Pick one vertical you have a genuine edge in (or the one whose deal flow you find most attractive) and build a single reference architecture for a recurring problem in it — for example, "RAG over enterprise documents for a regulated bank." Write the full one-pager: a one-sentence problem statement, the context in which it applies (and where it doesn't), a C4 Container-level diagram, each component choice with one sentence of *why*, two variants (K8s-experienced team vs. managed-only), a rough cost model parameterized by document volume and query rate, the top three failure modes with how you'd detect each, and the vertical-specific compliance notes (e.g., SR 11-7 model documentation, data residency). Then run a mock discovery conversation against it out loud — play the customer, ask yourself the six qualification questions (pain, who decides, who controls budget, timeline, criteria for "yes," executive sponsor), and note every place your reference architecture assumes a context the customer didn't actually confirm. Those gaps are exactly what separates a template from advice.
