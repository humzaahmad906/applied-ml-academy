# 06 — Advanced Topics: Everything Else Worth Knowing — Part 4 of 5: Disaster Recovery, Compliance, and Architectural Patterns

This is part 4 of the Advanced Topics reference (18 phases across 5 parts). [Part 3](06c-advanced-topics.md) covered federated query, Kubernetes, and specialized stores (Phases 11–13); here we cover Phases 14–16: backup and disaster recovery, compliance and privacy, and the architectural patterns that come up in senior interviews.

---

## Phase 14 — Backup, Disaster Recovery, and Business Continuity

When systems fail (and they will), you're the one being woken up at 3 AM. This phase is about reducing that probability and minimizing impact.

### RPO and RTO

Every system needs to answer two questions:

- **RPO (Recovery Point Objective):** How much data can you afford to lose? "5 minutes" means continuous replication. "24 hours" means daily backups are fine.
- **RTO (Recovery Time Objective):** How fast must you be back online? "5 minutes" means hot standby. "4 hours" means cold backup + restoration.

Different data has different RPO/RTO requirements. The CRM database might be RPO=1min, RTO=10min. The "fun analytics" data lake might be RPO=24h, RTO=24h.

### Backup Strategies

- **Snapshot-based** — Iceberg snapshots, Snowflake Time Travel, RDS snapshots
- **Continuous replication** — to a read replica or cross-region
- **External backup** — periodic dump to a separate cloud account / region

The rule: **backups that haven't been tested are not backups.** Schedule quarterly DR drills.

### Multi-Region Architectures

Three patterns:

1. **Active-passive** — primary region serves, secondary on standby. Failover takes minutes. Simple.
2. **Active-active** — both regions serve. No failover time. Hard — write conflicts.
3. **Read-replica only** — writes to primary, reads can go to replicas in other regions. Reduces read latency, doesn't help RPO.

### What Can Go Wrong (And What Doesn't Get Talked About)

- A single AZ outage (small)
- A region outage (rare but happens — `us-east-1` outages every 1–2 years on average)
- A control-plane bug (you can't issue API calls anywhere)
- A bad deploy by *the cloud provider* (yes, this happens)
- Account-level lockout (someone leaks creds, AWS suspends your account)

### Exercises

1. For one of your projects, write a runbook: what would you do if BigQuery in us-central1 had a 4-hour outage right now?
2. Set up cross-region replication for a critical bucket. Verify recovery time.
3. Practice restoring from an Iceberg snapshot to a point in time.

---

## Phase 15 — Compliance and Privacy

In F100 data work, "the data engineer who knows compliance" is the data engineer who gets promoted.

### The Frameworks

- **GDPR** (EU) — right to access, right to deletion ("right to be forgotten"), data residency, breach notification within 72 hours. Applies to anyone with EU customers.
- **CCPA/CPRA** (California) — similar shape, narrower scope. Applies to companies serving Californians.
- **HIPAA** (US healthcare) — PHI protection, audit logging, BAA agreements.
- **SOC 2** — security audit framework. Type 2 audits cover operations over time (typically 6–12 months). Required by most enterprise customers.
- **PCI-DSS** — payment card data. Strict tokenization requirements.

You don't need to be a lawyer. You need to know:

1. Which frameworks apply to your company
2. What technical controls each requires
3. How to design data platforms that bake the controls in from day one

### PII Handling Patterns

- **Tagging at the column level** — every PII column tagged in your catalog. Most modern catalogs support this.
- **Tokenization** — replace real PII with tokens; keep the mapping in a separate, heavily restricted service.
- **Format-preserving encryption** — encrypt values such that the encrypted form is the same type (SSN stays SSN-shaped). Useful for keeping schemas unchanged.
- **Differential privacy** — for aggregate analytics on sensitive data, add calibrated noise. Used at Apple and the US Census.
- **Synthetic data** — generate realistic-but-fake data for dev environments.

### The Right to Be Forgotten — Operationally

GDPR Article 17 requires deleting all data about a user on request. In a normal warehouse this is straightforward (DELETE). In a lakehouse this is harder:

- Hudi MOR tables handle deletes natively
- Iceberg supports equality deletes — write a delete file, compaction physically removes later
- Delta supports DELETE — `DELETE FROM ... WHERE user_id = X`

But: snapshots and time travel mean data persists in old versions. You may need to expire snapshots faster on tables containing PII to comply.

### What to Build for Your Portfolio

Add a "compliance" section to one of your existing projects:

- A column-level data classification (mark which columns are PII)
- A delete-user pipeline that propagates a deletion request to all downstream tables
- An audit log of every access to PII columns
- A snapshot expiration policy that respects retention rules

Mention this in your README. It's a senior signal.

---

## Phase 16 — Architectural Patterns

A grab bag of patterns that come up in senior interviews and real architecture work.

### The Outbox Pattern

Covered in full in [03b — Advanced Guide: Kafka, Streaming, and CDC](03b-advanced-guide.md#the-outbox-pattern) — the dual-write problem, the transactional fix, and Debezium's role as publisher.

### Event Sourcing and CQRS

**Event sourcing:** Store the *log of changes* as the source of truth, not the current state. Current state is a fold/aggregation of the events.

**CQRS (Command Query Responsibility Segregation):** Separate the write model (commands) from the read model (queries). Writes go to the event log; reads come from materialized projections.

For DE: any system with an event log is event-sourcing-shaped. Kafka + Iceberg gives you this naturally. Multiple read projections (warehouse, real-time OLAP, search index) is CQRS.

### Reverse ETL

The pattern that flipped the data flow: take warehouse data and push it back into operational tools.

- Lead scores → Salesforce
- User segments → Marketing automation
- Feature flags → Production app

Tools: Hightouch, Census, Polytomic, Grouparoo. dbt is increasingly adding native support.

This is now a real category of DE work. Senior DEs at SaaS companies often own a reverse-ETL platform.

### Lambda vs Kappa Architecture

**Lambda:** Two separate paths — batch (slow, accurate, historical) + speed (fast, approximate, recent). Merge for queries. Operationally complex (two codebases).

**Kappa:** One path — streaming all the way. Reprocess history by replaying the stream. Simpler. Requires the streaming substrate (usually Kafka) to retain history long enough.

Most modern architectures are Kappa-ish, using lakehouse formats so the "batch" view is just a view of the streaming-written table.

### Data as a Product

The mindset shift: data isn't a side effect of operations; it's a deliverable with consumers, SLAs, documentation, versioning.

Implications:

- Every dataset has an owner
- Every dataset has a documented schema and freshness SLA
- Breaking changes follow deprecation policies
- Consumers can self-discover and subscribe

This is the mindset behind data mesh (the federated data mesh portfolio project). Worth knowing even if you don't build a mesh — F100 hiring managers are increasingly looking for this thinking.

---

## You can now

- Design RPO/RTO-appropriate backup and multi-region strategies, and write a runbook for a regional cloud outage.
- Map GDPR/CCPA/HIPAA/SOC 2/PCI-DSS requirements to concrete technical controls (tokenization, format-preserving encryption, differential privacy) and implement the right to be forgotten in a lakehouse.
- Recognize and name the architectural patterns that come up in senior interviews: outbox, event sourcing/CQRS, reverse ETL, Lambda vs Kappa, and data-as-a-product.

This is part 4 of the Advanced Topics reference. Next: the bookshelf, agentic data engineering, and closing guidance (Phases 17–18) in [Part 5](06e-advanced-topics.md).
