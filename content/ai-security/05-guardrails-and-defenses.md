# 05 — Guardrails and Defenses

The previous lessons made a hard argument: the core LLM vulnerabilities cannot be detected or trained away, only contained. So why is there a whole lesson on filtering? Because filtering is the outer layer of a defense-in-depth strategy, and while no filter is a *guarantee*, a good one removes the high-volume, low-effort attacks so your expensive architectural defenses (Lesson 06) only have to hold against the determined adversary. The mistake is not *using* guardrails; the mistake is *believing* them. This lesson is about deploying them with clear eyes about exactly where they hold and where they break.

## What a guardrail is and is not

A **guardrail** is a check that runs *around* your main model — screening what goes in, what comes out, or both — and can allow, block, redact, or rewrite. It is a classifier or validator, and like any classifier it has false positives and false negatives. That single fact governs everything: a guardrail *shifts a distribution* (fewer bad things get through) but does not *close a hole* (some bad things still get through, and some good things get blocked). Deploy them as probabilistic filters, budgeted against a threat, not as authorization boundaries. Anything that must *never* happen — an irreversible action, access to a secret — needs a *capability* control (Lesson 06), not a guardrail.

The layered picture, outer to inner:

```
user input ─▶ [input guardrail] ─▶ LLM (+ retrieval, tools) ─▶ [output guardrail] ─▶ user/sink
                    │                                                    │
              block / rewrite                                     redact / block / validate
```

## Input-side guardrails

On the way in, you screen the user's message *and* any untrusted content headed for the context (retrieved documents, tool outputs):

- **Prompt-injection / jailbreak detectors.** Classifiers trained to flag adversarial inputs — Meta's **Prompt Guard** (a small classifier for injection/jailbreak text), commercial options like **Lakera Guard**, and the injection probes in red-team suites. Useful against known attack shapes; weak against novel or obfuscated ones.
- **Topic and policy rails.** Restrict the assistant to its domain ("only answer questions about our product"). Off-topic or policy-violating inputs are refused before they reach the model.
- **PII / secret detection on inputs.** Tools like **Microsoft Presidio** detect and optionally redact personal data before it is logged or sent to a third-party API — important for compliance regardless of attacks.

The key limitation, carried from Lesson 03: a single-message input filter cannot see a **multi-turn** attack (Crescendo, many-shot) whose payload is spread across the conversation. Input guardrails must consider conversational state, not just the latest message.

## Guardrail models

The reference open guardrail model is Meta's **Llama Guard**. **Llama Guard 3** (July 2024) shipped as 1B and 8B text models plus an 11B vision model; **Llama Guard 4** (12B, April 2025) unifies text and image classification. It is aligned to the **MLCommons hazards taxonomy** — 14 categories, S1–S14 (violent crimes, non-violent crimes, sex crimes, child exploitation, defamation, specialized advice, privacy, intellectual property, indiscriminate weapons, hate, self-harm, sexual content, elections, code-interpreter abuse) — and returns `safe` / `unsafe` plus the violated category codes. The call is structurally simple, and the same model screens both directions by changing the role of the classified message:

```python
def guardrail_check(text: str, role: str) -> tuple[bool, list[str]]:
    """Classify one message with Llama Guard. Returns (is_safe, categories)."""
    verdict = llama_guard.generate(conversation=[{"role": role, "content": text}])
    lines = verdict.strip().splitlines()          # "safe"  OR  "unsafe\nS1,S9"
    if lines[0] == "safe":
        return True, []
    categories = lines[1].split(",") if len(lines) > 1 else []
    return False, categories

ok_in, cats = guardrail_check(user_msg, "user")
if not ok_in:
    return refuse(cats)
answer = model.run(user_msg)
ok_out, cats = guardrail_check(answer, "assistant")
if not ok_out:
    return refuse(cats)
```

The field beyond Llama Guard is crowded and worth knowing: **ShieldGemma** (Google, Gemma-based), **WildGuard**, **IBM Granite Guardian**, and **NeMo Guard** family models. They differ in taxonomy, size, and modality but share the interface — text in, safety verdict out.

## Frameworks and managed services

You rarely wire guardrails by hand in production. The landscape:

- **NeMo Guardrails** (NVIDIA, open source): programmable rails written in a DSL called **Colang**, strong for *dialog-flow* policies ("the bot may only discuss X; if asked Y, deflect"). It orchestrates checks (including guardrail models and its own) around your app. Runs anywhere you run Python.
- **Guardrails AI** (open source): a *validator* framework — declarative checks on outputs (format, PII, toxicity, competitor mentions, valid JSON) that can auto-correct, re-ask, or reject. Its "Guardrails Hub" is a library of pluggable validators.
- **AWS Bedrock Guardrails**: managed, policy-configurable filtering — but only for **Bedrock-hosted** models, and priced per use.
- **Azure AI Content Safety** and **OpenAI Moderation API**: managed content classifiers. The OpenAI Moderation API is free and a reasonable `$0` baseline; both are content filters, not complete solutions.
- **Perspective API** (Google Jigsaw): toxicity scoring, narrower scope.

Pick by fit: NeMo for conversational policy, Guardrails AI for output validation, a guardrail model for content classification, a managed service if you are already on that cloud and want less to operate.

## Constrained decoding — what it does and does not buy you

**Constrained (structured) decoding** forces the model's output to conform to a grammar or JSON schema by masking, at each step, the tokens that would violate the structure. It is genuinely valuable — it guarantees the output *parses*, which eliminates a whole class of "the model returned almost-JSON" bugs and is how reliable tool-calling works.

But be precise about the guarantee. Constrained decoding guarantees **structure, not safety**. A schema-valid object can still carry a malicious *value*: valid JSON whose `sql` field is a `DROP TABLE`, a well-formed function call with a harmful argument, a syntactically perfect URL pointing at an exfiltration endpoint. Constraining the grammar does not validate the semantics. This is the direct bridge to output handling.

## Output-side guardrails and structured validation

The outbound side is its own OWASP entry — **LLM05 (Improper Output Handling)** and **LLM02 (Sensitive Information Disclosure)** — and it is where a surprising number of real bugs live:

- **PII / secret redaction.** Run a detector (Presidio, Guardrails AI validators, a Llama Guard privacy category) over the response before display or logging, and redact. This catches memorized-data leakage (Lesson 04) regardless of how it got into the output.
- **Content moderation on outputs.** The output classifier catches disallowed content the model produced — this is where jailbroken output is caught even when the input filter missed the multi-turn attack that produced it. Bidirectional screening matters precisely because the two directions catch different failures.
- **Never trust model output as code.** LLM05 exists because applications pass model output into a downstream *sink* unsanitized: model-generated SQL into a database, model-generated HTML into a page (stored XSS), model-generated shell into `exec`, a model-supplied URL into an image tag (the exfiltration channel of Lesson 02). **Treat model output as untrusted data at every boundary**: parameterize SQL, escape/sanitize HTML, never `eval`, and do not auto-render model-supplied URLs. Constrained decoding got you valid JSON; *this* step decides the values are safe to act on.

## When guardrails fail

The honesty section, because deploying guardrails without knowing their limits is worse than not deploying them (false confidence):

- **They are far from perfect.** Independent benchmarks put guardrail classifiers at **F1 roughly 0.75–0.88** on standard sets. That is a useful filter and a terrible guarantee — one in five to one in eight adversarial inputs slips or is wrongly blocked.
- **They collapse under adversarial and long-context pressure.** Accuracy that looks fine on a clean benchmark degrades sharply on obfuscated, encoded, or adversarially-optimized inputs, and on very long contexts. One benchmark reported a **1.0 false-positive rate on long-context inputs** for a managed offering — it flagged *everything*. The evaluation conditions matter enormously; a vendor's headline F1 was almost certainly not measured under attack.
- **Over-refusal is a real cost.** Tighten the filter and it starts blocking legitimate requests (the security-help question, the medical question, the code that looks like an exploit but is a unit test). Safety is a two-sided error; both directions have a price.
- **They add latency and cost.** Every guardrail call is another model invocation. Bidirectional screening with a guardrail model can double your per-request model cost and add latency, which pushes teams toward smaller/cheaper classifiers — which are weaker.
- **They do not authorize actions.** The recurring theme: a guardrail can say "this text looks unsafe," but it cannot make "the agent must not wire money without approval" true. That is a capability control, not a filter.

The design conclusion: **layer them, and rely on architecture for anything that must not happen.** Guardrails are the cheap, high-recall-ish outer net that cuts attack volume; least privilege, isolation, and human gates (Lesson 06) are what actually hold when the net has a hole — and it always has a hole.

## Key takeaways

- A guardrail is a **probabilistic filter**, not an authorization boundary. It shifts the distribution of what gets through; it does not close the hole. Anything that must *never* happen needs a capability control instead.
- Screen **both directions**: input guardrails (injection/jailbreak detectors, topic rails, PII detection) and output guardrails (moderation, PII redaction, structured validation) catch *different* failures. Input filters cannot see multi-turn attacks.
- The tool landscape: **Llama Guard 3/4** (MLCommons S1–S14 taxonomy) and peers (ShieldGemma, WildGuard, Granite Guardian) as models; **NeMo Guardrails** (Colang dialog rails), **Guardrails AI** (output validators), **Bedrock Guardrails**, and moderation APIs as frameworks/services.
- **Constrained decoding guarantees structure, not safety** — a schema-valid object can carry a malicious value. Always validate output *semantics* at the sink: parameterize SQL, escape HTML, never `eval`, do not auto-render model URLs (LLM05).
- Be honest about limits: guardrail F1 is roughly **0.75–0.88**, **collapses on adversarial/long-context inputs**, causes **over-refusal**, and **costs latency/money**. Layer them and lean on architecture for the guarantees.

## Try it

Stand up a two-layer guardrail around a small chatbot and measure it honestly. Put a guardrail model (Llama Guard, or a moderation API as a free stand-in) on both the input and the output. Then build two tiny eval sets: a "benign" set of ordinary, safe requests (including a few that *sound* edgy but are fine — "how do I kill a stuck Linux process?", "explain how SQL injection works so I can prevent it") and an "adversarial" set drawn from the jailbreak families in Lesson 03 (an obfuscated request, a role-play framing, a two-turn Crescendo). Run both sets and compute, per direction, how many adversarial inputs slipped through (false negatives) and how many benign ones were blocked (false positives). You will see F1 well under 1.0 with your own eyes, watch the multi-turn attack walk straight past the single-message input filter, and catch at least one over-refusal — the exact three failure modes this lesson warns about, which is why Lesson 06 stops trusting filters for the things that truly matter.
