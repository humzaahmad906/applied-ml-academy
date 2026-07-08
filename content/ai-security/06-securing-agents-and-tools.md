# 06 — Securing Agents and Tools

Everything so far has been building to this lesson. A chatbot that only emits text can, at worst, say something wrong or embarrassing. An **agent** — an LLM in a loop with tools that read email, run code, query databases, move money, and call APIs — can *act*, and a manipulated agent acts on the attacker's behalf. This is where the prompt injection of Lesson 02 stops being a curiosity and becomes a wire transfer to the wrong account. The GenAI course's *Agent Safety and Guardrails* chapter introduces these defenses; this lesson goes deeper on the capability model, the isolation-pattern catalog, and the MCP threat surface, and treats them as a security engineering discipline rather than a checklist.

The governing principle, stated once and applied throughout: **assume the model will be compromised, and design so that a compromised model still cannot do serious harm.** Detection (Lesson 5) reduces attack volume; capability limits reduce damage. Against a real adversary, the capability limits are what you are actually relying on.

## Excessive agency — the OWASP LLM06 anatomy

The highest-leverage control is bounding what the agent *can* do, because it caps the blast radius regardless of whether an injection or jailbreak succeeds. OWASP calls the failure **Excessive Agency (LLM06)** and decomposes it into three independent root causes, each with its own fix:

- **Excessive functionality** — the agent has tools it does not need. An email-summarizing agent that also holds `mail.send`, `mail.delete`, and a shell has three tools it will never legitimately use and an attacker will. Fix: give it exactly the tools the task requires, and no others.
- **Excessive permissions** — the tools it has are over-scoped. The database tool authenticated as `admin` when the task needs read-only on one table. Fix: per-tool credentials with the minimum scope; read-only by default.
- **Excessive autonomy** — it can take high-impact actions with no human in the loop. Fix: confirmation gates on irreversible actions (below).

The discipline is to assume the model *will* be hijacked and ask, tool by tool: *what is the worst a hijacked model can do with exactly these permissions?* Then trim until that answer is tolerable. This is **least privilege** applied to a component you must assume is adversarial.

## Least privilege and capability-based control

Concretely, a well-secured agent has:

- **Per-tool, minimally-scoped credentials.** Not one god-token; a distinct credential per tool, each read-only unless writing is essential, each scoped to the narrowest resource (one schema, one bucket prefix, one API method).
- **Allowlists on egress and file access.** The network tool can reach an explicit list of hosts, not the open internet; the file tool sees an explicit directory, not `/`. This directly attacks the "external comms" leg of the lethal trifecta.
- **Rate limits and budget caps on every tool and on token spend.** This is the fix for **Unbounded Consumption (LLM10)** — denial-of-wallet and denial-of-service — where an attacker (or a looping agent) drives runaway cost. A hard per-session budget and per-tool call cap turn "unbounded" into "bounded."
- **A kill switch.** You can halt the agent instantly, mid-trajectory.

The mental model is **capability security**: the agent can only do what it *holds a capability for*, and you grant capabilities narrowly. A guardrail says "this looks bad"; a capability says "this is impossible." For anything that must never happen, you want the capability, not the guardrail.

## Sandboxing

Any tool that executes model-influenced code or handles untrusted data runs in a **sandbox**: an isolated environment — a container or micro-VM — with no ambient credentials, no network egress except an explicit allowlist, an ephemeral filesystem, and resource limits. Two reasons. First, code the model writes (CodeAct-style agents, code interpreters) is model output and therefore untrusted (Lesson 05); it must not run with access to anything that matters. Second, loading untrusted model files or processing untrusted documents can itself trigger code execution (Lesson 04's pickle problem), and a sandbox contains the blast. The rule: *the environment that touches untrusted input holds nothing worth stealing and can reach nothing worth attacking.*

## Human-in-the-loop for irreversible actions

The cheapest strong control: **for any action that is irreversible or reaches outside the system, a human confirms before it runs.** The rule of thumb — *the model may plan; a human approves execution* — with risk tiering:

- **Low** (read-only, reversible) → auto-approve.
- **Medium** (write, recoverable) → batch approval.
- **High** (delete, send, purchase, deploy, move money) → **individual confirmation** showing exactly what will happen.

```python
IRREVERSIBLE = {"send_email", "delete_record", "make_payment", "deploy", "post_message"}

def execute_tool(call):
    if call.name in IRREVERSIBLE:
        preview = render_preview(call)        # resolved recipient, amount, target
        if not human_confirms(preview):        # blocking; agent cannot self-approve
            return ToolResult(status="denied_by_user")
    return dispatch(call)
```

Two rules make this actually work, and both are commonly gotten wrong. The preview must show the **resolved** arguments — the real recipient and body, not "an email" — so the human is approving reality, not a summary the model wrote. And the confirmation must be **out-of-band**: the model must not be able to synthesize its own approval, or an injection will simply instruct it to "confirm on the user's behalf." A confirmation the agent can produce is not a control.

## Isolation architectures — containing injection by design

Because prompt injection cannot be reliably detected (Lesson 02), the strongest defenses *architect it away* so untrusted content never reaches a privileged decision. The 2025 paper *Design Patterns for Securing LLM Agents against Prompt Injections* (ETH Zürich, Google DeepMind, IBM) catalogs six patterns; the essential ones, from least to most flexible:

- **Action-Selector.** The LLM only maps a request to one of a fixed set of pre-approved actions; untrusted output can never expand the action space. Maximum security, minimum flexibility.
- **Plan-Then-Execute.** The agent commits to its full plan *before* ingesting any untrusted data. A poisoned document read at step 3 can corrupt a *result* but cannot add a `send_email` step that was not in the plan decided at step 0. It cannot rewrite control flow.
- **LLM Map-Reduce.** Each untrusted item is processed in an isolated "map" call with no tools; a controlled "reduce" aggregates only sanitized outputs. Untrusted content never sits in a privileged context.
- **Dual-LLM pattern** (Willison, 2023). A **privileged LLM** holds the tools but never reads untrusted content directly; a **quarantined LLM** reads untrusted content but has no tools. The quarantined model returns opaque handles (`$summary_1`) that the privileged model routes ("display `$summary_1`") without ever seeing the potentially poisoned tokens.

**CaMeL** (Google DeepMind, April 2025) is the notable extension of the dual-LLM idea: it attaches **capability/provenance metadata** to every data value and enforces flow policies in a custom Python interpreter, so a value derived from untrusted content is *mechanically barred* from flowing into a sensitive sink. On the AgentDojo benchmark it neutralized roughly **67% of attacks** — meaningful, and a pointed reminder that even the strongest published architecture is not 100%. The trade-off across all of these is real: Action-Selector and Map-Reduce give strong security at the cost of flexibility; Plan-Then-Execute keeps more utility; Dual-LLM/CaMeL are strongest but costliest to build. Pick per the value at risk.

## MCP and the tool supply chain

Lesson 04 treated the *model file* as a supply-chain risk; the agent's *tools* are the other half. **MCP (Model Context Protocol)** is the standard way to expose tools to models — and it is a supply chain (OWASP LLM03) with distinctive failure modes, because an MCP server is untrusted code whose **tool descriptions are injected straight into the model's context**:

- **Tool poisoning.** Malicious instructions hidden in a tool's *description* or its returned output. The model reads the description to decide when to call the tool, so a poisoned description is a prompt injection delivered through the tool catalog. A 2025 study of ~1,899 open-source MCP servers found roughly 5.5% exhibited tool-poisoning vulnerabilities.
- **Rug pull.** A tool that is benign when you approve it *mutates its definition later*. `MCPoison` (**CVE-2025-54136**, Check Point) demonstrated this against Cursor — the client trusted the approved *name*, not the current *content*, so a swapped payload ran silently on every project open. The first malicious MCP package in the wild appeared September 2025.
- **Confused deputy.** The classic authorization flaw, sharp in MCP: the agent, acting with the user's legitimate credentials, is tricked into misusing them. Invariant Labs showed a crafted GitHub issue hijacking an assistant into exfiltrating private-repo data through a public PR. (See also CVE-2025-49596 in Anthropic's MCP Inspector.)

Defenses: **pin and hash** tool definitions and re-approve on any change (defeats rug pulls); run MCP servers **sandboxed** with least-privilege credentials; install only from trusted sources; and treat all tool descriptions and outputs as **untrusted content** subject to the isolation patterns above — never as trusted system text. When agents talk to *each other* (A2A, multi-agent systems), the same logic extends one level up: another agent's output is untrusted content, and a compromised sub-agent is a confused deputy with your credentials.

## Putting it together: the lethal trifecta as the design question

Recall the lethal trifecta from Lesson 02 — private data access, untrusted content exposure, external comms. Securing an agent is largely the discipline of **never granting all three legs in one session**, and when the task genuinely needs all three, inserting an isolation architecture between the untrusted leg and the privileged legs. Least privilege breaks legs; sandboxing contains the untrusted leg; human gates guard the external leg; dual-LLM/CaMeL let you keep all three legs but sever the *flow* between them. Every control in this lesson is, viewed correctly, a way of denying the trifecta.

## Key takeaways

- **Excessive Agency (LLM06)** has three root causes — excessive **functionality, permissions, and autonomy** — each with its own fix. Bounding capability caps the blast radius regardless of whether an attack succeeds.
- Apply **least privilege / capability security**: per-tool minimally-scoped credentials, egress and file allowlists, rate limits and budget caps (which also fix **Unbounded Consumption, LLM10**), and a kill switch. A capability makes bad actions *impossible*; a guardrail only makes them *less likely*.
- **Sandbox** anything that runs model-written code or touches untrusted data; the environment that meets untrusted input must hold nothing worth stealing.
- **Human-in-the-loop** on irreversible/outbound actions, with **resolved-argument previews** and **out-of-band** confirmation the model cannot self-issue.
- **Isolation architectures** contain injection by design — Action-Selector, Plan-Then-Execute, Map-Reduce, Dual-LLM, and **CaMeL** (~67% attack neutralization on AgentDojo, and still not 100%). **MCP** adds tool poisoning, rug pulls (CVE-2025-54136), and confused-deputy risks; pin+hash definitions, sandbox servers, and treat all tool text as untrusted. Every control is a way to deny the **lethal trifecta**.

## Try it

Build the trifecta on purpose, then dismantle the attack one control at a time. Give a small ReAct agent three tools: one that reads a "private" local file, one that fetches a URL you control, and one that makes an outbound HTTP request. Plant an indirect injection in the fetched page — an HTML comment telling the agent to read the private file and append its contents to the outbound URL — and watch it exfiltrate. Now add defenses and re-run the *same* attack after each: (1) a Llama Guard pass on tool outputs (Lesson 05); (2) **least privilege** — remove the private-file tool from the session that touches untrusted content; (3) a **human-confirmation gate** on the outbound request; (4) **plan-then-execute** so the fetch cannot inject a new step. You will observe directly that the content filter *reduces* the attack but the architectural controls — breaking a trifecta leg, gating the egress, freezing the plan — are what actually stop it. That result is the whole thesis of this course made concrete: capability, not detection, is what you rely on.
