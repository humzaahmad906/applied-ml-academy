# 06 — Advanced Topics: Everything Else Worth Knowing — Part 2 of 5: Alignment, Agentic Systems, and Retrieval Internals

This is part 2 of the Advanced Topics reference catalog. Here we cover RL/RLHF/DPO and modern alignment, agentic systems and their observability, and vector search / retrieval internals.

## Phase 5 — RL, RLHF, DPO, and Modern Alignment

For LLM training and recommendation/decision systems, RL is increasingly part of the MLOps stack.

### Classical RL — The Conceptual Backbone

- **Agent, environment, action, state, reward** — the basic loop
- **Policy** — the function that maps state to action distribution
- **Value functions** — V (state value), Q (state-action value)
- **On-policy vs off-policy** — does the training data come from the current policy?
- **PPO** — proximal policy optimization; the workhorse algorithm
- **DDPG, SAC, TD3** — actor-critic methods for continuous actions

You don't need to implement these from scratch. You need to recognize them and know when each is used.

### RLHF Specifically

For aligning LLMs:

1. **SFT (Supervised Fine-Tuning)** — train on instruction → response pairs
2. **Reward modeling** — train a model to predict human preferences
3. **PPO** — fine-tune the LLM to maximize predicted reward, with a KL constraint to the reference model

PPO is finicky. Reward hacking is real. Most modern alignment has moved past it.

### DPO (Direct Preference Optimization)

Skip the reward model. Given pairs (prompt, chosen, rejected), train the model to prefer chosen directly:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left( \beta \log \frac{\pi(\text{chosen} \mid \text{prompt})}{\pi_{\text{ref}}(\text{chosen} \mid \text{prompt})} - \beta \log \frac{\pi(\text{rejected} \mid \text{prompt})}{\pi_{\text{ref}}(\text{rejected} \mid \text{prompt})} \right)
$$

No reward model, no PPO. Much more stable. Often equivalent results. The 2026 default.

### Successors: ORPO, KTO, IPO, SimPO, GRPO

- **ORPO** — combines SFT and DPO in one loss. Simpler pipeline.
- **KTO** — uses single-rating (not pairs) preference data.
- **IPO** — robust to preference noise.
- **SimPO** — reference-model-free DPO variant.
- **GRPO** (Group Relative Policy Optimization) — used in DeepSeek-R1 and successors. No critic, no reference model in some variants; uses group statistics for advantage.

The field moves quickly. Internalize SFT + DPO + one PPO-variant; track the rest in the literature.

### RL for Other ML Systems

- **Contextual bandits** for recommendations — A/B testing's smarter cousin
- **Off-policy evaluation** — given logged data from policy A, estimate policy B's performance without deploying it
- **RL in pricing, inventory, routing** — when the action affects the next state and the reward, RL beats supervised learning

### Exercises

1. Implement a simple DPO loop on a small preference dataset.
2. Read the DPO paper and the GRPO/DeepSeek-R1 papers.
3. Experiment with TRL (HuggingFace's RL library) — SFT then DPO on a 1B model on a small task.

---

## Phase 6 — Agentic Systems

Multi-step LLM applications with tool use. Increasingly the F50 frontier.

### What Makes a System "Agentic"

- **Multi-step:** the model decides what to do next based on intermediate results
- **Tool use:** the model can call external functions / APIs / databases
- **Memory:** state persists across turns or even sessions
- **Self-correction:** the model can detect and recover from its own errors

### The Patterns

- **ReAct** — interleave reasoning ("I should look up X") and acting ("call tool X"). The classic.
- **Plan-and-execute** — generate a plan, then execute steps. Better for long horizons.
- **Tool-calling APIs** — OpenAI / Anthropic function-calling; the modern standard.
- **Code execution as the tool** — let the LLM write Python; execute it in a sandbox; feed back the output. Very general, very powerful, very risky.

### Frameworks

- **LangChain / LangGraph** — most mature; criticized for over-abstraction.
- **LlamaIndex** — strong for RAG-flavored agents.
- **Haystack** — search-flavored.
- **Autogen** (Microsoft) — multi-agent collaboration.
- **CrewAI** — role-based multi-agent.
- **Custom Python** — most production agents at frontier labs are bespoke.

For F50 portfolio work, **build at least one agent from scratch in plain Python** before reaching for a framework. Frameworks hide the design space you need to understand.

### Operational Challenges

- **Cost** — agents can spin in loops; bound the iteration count, the token budget per session
- **Latency** — many round trips; user-facing agents need streaming + intermediate status
- **Safety** — tool calls with side effects (sending email, modifying databases) need confirmation, dry-run modes, allowlists
- **Observability** — log every step; visualize the trace. Langfuse, Weave, LangSmith are designed for this.
- **Evaluation** — much harder than single-turn; end-to-end success rate, per-step quality, recovery rate from errors

### Agent Observability and AgentOps

Agent observability is qualitatively different from LLM observability. A single LLM call has one prompt and one response. An agent has a multi-step trace, a tool-call tree, branches that were considered and rejected, retries on failure, and potentially parallel sub-agents. Standard LLM tracing tools show you a flat list of spans; agent debugging requires a tree.

**What to capture per agent step:**

- Prompt version (which system prompt, which few-shot examples)
- Tool name, arguments passed, result returned
- Step latency (time to decision + time in tool)
- Token cost for this step (accumulating across the session)
- Decision rationale (the model's "thought" in ReAct; the plan step in plan-and-execute)
- Whether this step was a retry and why the previous attempt failed

**Tools:**

- **AgentOps** (MIT-licensed, 400+ LLM integrations) — wraps agent frameworks and provides a session replay UI. Time-travel debugging: step through the agent's execution, see what the model saw at each decision point, replay from any intermediate state. This is the capability that matters most when debugging an agent that took a wrong branch 12 steps into a 30-step run.
- **Braintrust** — 1M free spans per month; first-class support for multi-step traces with nested tool calls; eval integration means you can catch the same failure in CI before it reaches production.
- **Langfuse** — open-source, self-hostable; session-level traces with parent/child span hierarchy; good for teams that can't send data to a managed service.
- **OTel GenAI semconvs** — the underlying standard. LangChain, CrewAI, and AutoGen emit OTel-compliant agent spans; the `gen_ai.tool.name` and `gen_ai.tool.call.id` attributes on child spans give you the tree structure that maps to the agent's actual decision tree. Covered in the LLMOps and monitoring sections of this course.

**The distinct failure modes to instrument for:**

- **Loop detection** — agent revisits the same tool call with the same arguments N times; budget exceeded without progress
- **Tool error propagation** — a tool returns an error; does the agent recover, retry with different args, or hallucinate a result?
- **Context window pressure** — as the conversation grows, earlier context gets truncated; the agent "forgets" earlier reasoning
- **Latency per step** — a single slow tool call cascades through every downstream step; per-step latency attribution is essential

The ops rule: any agent that touches a production system (sends email, writes to a database, calls an external API) must have a **dry-run mode** that logs all intended tool calls without executing them. Run dry-run on every new prompt version before enabling live execution.

### Exercises

1. Build a 3-tool agent in plain Python (calculator, web search, file read). Implement ReAct manually.
2. Add tracing — every tool call logged with inputs, outputs, latency.
3. Build an eval harness with 30+ test scenarios. Measure success rate.
4. Instrument the agent with AgentOps or Langfuse. Replay a failed session. Identify the exact step where the agent went wrong.
5. Inject a tool that intermittently fails (returns an error 30% of the time). Observe how the agent handles it. Add detection for loop behavior.

---

## Phase 7 — Vector Search and Retrieval Internals

### ANN Algorithms in Depth

**HNSW** (Hierarchical Navigable Small World):

- Build a multi-layer graph; each layer is a "small-world" navigable graph
- Search top-down: at each layer, navigate to the nearest neighbor; descend
- Parameters: M (graph degree), efConstruction (build effort), efSearch (search effort)
- Trade-off: higher M / ef → better recall, more memory, slower

**IVF** (Inverted File Index):

- Cluster the vectors (k-means). Vectors stored by cluster.
- Search: find nearest centroids; scan only those clusters.
- Parameter: nlist (clusters), nprobe (clusters scanned at query). Higher nprobe → better recall, slower.

**PQ** (Product Quantization):

- Split each vector into subvectors; quantize each independently with a small codebook
- Massive memory savings (16x typical); some recall loss
- Often combined: **IVF-PQ** for billion-scale on cheap hardware

**DiskANN** — disk-backed; great for billion-scale at modest cost.

### Hybrid Search Math

$$
\text{score} = \alpha \cdot \text{normalize}(\text{vector\_score}) + (1 - \alpha) \cdot \text{normalize}(\text{bm25\_score})
$$

How to pick α? Use a labeled eval set. Tune α to maximize NDCG@10 or recall@10. Often α=0.5–0.7 works.

**Reciprocal Rank Fusion (RRF)** — alternative score combination that doesn't require score normalization. Robust default.

### Reranking

A cross-encoder (BERT-class model, fine-tuned for relevance) scores each candidate against the query. Way slower per pair but applied only to top 50–200 candidates from first-pass retrieval. Big quality lift.

### Embedding Model Choice

In 2026:

- **OpenAI text-embedding-3-large** — strong, expensive
- **Cohere embed-english-v3** — strong, expensive, multilingual
- **BGE-large-v1.5**, **E5-large-v2** — strong open-source baselines
- **Domain-specific** — fine-tune on your data with contrastive learning (in-batch negatives, hard negative mining)

Dimensionality matters: 1536-dim has more capacity but higher storage; 256-dim or 384-dim is often sufficient and 4–6x cheaper.

### Exercises

1. Build an HNSW index from scratch (or use `hnswlib`). Tune M / ef. Measure recall vs latency.
2. Build IVF-PQ index. Compare to HNSW on a 1M-vector corpus.
3. Fine-tune an embedding model on your domain. Use contrastive loss with in-batch negatives.

---

## You can now

- Choose between RLHF's reward-model + PPO pipeline and DPO/its successors (ORPO, KTO, IPO, SimPO, GRPO) for aligning an LLM, and know when classical RL or contextual bandits fit a non-LLM decision system better.
- Diagnose an agentic system through a tool-call trace tree — loop detection, tool-error propagation, context-window pressure — and enforce dry-run mode on side-effecting tools.
- Pick the right ANN algorithm (HNSW, IVF, PQ, DiskANN) and hybrid-search fusion strategy for a retrieval workload, and know when a reranker or a fine-tuned embedding model is worth the added latency.
