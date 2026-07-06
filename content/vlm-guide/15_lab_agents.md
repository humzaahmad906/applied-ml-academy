# Lab 6 — Build a ReAct Agent with Tool Use

You implement a ReAct (Reason + Act) agent from scratch: a Qwen2.5-1.5B-Instruct model in a Thought→Action→Observation loop with three real tools, a registry/dispatch layer, a regex parser, and guards against common failure modes. Makes the Agents theory chapter concrete.

## Setup

```bash
pip install transformers accelerate torch
```

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (~3.1 GB bf16). CPU works at ~10 s/step; MPS or CUDA cuts that to 1–2 s. Each agent step is one full forward pass — budget 30–60 s on CPU for a 3-step question.

```python
import re, math, random, datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
)
model.eval()
```

## The ReAct format

ReAct (Yao et al., 2022) interleaves reasoning and acting. The model never sees the tool result until it requests it — each Observation grounds the next Thought in a real return value, not a hallucinated one.

```
User:        How tall is the Eiffel Tower in feet?
Assistant →  Thought: I need the height in meters, then convert.
             Action: doc_search(Eiffel Tower height)
You inject → Observation: The Eiffel Tower, Paris. Height: 330 m. Built 1889.
Assistant →  Thought: 330 × 3.28084 = 1082.68. Verify with calculator.
             Action: calculator(330 * 3.28084)
You inject → Observation: 1082.6772
Assistant →  Answer: The Eiffel Tower is approximately 1,082.7 feet tall.
```

## System prompt

Small models (1.5B) need unambiguous format instructions — any vagueness produces violations that break the parser.

```python
SYSTEM_PROMPT = """\
You reason step-by-step using tools. At each turn emit EXACTLY ONE block:

  Thought: <your reasoning about what to do>
  Action: tool_name(argument)
  Answer: <your final, complete answer>

Rules:
- A Thought MUST appear before every Action.
- After an Action you will receive an Observation — use it before continuing.
- When you have all information needed, emit Answer:.
- NEVER emit Observation: yourself — the system injects that.

Available tools:
  calculator(expression)  – evaluate any Python math expression (math module in scope)
  get_datetime()          – return the current local date and time
  doc_search(query)       – search a small local knowledge base
"""
```

## Tools

```python
def calculator(expression: str) -> str:
    # Strip builtins, expose only math.* — eval over user input is injection-prone.
    safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe["__builtins__"] = {}
    try:
        return str(eval(expression, safe))  # noqa: S307
    except Exception as exc:
        return f"Error: {exc}"


def get_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_KB: dict[str, str] = {
    "eiffel tower": "The Eiffel Tower, Paris. Height: 330 m (1,083 ft). Built 1889.",
    "great wall":   "The Great Wall of China stretches ~21,196 km (13,171 mi).",
    "transformer":  "Introduced in 'Attention Is All You Need' (Vaswani et al., 2017). Decoder-only variants dominate modern LLMs.",
    "python":       "Python: high-level, dynamically typed. First released 1991.",
}


def doc_search(query: str) -> str:
    q = query.lower()
    for key, val in _KB.items():
        if key in q or any(w in q for w in key.split()):
            return val
    return f"No entry found for: '{query}'"
```

## Tool registry and dispatch

```python
REGISTRY: dict[str, callable] = {
    "calculator": calculator,
    "get_datetime": get_datetime,
    "doc_search": doc_search,
}


def dispatch(name: str, args: list[str]) -> str:
    if name not in REGISTRY:
        return f"Unknown tool '{name}'. Available: {list(REGISTRY)}"
    try:
        return REGISTRY[name](*args)
    except TypeError as exc:
        return f"Wrong arguments for '{name}': {exc}"
    except Exception as exc:
        return f"Tool '{name}' raised: {exc}"
```

## Parsing model output

```python
_ACTION_RE = re.compile(r"Action:\s*(\w+)\(([^)]*)\)")
_ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.DOTALL)


def _parse_args(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [a.strip().strip("\"'") for a in raw.split(",")]


def parse_step(text: str) -> tuple[str, str | None, list[str] | None]:
    """Returns ("answer", text, None) | ("action", name, args) | ("none", None, None)."""
    m = _ANSWER_RE.search(text)
    if m:
        return "answer", m.group(1).strip(), None
    m = _ACTION_RE.search(text)
    if m:
        return "action", m.group(1), _parse_args(m.group(2))
    return "none", None, None
```

## Generation and the agent loop

```python
def _generate(messages: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,           # greedy — more stable for tool-use reasoning
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_agent(question: str, max_steps: int = 8, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    for step in range(1, max_steps + 1):
        response = _generate(messages)
        if verbose:
            print(f"\n── step {step} ──────────────────────\n{response}")

        kind, payload, args = parse_step(response)

        if kind == "answer":
            return payload

        if kind == "action":
            obs = dispatch(payload, args or [])
            if verbose:
                print(f"Observation: {obs}")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user",      "content": f"Observation: {obs}"})
            continue

        # Stalled — no parseable step; nudge toward completing it
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user",
                         "content": "Continue. Your next output must start with Action: or Answer:."})

    return "Agent did not reach a conclusion within the step budget."


if __name__ == "__main__":
    for q in ["What is 2**10 + sqrt(144)?", "What time is it?",
              "How tall is the Eiffel Tower in feet? (1 m = 3.28084 ft)"]:
        print(f"\n{'='*50}\nQ: {q}\n{'='*50}")
        print(f">>> {run_agent(q)}\n")
```

## Failure modes and guards

- **Infinite loops.** `max_steps` is the primary guard. Log step count per question — systematic overrun means the system prompt needs sharpening.
- **Hallucinated tools.** Model invents `web_search`. Registry check returns an error observation naming the real tools; most instruction-tuned models self-correct on the next step.
- **Bad argument parses.** `_parse_args` splits on `,` naively. `calculator(max(1,2))` becomes `["max(1", "2)"]`. Use JSON-structured calling (§ Stacks) to eliminate this class of bug in production.
- **Model emits `Observation:` itself.** Forbidden by the system prompt. If it happens, strip from `"Observation:"` onward before parsing, or add `"\nObservation"` as a stop sequence in `model.generate`.

## Stacks & alternatives

### LangGraph — stateful graph with typed state

Reach for LangGraph when you need branching logic, parallel tool calls, human-in-the-loop interrupts, or durable checkpointing. The while-loop above becomes a typed graph with explicit nodes and conditional edges.

```python
# pip install langgraph langchain-huggingface
from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# model_node: {"messages": [llm_with_tools.invoke(state["messages"])]}
# tool_node:  for each c in last.tool_calls → ToolMessage(tools_by_name[c["name"]].invoke(c["args"]))
graph = StateGraph(AgentState)
graph.add_node("model", model_node); graph.add_node("tools", tool_node)
graph.set_entry_point("model")
graph.add_conditional_edges("model", lambda s: "tools" if s["messages"][-1].tool_calls else END)
graph.add_edge("tools", "model")
app = graph.compile()  # app.invoke({"messages": [HumanMessage(content=question)]})
```

Tradeoff: richer orchestration + checkpointing, opinionated message schema, more setup. The from-scratch loop is more portable and exposes exactly what frameworks hide.

### Native/structured tool-calling via JSON schemas

Qwen2.5-Instruct supports OpenAI-style tool schemas in its chat template; the model emits structured JSON call objects. Zero regex parsing.

```python
tools = [{"type": "function", "function": {
    "name": "calculator",
    "description": "Evaluate a Python math expression",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]},
}}]
# apply_chat_template encodes tool schemas; model emits <tool_call>{...}</tool_call>
text = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=True
)
# Parse json.loads on the tool_call block; feed result back as a tool-role message
```

Tradeoff: clean, production-grade, zero regex. Requires a model post-trained for tool-calling; also available natively via Ollama's tool API.

### MCP — Model Context Protocol

MCP decouples tool providers from agent builders. Expose tools once as an MCP server; any MCP-compatible client (Claude Desktop, a custom agent, an IDE) discovers and calls them without knowing the implementation. Donated to the Linux Foundation (2025), 10k+ servers in production.

```python
# pip install mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-tools")
@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a Python math expression safely."""
    safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe["__builtins__"] = {}
    return str(eval(expression, safe))

mcp.run()   # stdio or SSE — any MCP client connects automatically
```

Tradeoff: overkill for a single-agent codebase; right default when tools are shared across agents, teams, or client surfaces. See the Agents chapter for MCP architecture and its multi-agent complement A2A.

## What you built

- A complete ReAct loop (Thought→Action→Observation→repeat) grounded in real tool dispatch, not hallucinated returns.
- Three real tools plus a registry/dispatch layer that guards against hallucinated names and argument errors.
- A system prompt that reliably teaches the Thought/Action/Answer format to a 1.5B-parameter model.
- A regex parser distinguishing actions, answers, and stalled steps with a handler for each case, plus `max_steps` guard and failure-mode defenses.

## Build it further

Add a `python_repl(code: str) -> str` tool that executes arbitrary Python in a subprocess with a 5-second timeout (`subprocess.run`, `capture_output=True`, `timeout=5`). Benchmark 20 arithmetic questions from GSM8K: record step count, answer accuracy, and timeout rate with and without the `calculator` tool. Deliver a table — `tool_config | accuracy | avg_steps | timeout_pct` — and one sentence on why tool use affects accuracy more than step count.
