# 02 — Data Preparation: The Part That Actually Decides Your Result

Fine-tuning quality is decided at the dataset, not the training loop. You can get every
hyperparameter right and still ship a worse model than you started with if the data is noisy,
inconsistently formatted, or subtly leaks the wrong behavior. This lesson is the hands-on guide to
building a dataset the trainer will accept and learn from: the formats TRL expects, how chat
templates turn structured messages into training text, why quality beats quantity, how to split
before you train, and where synthetic data helps and where it quietly poisons the run.

## The formats TRL actually accepts

Modern TRL (`SFTTrainer`) recognizes a small set of dataset shapes. Get your data into one of these
and the trainer handles tokenization and templating for you. The two you'll use most:

**Conversational (language modeling)** — a `messages` list per row. This is the format for chat and
instruction tuning:

```python
{"messages": [
    {"role": "system", "content": "You are a terse SQL assistant."},
    {"role": "user", "content": "Count users created today."},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM users WHERE created_at::date = CURRENT_DATE;"}
]}
```

**Prompt-completion** — separate `prompt` and `completion` fields. Use this when you want loss
computed on the completion only (the default), which is almost always what you want:

```python
{"prompt": [{"role": "user", "content": "Count users created today."}],
 "completion": [{"role": "assistant", "content": "SELECT COUNT(*) FROM users WHERE created_at::date = CURRENT_DATE;"}]}
```

There's also **standard** (`{"text": "..."}` — pre-formatted raw string) and its prompt-completion
variant. When you pass a conversational dataset, **the trainer automatically applies the model's
chat template** — you do not pre-render the string yourself. This is the biggest change from older
tutorials that manually built `text` fields; let the trainer do it so the formatting always matches
the model.

### ChatML and ShareGPT — the two you'll meet in the wild

- **ChatML** is the `role`/`content` message structure above — it's what OpenAI popularized and what
  Qwen, and most modern instruct models, use natively (`<|im_start|>role ... <|im_end|>`). TRL's
  `messages` format *is* ChatML-shaped. This is your target format.
- **ShareGPT** is an older community format from dumped ChatGPT conversations. Rows look like
  `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`. Many
  public datasets ship in ShareGPT. You'll need to convert `from`/`value` → `role`/`content` and map
  `human`→`user`, `gpt`→`assistant`:

```python
ROLE = {"human": "user", "gpt": "assistant", "system": "system"}

def sharegpt_to_messages(row):
    return {"messages": [
        {"role": ROLE[t["from"]], "content": t["value"]} for t in row["conversations"]
    ]}

dataset = dataset.map(sharegpt_to_messages, remove_columns=dataset.column_names)
```

Axolotl and other config-driven tools have built-in ShareGPT loaders; when you write your own
pipeline, this three-line map is the whole conversion.

## Chat templates: the mechanism, and why it's load-bearing

A **chat template** is a Jinja snippet stored on the tokenizer that turns a `messages` list into the
exact token string the model was trained on — role markers, turn delimiters, and (for reasoning
models) structural tags. It is not cosmetic. If you fine-tune with a different format than the model
serves with, you push it out of distribution and quality collapses.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

msgs = [{"role": "user", "content": "What is LoRA?"},
        {"role": "assistant", "content": "Low-rank adapters added to a frozen base."}]

# Training data: full turn, no generation prompt
print(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))

# Inference: stop after the assistant marker so the model completes it
print(tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True))
```

Two rules that save you hours of debugging:

- **Never hand-construct the special tokens.** Always go through `apply_chat_template`. The template
  is model-specific; typing `<|im_start|>` yourself will silently break on the next model.
- **`add_generation_prompt=True` only for inference.** It appends the assistant-role prefix so the
  model knows to start generating. In training data your assistant turn is already present, so use
  `False`.

**Fine-tuning a base (non-instruct) model?** It may have *no* chat template. TRL lets you attach one
via `SFTConfig(chat_template_path="...")` (point it at an instruct model whose template you want) or
`clone_chat_template()`. Align the EOS token too, or generations won't terminate.

## Loss only on the response — the detail that matters most

You do not want the model spending capacity learning to *generate the user's question*, only to
generate the answer given it. TRL handles this for you:

- With a **prompt-completion** dataset, loss is computed on the completion only by default
  (`completion_only_loss=True`).
- With a **conversational** dataset, set `assistant_only_loss=True` to mask everything but assistant
  turns (requires a template with `{% generation %}` markers; TRL auto-patches known families like
  Qwen3).

Under the hood this is the masked negative-log-likelihood from the alignment lesson: a
`response_mask` that's 0 over prompt tokens and 1 over response tokens, so the loss averages only
over positions you actually want the model to produce.

## Quality over quantity — the empirical result that should reshape your effort

The strongest, most repeated 2026 finding: **500 clean, consistent examples beat 5,000 noisy ones**
for most adaptation tasks. This is not a minor tuning tip; it inverts how you spend your time. Every
inconsistent label teaches the model to be inconsistent. A single systematic error (say, half your
examples use one JSON key casing and half use another) teaches the model to be unreliable at exactly
the thing you're fine-tuning it to be reliable at.

Practical quality bar for your dataset:

- **Consistency.** Same format, same schema, same voice across every example. Pick conventions and
  enforce them mechanically (validate every `completion` parses as JSON, matches the schema, uses the
  agreed casing).
- **Correctness.** Every demonstration is a *good* answer. The model imitates; garbage in, garbage
  out — literally.
- **Coverage.** Span the real input distribution, including the hard and edge cases. If production
  sees multi-turn, include multi-turn.
- **Deduplicate.** Near-duplicates inflate your count and your eval, and bias the model toward the
  over-represented pattern.
- **Length sanity.** Check the token-length distribution; a few pathologically long rows blow up
  memory and waste padding. Set `max_length` with the distribution in mind (Lesson 04).

## How much data, and where it comes from

The honest answer to "how many examples?" is: as few as it takes to be *clean and cover the
distribution*, which is usually less than people fear. Rough bands for a LoRA fine-tune:

- **~50–200** — enough for narrow style/format locking on a strong base, but you're near the floor;
  compare against few-shot prompting, which uses the same examples with no training risk.
- **~500–2,000** — the sweet spot for most narrow-task adaptation (extraction, classification, a
  specific voice). This is where quality curation pays off most.
- **~5,000–50,000+** — broader instruction tuning or larger behavioral shifts; here you can raise
  rank and epochs, but curation cost dominates and returns diminish.

Sources, in rough order of value: **real production data** (the gold standard — it *is* your
distribution), **hand-written demonstrations** (expensive, highest control), **converted public
datasets** (fast, but check fit and license), and **synthetic generation** (cheap volume, validate
hard). Most real projects blend all four: a core of hand-curated real examples, topped up with
filtered synthetic data for coverage of rare cases.

## Multi-turn data

If production is conversational, your training data must be too — a single-turn dataset teaches
single-turn behavior. A multi-turn row is just a longer `messages` list, and with
`assistant_only_loss=True` the loss lands on *every* assistant turn while user turns are masked:

```python
{"messages": [
    {"role": "system", "content": "You are a terse SQL assistant."},
    {"role": "user", "content": "How many users signed up today?"},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM users WHERE created_at::date = CURRENT_DATE;"},
    {"role": "user", "content": "Only the paying ones."},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM users WHERE created_at::date = CURRENT_DATE AND plan != 'free';"}
]}
```

The model learns to use prior turns as context. Keep multi-turn examples within `max_length` (Lesson
04) or the earliest turns get truncated away and you lose the context you were trying to teach.

## Split before you touch anything

Hold out an evaluation set *before* any augmentation, dedup-against-train, or synthetic generation —
and freeze it. Adding samples to a frozen eval later is fine; rebalancing or removing invalidates
every prior number. The cardinal sin is **eval contamination**: a paraphrase of a training example
leaking into eval, which makes the fine-tune look better than it is (more in Lesson 06).

```python
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]
# Dedup eval against train BEFORE freezing:
train_texts = {" ".join(m["content"] for m in r["messages"]) for r in train_ds}
eval_ds = eval_ds.filter(
    lambda r: " ".join(m["content"] for m in r["messages"]) not in train_texts
)
```

Split on the natural unit, not the row. If several rows come from one document, customer, or
conversation, keep that whole group on one side of the split — otherwise near-identical rows straddle
train and eval and you leak.

## Synthetic data: powerful, and a footgun

Generating training data from a stronger model (distillation) is now standard and often the fastest
way to a few thousand clean examples. It works. But the cautions are real:

- **You inherit the teacher's errors and style.** If the teacher hallucinates a field or has a verbose
  tic, your student learns it faithfully.
- **Model collapse on self-generated data.** Fine-tuning a model on its *own* outputs, repeatedly,
  degrades diversity and quality. Use a *stronger, different* model as the teacher, and always mix in
  real human examples.
- **Validate, don't trust.** Run every synthetic example through the same schema/correctness checks
  as human data — arguably stricter, since there's no human in the loop. Where you can check the
  answer (does the SQL run, does the JSON validate), filter to only-correct examples; that filtering
  is what turns cheap generation into a quality dataset.
- **Licensing.** Some frontier providers' terms restrict training competing models on their outputs.
  Check before you build a product on it.

## Key takeaways

- Target the **conversational (`messages`)** or **prompt-completion** format; let `SFTTrainer` apply
  the chat template automatically — don't pre-render strings.
- **ChatML** = the `role`/`content` structure modern models use; **ShareGPT** = older `from`/`value`
  dumps you convert with a three-line map.
- Chat templates are load-bearing: always use `apply_chat_template`, never hand-build special tokens,
  and use `add_generation_prompt=True` only at inference.
- Compute loss on the **response only** (`completion_only_loss` / `assistant_only_loss`) — masked NLL.
- **Quality beats quantity:** ~500 clean, consistent, correct, deduplicated examples outperform
  thousands of noisy ones. Data curation is the job.
- **Split (on the natural unit) and freeze the eval before any augmentation**; dedup eval against
  train to avoid contamination.
- Synthetic data is a strong accelerant but validate everything, use a stronger teacher, mix in real
  data, and check licensing.

## Try it

Build and validate a small SFT dataset for a task you care about. (1) Assemble ~200 examples in the
`messages` format — hand-written, converted from ShareGPT, or generated from a stronger model. (2)
Write a validator that runs on *every* row: assert the schema, that assistant content is non-empty,
and (if structured output) that it parses. Print how many rows fail — fix or drop them. (3) Render
three rows with `apply_chat_template(..., add_generation_prompt=False)` and read the raw string:
confirm the role markers and EOS token look right for your model. (4) Do a seeded 90/10 split, dedup
eval against train, and print the token-length distribution (`len(tok(text).input_ids)`) so you can
pick a sane `max_length` next lesson. You now have a training-ready, contamination-checked dataset —
the deliverable that decides everything downstream.
