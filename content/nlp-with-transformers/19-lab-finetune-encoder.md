# 19 — Lab 3: Fine-Tune an Encoder (the production workhorse)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/nlp-with-transformers/notebooks/19-lab-finetune-encoder.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup.

Most NLP shipping in production is not a chat model. It is a 100M-parameter encoder fine-tuned on a few thousand labeled examples, serving classification or extraction at single-digit-millisecond latency inside a private VPC. This lab builds two of those and then does the arithmetic that decides whether you deploy one or call an API: (A) a DistilBERT text classifier with a confusion matrix and a calibration check, (B) a DistilBERT NER tagger with explicit subword label alignment and entity-level `seqeval` scoring, and (C) a latency-and-cost table that tells you when the encoder wins. Grounds the [transfer-learning chapter](06-transfer-learning-tasks.md).

## Setup

```bash
pip install -q -U transformers datasets accelerate seqeval scikit-learn
```

Set Runtime → Change runtime type → **T4 GPU**. Everything is seeded to 42. Both fine-tunes together take **8–12 min on a T4** (a few minutes each) and peak under ~3 GB of GPU memory — `distilbert-base-uncased` is 66M parameters. The whole notebook finishes well under 25 minutes.

```python
import random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  torch={torch.__version__}")
```

---

## Part A — Fine-Tune a Classifier

Task: AG News topic classification — 4 classes (World, Sports, Business, Sci/Tech). We cap the training set to 2,000 examples so it trains in a few minutes; a real run uses the full 120k. The point is the *workflow*, which is identical at any scale: dataset → tokenize → `Trainer` → metrics.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "distilbert-base-uncased"
raw = load_dataset("ag_news")
LABELS = raw["train"].features["label"].names   # ['World','Sports','Business','Sci/Tech']
id2label = dict(enumerate(LABELS)); label2id = {v: k for k, v in id2label.items()}

train_ds = raw["train"].shuffle(seed=SEED).select(range(2000))
test_ds  = raw["test"].shuffle(seed=SEED).select(range(2000))

tok = AutoTokenizer.from_pretrained(MODEL)
def encode(batch):
    return tok(batch["text"], truncation=True, max_length=128)

train_enc = train_ds.map(encode, batched=True).rename_column("label", "labels")
test_enc  = test_ds.map(encode,  batched=True).rename_column("label", "labels")
print(f"train={len(train_enc)}  test={len(test_enc)}  classes={len(LABELS)}")
```

`truncation=True, max_length=128` caps sequence length — AG News snippets are short, so almost nothing is cut, and shorter sequences train faster. We rename `label`→`labels` because that is the argument name the model's `forward` expects; `Trainer` drops any column that is not a forward argument.

```python
from transformers import (AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id)

args = TrainingArguments(
    output_dir="clf", num_train_epochs=3,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    learning_rate=3e-5, seed=SEED, fp16=torch.cuda.is_available(),
    logging_steps=25, report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=train_enc,
                  data_collator=DataCollatorWithPadding(tok), tokenizer=tok)
trainer.train()
```

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

pred_out = trainer.predict(test_enc)
logits = pred_out.predictions
y_pred = logits.argmax(-1)
y_true = pred_out.label_ids

acc = accuracy_score(y_true, y_pred)
f1  = f1_score(y_true, y_pred, average="macro")
print(f"accuracy={acc:.3f}  macro-F1={f1:.3f}\n")
print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))
```

Expect ~0.90 accuracy from only 2,000 examples — encoders are extremely sample-efficient on well-separated classification tasks because the pretrained representations already cluster topics; fine-tuning just learns a linear head plus a light nudge to the body. Report **macro-F1** alongside accuracy: it weights every class equally, so a model that quietly fails on the rarest class cannot hide behind a high average.

### Confusion matrix

Accuracy tells you *how often*; the confusion matrix tells you *which mistakes*. That is what you take to a product review.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, cm[i, j], ha="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
plt.title("AG News confusion matrix"); plt.tight_layout(); plt.show()
```

The heavy off-diagonal cells are almost always Business ↔ Sci/Tech — genuinely overlapping topics. That is a labeling-boundary problem, not a model bug.

### Calibration: are the confident wrong answers?

A model's softmax max is its confidence. For deployment you need to know whether that number means anything — can you route low-confidence predictions to a human? Plot the confidence distribution of the *wrong* predictions.

```python
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
conf  = probs.max(-1)
wrong = y_pred != y_true

plt.figure(figsize=(6, 3.5))
plt.hist(conf[wrong], bins=20, range=(0, 1), color="#c0392b", alpha=0.8)
plt.xlabel("model confidence on WRONG predictions"); plt.ylabel("count")
plt.title(f"{wrong.sum()} errors — where does the model go wrong?")
plt.tight_layout(); plt.show()

# Expected Calibration Error (ECE), 10 bins
def ece(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e
print(f"ECE = {ece(conf, (~wrong).astype(float)):.3f}")
```

If errors cluster at low confidence, a threshold ("send anything under 0.8 to review") recovers the precision you need. If a fat tail of errors sits at 0.95+, the model is **overconfident** — a well-documented behavior of cross-entropy-trained classifiers — and raw softmax is unsafe as a routing signal until you temperature-scale it on a held-out set. ECE quantifies the confidence-vs-accuracy gap in one number; above ~0.05 is worth a calibration pass.

---

## Part B — Token Classification: NER on CoNLL-2003

Named-entity recognition is per-*token* classification: every token gets a BIO tag (`B-PER`, `I-ORG`, `O`, …). The one hard part — the part interviewers probe — is that the tokenizer splits words into subwords, but the labels are per-word. You must align them, and mask the pieces you do not supervise with `-100`.

```python
ner_raw = load_dataset("conll2003", trust_remote_code=True)  # HF mirror; script-based
NER_LABELS = ner_raw["train"].features["ner_tags"].feature.names
ner_id2label = dict(enumerate(NER_LABELS)); ner_label2id = {v: k for k, v in ner_id2label.items()}
print(NER_LABELS)   # ['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC','B-MISC','I-MISC']

ner_train = ner_raw["train"].shuffle(seed=SEED).select(range(3000))
ner_eval  = ner_raw["validation"].select(range(800))
ner_tok   = AutoTokenizer.from_pretrained(MODEL)
```

### Subword label alignment (the `-100` masking)

This is the whole trick. Words are already split (`is_split_into_words=True`), so we ask the tokenizer for `word_ids()` — a map from each subword back to its source word. We give the **first** subword of each word its label and set every continuation subword, plus every special token, to `-100`. PyTorch's cross-entropy ignores `-100`, so those positions contribute no loss and no gradient.

```python
def align_labels(examples):
    enc = ner_tok(examples["tokens"], truncation=True, max_length=128,
                  is_split_into_words=True)
    all_labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = enc.word_ids(batch_index=i)
        prev, label_ids = None, []
        for wid in word_ids:
            if wid is None:                 # [CLS], [SEP], padding
                label_ids.append(-100)
            elif wid != prev:               # first subword of a new word → keep label
                label_ids.append(word_labels[wid])
            else:                           # continuation subword → mask
                label_ids.append(-100)
            prev = wid
        all_labels.append(label_ids)
    enc["labels"] = all_labels
    return enc

ner_train_enc = ner_train.map(align_labels, batched=True, remove_columns=ner_train.column_names)
ner_eval_enc  = ner_eval.map(align_labels,  batched=True, remove_columns=ner_eval.column_names)

# sanity check: continuation subwords must be -100
ex = ner_train_enc[0]
pairs = list(zip(ner_tok.convert_ids_to_tokens(ex["input_ids"]), ex["labels"]))
print(pairs[:12])
```

You will see rows like `('##ing', -100)` — a masked continuation — next to `('germany', 5)` labeled `B-LOC`. Getting this wrong is the classic silent NER bug: label the *last* subword instead of the first, or forget to mask continuations, and your entity-level F1 quietly drops several points while token accuracy still looks fine.

```python
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

ner_model = AutoModelForTokenClassification.from_pretrained(
    MODEL, num_labels=len(NER_LABELS), id2label=ner_id2label, label2id=ner_label2id)

ner_args = TrainingArguments(
    output_dir="ner", num_train_epochs=3,
    per_device_train_batch_size=16, per_device_eval_batch_size=64,
    learning_rate=3e-5, seed=SEED, fp16=torch.cuda.is_available(),
    logging_steps=50, report_to="none")

ner_trainer = Trainer(model=ner_model, args=ner_args, train_dataset=ner_train_enc,
                      data_collator=DataCollatorForTokenClassification(ner_tok), tokenizer=ner_tok)
ner_trainer.train()
```

### Entity-level evaluation with seqeval

Token accuracy lies about NER: predicting `O` everywhere scores ~85% because most tokens are `O`. What matters is whether you got whole *entities* right — correct type and correct span boundaries. `seqeval` does exactly that: it reconstructs spans from BIO tags and scores them, which is the number every NER paper and every interviewer means by "F1".

```python
from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1

out = ner_trainer.predict(ner_eval_enc)
preds = out.predictions.argmax(-1)
labels = out.label_ids

true_tags, pred_tags = [], []
for p_row, l_row in zip(preds, labels):
    t, p = [], []
    for p_i, l_i in zip(p_row, l_row):
        if l_i != -100:                          # skip masked positions
            t.append(ner_id2label[l_i]); p.append(ner_id2label[p_i])
    true_tags.append(t); pred_tags.append(p)

print(f"entity-level F1 = {seq_f1(true_tags, pred_tags):.3f}\n")
print(seq_report(true_tags, pred_tags, digits=3))
```

Expect entity-level F1 around 0.85–0.90 on this subset (the full CoNLL-2003 training set pushes DistilBERT past 0.90). Note in the per-type report that `PER` and `LOC` score high while `MISC` lags — `MISC` is a grab-bag class with fuzzy boundaries, the NER analog of the Business/Sci/Tech confusion in Part A.

---

## Part C — The Economics: Encoder vs API LLM

You have two models that do a real job. Now the question a staff engineer actually asks: do we self-host this encoder or just call an API LLM with a prompt? Measure, then decide.

```python
sample = test_ds[0]["text"]
enc_in = tok(sample, return_tensors="pt", truncation=True, max_length=128)

def latency_ms(model, inputs, dev, n=50):
    model = model.to(dev).eval()
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        for _ in range(5): model(**inputs)          # warmup
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n): model(**inputs)
        if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

gpu_ms = latency_ms(model, enc_in, "cuda") if torch.cuda.is_available() else float("nan")
cpu_ms = latency_ms(model, enc_in, "cpu")
print(f"single-example latency  —  GPU: {gpu_ms:.1f} ms   CPU: {cpu_ms:.1f} ms")
```

Single-example latency on a T4 is typically **3–8 ms** (GPU) and **20–60 ms** (CPU) for DistilBERT. Now put it next to an API call for the same 4-class task. A classification prompt costs roughly 200 input tokens plus a few output tokens; at a small hosted model's price of ~\$0.15 / 1M input tokens, 1M requests ≈ \$30. The self-hosted encoder on a ~\$0.35/hr T4 processing hundreds of examples per second costs on the order of \$1 for the same 1M requests.

| dimension | fine-tuned DistilBERT (self-host) | API LLM (prompted, small model) |
|---|---|---|
| latency / request | ~3–8 ms (GPU), ~20–60 ms (CPU) | ~300–800 ms (network + generation) |
| cost / 1M requests | ~\$1 (T4 compute) | ~\$30+ (tokens in + out) |
| throughput | thousands/sec batched on one GPU | rate-limited by the provider |
| data privacy | stays in your VPC | leaves to a third party |
| labeled data needed | ~1–5k examples | zero (prompt only) |
| task flexibility | one fixed task per head | any task, instantly, by editing the prompt |
| maintenance | you own training + serving | provider owns it |

**When the encoder wins:** high, steady request volume; a *fixed* task with a stable label set; tight latency budgets (search ranking, live moderation, ticket routing); regulated or private data that cannot leave your network; and cost sensitivity at scale — at millions of requests the ~30x gap is the whole infra bill. **When the API wins:** low or spiky volume where a GPU sits idle; tasks that change often or need reasoning/generation, not a fixed label; and the cold-start phase before you have labeled data. The mature answer is often both — prototype and collect labels with the API, then distill the winning behavior into an encoder once volume and the label schema stabilize.

---

## What you built

- A DistilBERT topic classifier fine-tuned with `Trainer` on 2,000 examples, scored with accuracy and macro-F1, a confusion matrix, and an ECE/confidence-histogram calibration check.
- A DistilBERT NER tagger with **explicit subword→label alignment and `-100` masking**, evaluated with entity-level `seqeval` F1 and a per-type breakdown, plus a measured GPU/CPU latency benchmark and a cost/latency/privacy table naming when a small encoder beats an API LLM.

## Exercises

1. **Temperature scaling.** Fit a single scalar temperature $T$ on the test logits (minimize NLL by grid search over $T \in [0.5, 3]$), divide logits by $T$ before softmax, and recompute ECE. How much does calibration improve, and does accuracy change?
2. **Swap the backbone.** Rerun Part A with `distilroberta-base` and with `google/bert_uncased_L-4_H-256_A-4` (a tiny BERT). Plot accuracy vs single-example latency for the three models — where is the knee of the curve?
3. **Break the alignment on purpose.** In `align_labels`, label the *last* subword of each word instead of the first (and unmask continuations). Retrain and report the seqeval F1 drop. Explain why token accuracy barely moves while entity F1 falls.
4. **Confidence-thresholded routing.** Using the Part A calibration, pick a confidence threshold that sends the lowest-confidence 10% of predictions to "human review." Report the accuracy on the auto-handled 90% and the number of errors you caught.
5. **Fill in the cost table with real numbers.** Take current published per-token prices for one hosted small model, estimate your true prompt length, and recompute the \$/1M figure. At what monthly request volume does self-hosting a T4 break even?

## What interviews ask here

- Why do you align NER labels to the *first* subword and mask the rest with `-100`, and what breaks if you don't? (cross-entropy ignore_index; entity-F1 vs token-accuracy divergence)
- Why report macro-F1 and a confusion matrix instead of just accuracy? (class imbalance; which mistakes vs how many; product-review artifact)
- Why is `seqeval` entity-level F1 the right NER metric and token accuracy the wrong one? (span reconstruction; the all-`O` baseline scores ~85%)
- Your classifier is 95% confident on wrong answers — what does that mean and how do you fix it? (overconfidence of cross-entropy softmax; temperature scaling on a held-out set; ECE)
- When do you fine-tune a 100M encoder instead of prompting an API LLM? (volume, fixed task, latency, privacy, cost-at-scale; distill-after-prototyping)
- Why is dynamic padding (`DataCollatorWithPadding`) faster than padding to a fixed length? (fewer wasted FLOPs on pad tokens; batch padded to its own max)
