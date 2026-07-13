# 06 — Transfer Learning: The Applied-NLP Workhorse

Most NLP in production is not a chatbot. It is a classifier deciding which of 40
queues a support ticket belongs to, a tagger pulling company names out of
filings, a retriever ranking documents, a model that has to answer in under 30
milliseconds on a CPU for a tenth of a cent. All of that runs on the same durable
pattern: take a model pretrained with the objectives from
[05-pretraining.md](05-pretraining.md), add a small task-specific head, fine-tune
on a few thousand labeled examples. This chapter is that workflow end to end —
classification, NER, extractive QA, seq2seq generation and its metrics, sentence
embeddings, and the economics of when a 100M-parameter encoder beats an API LLM.
It is the most immediately employable chapter in the course.

## Why fine-tune an encoder at all

The reflex in 2026 is to prompt a large API model for everything. For a narrow,
high-volume task that reflex is often wrong. Pretraining already taught a
DeBERTa-v3 encoder what language means; you're only teaching it a decision
boundary. A few thousand labels and ten minutes on one GPU produce a 100M-param
model that runs on cheap hardware, has predictable latency, keeps your data
in-house, and gives you *calibrated probabilities* you can threshold — none of
which a prompted API LLM offers cleanly. The generate-vs-represent rule from
[05-pretraining.md](05-pretraining.md) decides the family: **fixed-input
understanding → encoder.**

## Text classification, end to end

The canonical pipeline. Load data, tokenize
([03-tokenization.md](03-tokenization.md)), attach a classification head over the
pooled `[CLS]`/first-token representation, fine-tune, measure.

```python
import numpy as np, torch, random
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import evaluate

random.seed(42); np.random.seed(42); torch.manual_seed(42)

ds = load_dataset("banking77")          # 77 intent classes, ~13k examples
ckpt = "microsoft/deberta-v3-small"
tok = AutoTokenizer.from_pretrained(ckpt)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=64)
ds = ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=77)

f1 = evaluate.load("f1")
def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return f1.compute(predictions=preds, references=labels, average="macro")

args = TrainingArguments(output_dir="out", learning_rate=2e-5,
    per_device_train_batch_size=32, num_train_epochs=3,
    eval_strategy="epoch", seed=42)
Trainer(model, args, train_dataset=ds["train"], eval_dataset=ds["test"],
        compute_metrics=metrics, processing_class=tok).train()
```

Two things practitioners get wrong. First, **use macro-F1, not accuracy**, on
imbalanced label sets — accuracy rewards a model that only ever predicts the
majority class. Second, **calibrate.** A fine-tuned classifier's softmax is
overconfident; if you're going to route on a confidence threshold ("auto-resolve
above 0.9, else escalate to a human"), check the reliability diagram and apply
temperature scaling. Calibration is what makes the model *usable* in a pipeline,
and it's a detail API LLMs can't give you.

## NER as token classification

Named-entity recognition labels each token, so the head is per-token instead of
per-sequence. Two mechanics trip everyone up.

**BIO tagging.** Entities span multiple tokens, so you tag with a
Begin-Inside-Outside scheme: `B-PER` starts a person span, `I-PER` continues it,
`O` is outside any entity. "Tim Cook visited Berlin" →
`B-PER I-PER O B-LOC`. The `B`/`I` distinction is what lets you separate two
adjacent entities of the same type ("...Berlin, Munich..." → `B-LOC B-LOC`, two
spans, not one).

**Subword label alignment.** Your labels are per *word*, but the tokenizer emits
*subwords* — "Cook" might stay whole while "Berlin" splits into `Ber` + `lin`.
You must spread each word's label across its subwords and, critically, mask the
continuation pieces with `-100` so they don't contribute to the loss (and so you
don't double-count in metrics):

```python
def align_labels(examples):
    tokenized = tok(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev, out = None, []
        for wid in word_ids:
            if wid is None:            # special tokens [CLS]/[SEP]
                out.append(-100)
            elif wid != prev:          # first subword of a word: real label
                out.append(labels[wid])
            else:                      # continuation subword: ignore in loss
                out.append(-100)
            prev = wid
        all_labels.append(out)
    tokenized["labels"] = all_labels
    return tokenized
```

**Evaluate with seqeval-style span F1, not token accuracy.** The `seqeval`
library scores whole entity spans: a prediction counts as correct only if the
full span *and* its type match. Token-level accuracy looks great (most tokens are
`O`) while the model is quietly mangling every boundary. On CoNLL-2003, a
DeBERTa-v3 fine-tune lands around 92–94 span-F1 — a number you'll be asked to
reproduce.

## Extractive question answering

SQuAD-style QA doesn't generate an answer; it *points* at one. Feed
`[CLS] question [SEP] context [SEP]`, and the head predicts two distributions
over context positions — probability that each token is the **start** and the
**end** of the answer span:

$$
p_{\text{start}} = \mathrm{softmax}(W_s H), \qquad
p_{\text{end}} = \mathrm{softmax}(W_e H)
$$

where `H` is the sequence of token representations. The predicted span is the
`(i, j)` with `i ≤ j` maximizing `p_start[i] · p_end[j]` within a max length.
Two production realities: contexts longer than the model's window are split into
overlapping chunks with a `doc_stride`, and each chunk gets a "no answer" option
(the `[CLS]` position) so the model can abstain when the passage doesn't contain
the answer — the SQuAD 2.0 change that made the task honest. Extractive QA is the
grounded, non-hallucinating core underneath a lot of RAG
([09-rag-agents.md](09-rag-agents.md)): when you must guarantee the answer is a
verbatim substring of a source document, a span head beats a generative model.

## Summarization and translation — and metric lies

These are generative, so they want a seq2seq model: **BART** (Lewis et al., 2019)
or **T5** (Raffel et al., 2019), fine-tuned with teacher forcing to produce a
target sequence. The hard part isn't training; it's knowing that the standard
metrics are systematically misleading.

- **ROUGE** (summarization) measures n-gram *overlap* with a reference summary —
  ROUGE-1/2 for unigram/bigram recall, ROUGE-L for longest common subsequence. It
  rewards copying reference words and is blind to a paraphrase that's perfect but
  uses different words, and blind to a fluent summary that's factually wrong. A
  high-ROUGE summary can hallucinate a number that isn't in the source.
- **BLEU** (translation) is n-gram *precision* with a brevity penalty. It's
  corpus-level, punishes valid alternative phrasings, and is notoriously
  sensitive to tokenization — quote **sacreBLEU** so your number is comparable to
  anyone else's, or it means nothing.
- **chrF** uses character-level n-grams, so it degrades gracefully on morphology
  and handles low-resource and agglutinative languages better than BLEU — prefer
  it there.

The honest 2026 practice: use ROUGE/BLEU/chrF as cheap regression signals during
development, but judge real quality with a learned metric (**BERTScore**,
**COMET** for MT, which correlate far better with humans) or an LLM-as-judge
rubric ([10-evaluation.md](10-evaluation.md)), and confirm factuality separately.
Never ship a summarizer whose only evidence is a ROUGE number — that's the mistake
that reads as junior in an interview.

## Sentence embeddings: bi- vs cross-encoders

For semantic search, dedup, clustering, and RAG retrieval you need a single
vector per sentence. A raw BERT `[CLS]` token is a poor sentence embedding.
**SBERT** (Reimers & Gurevych, 2019) fixes this by fine-tuning with a siamese
objective (mean-pooling + a contrastive/triplet loss) so that
cosine similarity between embeddings tracks semantic similarity. This is a
**bi-encoder**: encode each text independently into a vector, compare with a dot
product.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util
bi = SentenceTransformer("all-MiniLM-L6-v2")           # 22M params
emb = bi.encode(["how do I reset my password?", "password recovery steps"])
score = util.cos_sim(emb[0], emb[1])                    # ~0.7
```

The bi-encoder's superpower is that document embeddings are **precomputed once**
and indexed; a query is one forward pass plus a vector search over millions of
docs in milliseconds. The tradeoff: query and document never interact until the
dot product, so it misses fine-grained relevance.

A **cross-encoder** feeds the pair *together* — `[CLS] query [SEP] document [SEP]`
— and outputs one relevance score, letting every query token attend to every
document token. Far more accurate, but you can't precompute anything: scoring
against N documents is N forward passes. So the production pattern is a **two-stage
retrieve-then-rerank**: the bi-encoder cheaply pulls the top ~100 candidates from
millions, the cross-encoder re-scores just those 100 for precision. This is the
backbone of every serious RAG system ([09-rag-agents.md](09-rag-agents.md)).

## When a 100M encoder beats an API LLM

The decision that earns your salary. Consider intent classification on a support
stream at 5M requests/day.

| | Fine-tuned DeBERTa-v3-small (~140M) | Frontier API LLM |
|---|---|---|
| Latency (p50) | ~10–20 ms on CPU, ~3 ms on GPU | ~500–2000 ms/request |
| Cost @ 5M req/day | pennies to a few $/day of your own compute | ~$1–5k/day at per-token API prices |
| Accuracy (narrow task, enough labels) | often *higher* — trained on your labels | good zero-shot, rarely beats a tuned model |
| Calibrated probabilities | yes — threshold and route | no clean access |
| Data residency | stays in your VPC | leaves to a third party |
| Iteration | retrain in minutes | prompt-fiddle, no ground truth |

The API LLM wins when labels are scarce, classes change weekly, the task needs
world knowledge or reasoning, or volume is low enough that engineering cost
dominates inference cost. The encoder wins on high-volume, stable, narrow tasks —
which is most of production NLP. A frequent hybrid: use the LLM to *label* a few
thousand examples (cheap, one-time), then distill that into an encoder you own and
serve. Being able to run this table with real numbers is exactly the applied-NLP
design signal interviewers look for.

## Distillation to small models

When even a fine-tuned base is too big or slow, **knowledge distillation**
(Hinton et al., 2015) trains a small student to match a larger teacher's *soft*
output distribution, not just the hard labels. The soft targets carry "dark
knowledge" — the teacher's relative confidence across wrong classes — so the
student learns more per example than from labels alone. DistilBERT (Sanh et al.,
2019) keeps ~97% of BERT's task performance at ~40% fewer parameters and ~60%
faster inference. In practice you distill task-specifically: fine-tune a strong
teacher (or use an API LLM's outputs), then train a small encoder on its logits
plus your labels. This is how you hit a strict latency or edge budget without
giving up much accuracy — and it closes the loop with the economics table above.

## What interviews ask here

- Why does NER need BIO tags, and what's the `B`/`I` distinction for? (Multi-token
  spans; separating adjacent same-type entities.)
- How do you align word-level labels to subword tokens? (Label the first subword,
  mask continuations with `-100`.)
- Why report span-F1 (seqeval) instead of token accuracy for NER? (`O` dominates;
  token accuracy hides boundary errors.)
- Explain ROUGE and BLEU and one failure mode of each. (N-gram overlap/precision;
  blind to paraphrase and to factuality.)
- Bi-encoder vs cross-encoder — when each, and why the retrieve-then-rerank
  pattern? (Precompute + speed vs pairwise accuracy.)
- When would you fine-tune a 100M encoder instead of calling an API LLM? (High
  volume, stable narrow task, latency/cost/privacy/calibration.)
- How does extractive QA predict a span? (Start and end distributions over context
  tokens; argmax over valid `(i,j)`.)

## Where this shows up on the job

- **Ticket/intent/content classification** at scale — the single most common
  applied-NLP deliverable, and the one that most clearly beats an API LLM on cost.
- **Information extraction** — NER and span QA pipelines pulling structured fields
  (names, amounts, dates) out of documents, contracts, and logs.
- **Semantic search and RAG retrieval** — bi-encoder embeddings plus a
  cross-encoder reranker are the retrieval core of nearly every RAG product.
- **Cost-reduction mandates** — "make this 10× cheaper" almost always means
  distilling or replacing an LLM call with a fine-tuned encoder for the
  high-volume slice of traffic.
