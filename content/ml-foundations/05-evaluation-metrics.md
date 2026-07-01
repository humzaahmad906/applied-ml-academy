# 05 — Evaluation Metrics

"My model is 95% accurate!" sounds great — until you learn that 95% of the emails were legitimate, so a model that blindly guesses "not spam" every single time also scores 95%. A single number can hide a broken model. Choosing the right metric, and understanding when each one misleads, is one of the most important skills you'll build. This module is about classification metrics, where the traps are richest.

## The confusion matrix: where every metric comes from

Every classification metric is just a different way of summarizing four counts. Line up your predictions against the truth for a yes/no problem and you get:

- **True Positive (TP):** predicted positive, actually positive. Correct.
- **True Negative (TN):** predicted negative, actually negative. Correct.
- **False Positive (FP):** predicted positive, actually negative. A false alarm.
- **False Negative (FN):** predicted negative, actually positive. A miss.

```python
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

The `confusion_matrix` prints a 2x2 grid of those counts. Internalize it — every metric below is a ratio of these four numbers.

## Accuracy and why it lies

**Accuracy** is the fraction of predictions that were correct: `(TP + TN) / everything`. It's intuitive and it's the right metric when your classes are roughly balanced and every kind of error costs about the same.

It falls apart on **imbalanced** data. If 99% of transactions are legitimate, a model that predicts "legitimate" for everything scores 99% accuracy while catching zero fraud. The number looks fantastic and the model is worthless. Whenever one class is rare, treat accuracy with deep suspicion.

## Precision and recall

These two split the errors apart and answer different questions.

**Precision** = `TP / (TP + FP)`. Of everything the model *flagged* as positive, how much was actually positive? High precision means few false alarms. You care about precision when acting on a positive is expensive or annoying — you don't want to wrongly flag good customers as fraudsters and freeze their accounts.

**Recall** = `TP / (TP + FN)`. Of all the *actual* positives out there, how many did the model catch? High recall means few misses. You care about recall when missing a positive is dangerous — screening for a serious disease, you'd rather have some false alarms than let a real case slip through.

Here's the tension: precision and recall usually trade off against each other. Lower your decision threshold to catch more positives and recall rises — but you catch more false alarms too, so precision drops. Raise the threshold for higher precision and you miss more real cases. There's no free lunch; you pick the balance your problem demands.

## F1: one number for the tradeoff

When you want a single score that respects both precision and recall, use the **F1 score** — their harmonic mean:

```
F1 = 2 * (precision * recall) / (precision + recall)
```

The harmonic mean punishes imbalance: you can't get a high F1 by acing one and tanking the other. A model with 99% precision and 2% recall has an F1 near 4%, correctly reflecting that it's useless. F1 is a solid default for imbalanced classification. Its limitation: it ignores true negatives entirely and treats precision and recall as equally important, which isn't always true for your problem.

## ROC-AUC: judging across all thresholds

Everything above depends on a fixed threshold (usually 0.5). But your model outputs a *probability*, and the threshold is a choice. **ROC-AUC** evaluates the model across *every* possible threshold at once.

The ROC curve plots the true-positive rate against the false-positive rate as you sweep the threshold from 0 to 1. The **AUC** (area under that curve) collapses it to one number: the probability that the model ranks a random positive example above a random negative one. AUC of 1.0 is perfect; 0.5 is a coin flip.

```python
from sklearn.metrics import roc_auc_score

# Use probabilities of the positive class, not hard predictions
probs = model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, probs))
```

AUC is threshold-independent, which makes it great for comparing models. But it has its own trap: on heavily imbalanced data, ROC-AUC can look reassuringly high even when the model is poor at finding the rare class, because the huge number of true negatives flatters the score. In that situation, the **precision-recall AUC** is more honest.

## So which one do I use?

There's no universal answer — the metric must match what a mistake actually costs.

- **Balanced classes, equal error costs:** accuracy is fine.
- **Imbalanced classes:** avoid accuracy; look at precision, recall, and F1.
- **Missing a positive is dangerous** (disease, fraud): prioritize recall.
- **False alarms are costly** (flagging good users): prioritize precision.
- **Comparing models regardless of threshold:** ROC-AUC (or PR-AUC when very imbalanced).

Always decide your metric *before* you start modeling. Picking it afterward, once you've seen which one makes your model look good, is a subtle way to fool yourself.

## Key takeaways

- Every classification metric derives from four counts: TP, TN, FP, FN — the confusion matrix.
- Accuracy is misleading on imbalanced data; a lazy model can score high while being useless.
- Precision (few false alarms) and recall (few misses) trade off; choose based on which error hurts more.
- F1 combines precision and recall into one number that punishes ignoring either.
- ROC-AUC judges ranking across all thresholds but can flatter models on imbalanced data — prefer PR-AUC there.

## Try it

Build a deliberately imbalanced dataset where 95% of labels are class 0. Train any classifier, then print both its accuracy and its `classification_report`. Compare the accuracy to the recall on the rare class. Write one sentence explaining why accuracy alone would have fooled you here.
