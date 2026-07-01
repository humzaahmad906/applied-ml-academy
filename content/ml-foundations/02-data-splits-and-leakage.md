# 02 — Data Splits and Leakage

Here's the single most important idea in this whole course: **a model that has seen the answers isn't being tested — it's being quizzed on material it already memorized.** If you train a model and then check its accuracy on the very same data, you learn almost nothing about how it will behave in the real world. You have to hold data back.

This module is about doing that correctly, and about the sneaky ways data can "leak" the answers even when you think you've been careful.

## Why you can't grade on training data

A powerful model can memorize its training data outright. Imagine a student who gets the exam questions in advance and simply memorizes the answer key. They'll ace that exam and tell you nothing about whether they understand the subject. Hand them a new exam and they fall apart.

Models do exactly this. The behavior has a name — **overfitting** — and we devote a whole module to it later. For now, the fix is simple: keep some data locked away so you can grade the model on questions it has never seen.

## Train, validation, test

The standard practice is to split your data into three parts:

- **Training set** (often ~60–70%): the model learns from this.
- **Validation set** (~15–20%): you use this to compare models and tune settings. You'll look at it many times while developing.
- **Test set** (~15–20%): the final exam. You touch it **once**, at the very end, to report an honest estimate of real-world performance.

The validation/test distinction confuses beginners. Why two held-out sets? Because every time you look at a result and change something in response, you're subtly fitting to that data. Tune your model against the validation set fifty times and you've indirectly overfit to it. The test set stays sealed so you have one clean, uncontaminated measurement left.

A common rule: if you've looked at the test set more than once, it's no longer a test set.

## Splitting in code

scikit-learn makes the basic split a one-liner. To get three sets, split twice:

```python
from sklearn.model_selection import train_test_split

# First split off the test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Then split the remainder into train (75%) and validation (25%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(len(X_train), len(X_val), len(X_test))
```

The `random_state=42` makes the split reproducible — you get the same split every time you run it, which matters when you're comparing results. For classification, add `stratify=y` so each split keeps the same proportion of classes; otherwise a rare class might land entirely in one set.

## Data leakage: the silent killer

**Data leakage** is when information from outside the training set sneaks into training, giving the model a peek at answers it shouldn't have. A leaky model looks brilliant during development and then collapses in production. It is the most common way beginners fool themselves, and it's worth learning to spot.

Here are the classic forms.

### Leaking through preprocessing

Suppose you scale your features so they have mean zero. If you compute that mean over the *entire* dataset before splitting, the mean carries information about the test set into the training process. The training set now "knows" something about the test data.

The fix: **fit your preprocessing on the training set only**, then apply it to validation and test.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std HERE
X_test_scaled = scaler.transform(X_test)         # only apply
```

Using `fit_transform` on the test set would be the leak. Scikit-learn `Pipeline` objects handle this correctly for you, which is a big reason to use them.

### Target leakage: features that contain the answer

This one is brutal. Imagine predicting whether a patient has a disease, and one of your features is `medication_prescribed`. That medication is only given *after* diagnosis. It practically is the label. Your model will look 99% accurate and be useless — in the real world, at prediction time, you don't yet know the medication.

Always ask of every feature: **would this value actually be available at the moment I need to make a prediction?** If it's only known after the outcome, drop it.

### Leaking through time

If your data has a time dimension — stock prices, user behavior, sales — a random split lets the model train on the future and predict the past. That's cheating. For time series, split by time: train on older data, test on newer data.

### Leaking through duplicates and groups

If the same patient appears in multiple rows, a random split can put some of their rows in training and some in test. The model recognizes the patient, not the pattern. When examples come in groups (multiple readings per person, multiple photos per product), split by *group* so an entire group lands on one side.

## Key takeaways

- Never evaluate a model on data it trained on — hold data back.
- Use three sets: train to learn, validation to tune, test as a one-time final exam.
- Fit preprocessing on the training set only, then apply to the rest.
- Target leakage — a feature that's only known after the outcome — is the most dangerous and common mistake.
- Split by time for time-series data and by group when examples are related.

## Try it

Take any dataset and deliberately create a leak: add a new feature that's just the label plus a tiny bit of noise (`y + small_random`). Train a model with and without that feature and compare the test scores. Watch the leaky version score near-perfectly. Then write one sentence explaining why that "great" score is a lie.
