# 03 — Linear and Logistic Regression

Before reaching for anything fancy, reach for these two. Linear regression and logistic regression are the workhorses of classical machine learning: fast, interpretable, hard to break, and good enough to solve a surprising number of real problems. They also teach the core intuition that every more complex model builds on. Start here, always.

## Linear regression: drawing the best line

Linear regression predicts a number by assuming the answer is a weighted sum of the features. With one feature — say, house size — it literally fits a straight line through your data points:

```
price = w * size + b
```

Here `w` (the weight, or slope) says how much the price changes per extra square foot, and `b` (the intercept, or bias) is the baseline. With many features, it's the same idea extended:

```
price = w1*size + w2*bedrooms + w3*age + ... + b
```

"Fitting" the model means finding the weights that make the predictions as close as possible to the true values. Closeness is measured by the **sum of squared errors** — take each prediction's gap from the truth, square it (so big misses hurt more and negatives don't cancel positives), and add them up. The best line is the one with the smallest total.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Weights:", model.coef_)
print("Intercept:", model.intercept_)
```

The great thing about linear models is that `coef_` is readable. A weight of `120` on `size_sqft` means: holding everything else fixed, each additional square foot adds about $120 to the predicted price. Very few models hand you that kind of plain-English explanation.

## Reading the coefficients (carefully)

A word of caution: coefficients are only comparable if the features are on similar scales. `size_sqft` ranges into the thousands while `bedrooms` ranges 1–6, so their raw weights aren't directly comparable. Scale your features first (as in the previous module) if you want to judge which feature matters most. And "matters most" here means *in this model on this data* — it's not proof of cause and effect. A model can put a big weight on a feature that's merely correlated with the real driver.

## Logistic regression: linear regression for categories

Despite the name, logistic regression does **classification**, not regression. It predicts the *probability* that an example belongs to a class.

The problem with using plain linear regression for a yes/no question is that a line runs off to infinity in both directions — it'll happily predict a "probability" of 1.7 or -0.3, which is nonsense. Logistic regression fixes this by passing the weighted sum through the **sigmoid** function, an S-shaped curve that squashes any number into the range 0 to 1:

```
probability = sigmoid(w1*x1 + w2*x2 + ... + b)
```

Very negative inputs map near 0, very positive inputs near 1, and 0 maps to 0.5. Now the output is a genuine probability. To make a hard decision, you pick a **threshold** — usually 0.5. Above it, predict the positive class; below, the negative.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Hard class predictions
print(model.predict(X_test[:5]))
# Probabilities — the second column is P(class = 1)
print(model.predict_proba(X_test[:5]))
```

`predict` gives you `[0 1 1 0 1]`; `predict_proba` gives you the underlying probabilities like `[0.12, 0.88]` per row. The probabilities are often more useful than the hard labels — they tell you how *confident* the model is, which lets you set the threshold to suit your problem. If missing a fraud case is far worse than a false alarm, you'd lower the threshold to catch more positives.

## When to use them (and when not)

These models assume the relationship is roughly a weighted sum — a straight-line trend for regression, a smooth probability boundary for classification. That assumption is their strength and their limit.

Reach for linear/logistic regression when:

- You want an interpretable model you can explain to a non-technical stakeholder.
- You have limited data — simple models overfit less.
- The relationship is plausibly smooth and additive.
- You need a fast, honest **baseline** before trying anything complex.

That last point is underrated. Always fit a linear or logistic baseline first. If your fancy model can't beat it, the fancy model isn't worth its complexity.

They struggle when the true relationship is highly non-linear or full of interactions (feature A only matters when feature B is high). You can sometimes rescue them by adding engineered features — squares, products, buckets — but at some point it's easier to switch to trees, which is exactly the next module.

## Key takeaways

- Linear regression predicts a number as a weighted sum of features; it minimizes squared error.
- Coefficients are interpretable, but scale your features before comparing them and don't confuse correlation with cause.
- Logistic regression does classification by squashing the weighted sum through a sigmoid into a 0–1 probability.
- `predict_proba` and an adjustable threshold let you tune for your problem's real costs.
- Always fit one of these as a baseline before reaching for a complex model.

## Try it

Fit a `LogisticRegression` on the built-in breast cancer dataset (`from sklearn.datasets import load_breast_cancer`). Use `predict_proba` to get the probabilities for the test set. Then count how many predictions change if you move the threshold from 0.5 to 0.3 instead of using `predict`. What kind of problem would justify that lower threshold?
