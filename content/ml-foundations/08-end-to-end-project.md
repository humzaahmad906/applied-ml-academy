# 08 — A First End-to-End Project

You now have every piece: framing a problem, splitting data, avoiding leakage, fitting models, reading metrics, controlling overfitting, and validating honestly. This final module ties them into one continuous workflow. We'll walk a realistic classification problem from raw data to an evaluated model, in the order you'd actually do it. The specific dataset matters less than the *sequence* — burn this sequence into your habits and you'll approach any new problem with confidence.

## Step 1: Frame the problem

Before touching data, answer three questions. What am I predicting? Is it a number (regression) or a category (classification)? And what does a mistake cost?

For our walkthrough: we're predicting whether a customer will churn (leave) next month — a binary classification. A false negative (missing a churner) means losing a customer we could have saved; a false positive (flagging a loyal customer) means a wasted retention offer. Missing churners is worse, so we'll care about **recall**, and we'll watch **F1** because churn is usually imbalanced. We commit to these metrics *now*, before modeling, so we can't cheat by picking whatever looks good later.

## Step 2: Load and look at the data

Never model data you haven't looked at. Load it and inspect.

```python
import pandas as pd

df = pd.read_csv("customers.csv")
print(df.shape)
print(df.head())
print(df.isnull().sum())          # missing values per column
print(df["churn"].value_counts()) # class balance
```

Here you're checking: How big is it? What are the columns? Where are values missing? How imbalanced is the target? Suppose we find churn is 20% of rows (imbalanced, as expected) and a few columns have missing values. These findings shape every later decision.

## Step 3: Split first — before anything else

This is the discipline that prevents leakage. Split *before* you compute any statistic, fill any missing value, or scale anything. If you fill missing values using the column mean computed over the whole dataset, you've leaked test information into training. Split first, always.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

Note `stratify=y` — with imbalanced data, it keeps the 20% churn rate consistent across both sets. And we lock the test set away; from here on we work only with the training data until the very end.

## Step 4: Build a preprocessing + model pipeline

Real data needs cleaning: fill missing values, scale numbers, encode categories. Doing these steps by hand invites leakage, because it's easy to accidentally fit them on the wrong data. A **Pipeline** bundles preprocessing with the model so that during cross-validation, preprocessing is refit on each fold's training portion only — leakage-proof by construction.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

numeric = X.select_dtypes("number").columns
categorical = X.select_dtypes("object").columns

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
])
```

`class_weight="balanced"` tells the model to pay more attention to the rare churn class — a simple, effective way to handle imbalance.

## Step 5: Establish a baseline

Before celebrating any fancy model, know what "doing nothing clever" scores. Fit a simple logistic regression baseline and a trivial "always predict the majority class" reference. If your real model can't beat these, it isn't earning its complexity.

## Step 6: Cross-validate and tune

Now use cross-validation on the *training* data to estimate performance and tune hyperparameters — scoring on F1 to match the goal we set in step 1.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

base = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print(f"Baseline model F1: {base.mean():.3f} ± {base.std():.3f}")

grid = {"clf__n_estimators": [100, 300], "clf__max_depth": [5, 10, None]}
search = GridSearchCV(model, grid, cv=5, scoring="f1")
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
```

Because the pipeline is inside cross-validation, imputation and scaling are refit correctly on each fold. Nothing leaks.

## Step 7: The final test — once

Only now, with a chosen model and settled hyperparameters, do we unlock the test set. This is the one honest measurement, and we look at all the metrics we committed to.

```python
from sklearn.metrics import classification_report, roc_auc_score

best = search.best_estimator_
preds = best.predict(X_test)
probs = best.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
```

We read the churn class's recall and F1 first, since those match our goal. If the test numbers roughly match the cross-validation numbers, great — the estimate was honest. If the test score is far worse, something leaked or we overfit during tuning, and we investigate rather than ship.

## Step 8: Interpret and decide

A number isn't a decision. Look at `best.named_steps["clf"].feature_importances_` to see what drove predictions — do they make business sense? Consider adjusting the decision threshold if recall matters more than the default 0.5 gives. And write down the model's limits: what data it was trained on, and where it shouldn't be trusted. A model deployed without understanding its blind spots is a liability.

## Key takeaways

- The workflow is a fixed sequence: frame, inspect, split first, build a leak-proof pipeline, baseline, cross-validate and tune, test once, interpret.
- Commit to your metric before modeling so you can't rationalize later.
- Split before any preprocessing, and wrap preprocessing in a Pipeline so cross-validation can't leak.
- Always beat a simple baseline before trusting a complex model.
- The test set is opened exactly once; a large gap from your CV estimate means something went wrong.

## Try it

Pick any tabular dataset with a categorical target — the built-in wine or breast cancer datasets work, or a Kaggle CSV. Run all eight steps end to end in a single notebook, writing one comment above each step naming which step it is. Compare your tuned model's test F1 against your logistic-regression baseline. Did the extra complexity earn its keep? If not, that's a real and valuable result — the simple model wins.
