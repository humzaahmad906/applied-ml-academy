# 01 — What Machine Learning Is

Machine learning is the art of getting a computer to find patterns in data instead of following rules you write by hand. If you've ever written a program with a giant pile of `if` statements to catch spam email, you know how quickly that approach breaks down. Spammers change their wording, and your rules go stale. Machine learning flips the problem around: you show a program thousands of emails already labeled "spam" or "not spam," and it figures out the rules on its own.

That's the whole idea. You don't tell the machine *how* to solve the problem. You give it examples and let it learn the pattern.

## Learning from examples

Every machine learning problem starts with data arranged as a table. Each row is one **example** (also called a sample or an observation). Each column is a **feature** — a measurable property of that example.

Imagine a table of houses:

| size_sqft | bedrooms | age_years | price |
|-----------|----------|-----------|-------|
| 1400      | 3        | 20        | 250000 |
| 2100      | 4        | 5         | 410000 |
| 900       | 2        | 45        | 150000 |

The first three columns (`size_sqft`, `bedrooms`, `age_years`) are the features. The last column, `price`, is the **label** — the thing we want to predict. In machine learning notation you'll often see the features called `X` and the labels called `y`.

The goal is to learn a relationship: given the features `X`, predict the label `y`. Once learned, you can feed in a brand-new house the model has never seen and get a price estimate.

## Supervised vs. unsupervised

The house example is **supervised learning**: every training example comes with a known answer (the label). The model learns by comparing its guesses to the true answers and adjusting. Supervised learning splits into two flavors:

- **Regression** predicts a number. House price, tomorrow's temperature, how many minutes a delivery will take.
- **Classification** predicts a category. Spam or not spam, which of three species a flower is, whether a transaction is fraud.

**Unsupervised learning** has no labels at all. You hand the algorithm a pile of data and ask it to find structure. The most common form is **clustering** — grouping similar examples together. A store might cluster its customers into segments without knowing in advance what those segments are. There's no "right answer" to check against, which makes unsupervised learning harder to evaluate.

This course focuses almost entirely on supervised learning, because it's where beginners get the most traction and the clearest feedback.

## The workflow

Nearly every supervised project follows the same loop. Learning this loop is more valuable than memorizing any single algorithm.

1. **Frame the problem.** What are you predicting, and is it a number or a category? What would a good prediction be worth?
2. **Get the data.** Collect examples with features and labels. This is usually the messiest, most time-consuming step.
3. **Split the data.** Hold some examples aside so you can honestly test the model later. (This matters enough that the next module is entirely about it.)
4. **Train a model.** Feed the training examples to a learning algorithm. It adjusts its internal settings to fit the pattern.
5. **Evaluate.** Measure how well the model does on data it hasn't seen. This tells you whether it actually learned something useful or just memorized.
6. **Iterate.** Try different features, different models, different settings. Machine learning is loops, not straight lines.

## Your first model in code

Here's the entire workflow in a few lines using scikit-learn, the most popular classical ML library in Python. We'll use a built-in dataset of house features from California.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load features (X) and labels (y)
data = fetch_california_housing()
X, y = data.data, data.target

# Hold 20% of the data aside for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on unseen data
predictions = model.predict(X_test)
print(predictions[:5])
```

Running this prints five predicted house values (in hundreds of thousands of dollars), something like `[0.72 1.76 2.71 2.84 2.60]`. The model has never seen these particular houses — it's applying the pattern it learned from the training set.

Notice how little code it took. That's deliberate. scikit-learn gives every model the same three methods: `fit` to train, `predict` to make predictions, and `score` to evaluate. Once you know the pattern, swapping in a completely different algorithm is a one-line change. The hard part was never the code — it's understanding what the model is doing and whether you can trust the result.

## A word of caution

That model *will* produce a number for any house you give it, even a nonsensical one. Machine learning models are confident by default and rarely say "I don't know." A huge part of being good at this is knowing when to distrust your own model. We'll return to this theme constantly — especially the ways a model can look great on paper and fail in the real world.

## Key takeaways

- Machine learning finds patterns from examples instead of hand-written rules.
- Data is a table: rows are examples, columns are features (`X`), and the target is the label (`y`).
- Supervised learning uses labeled data; regression predicts numbers, classification predicts categories. Unsupervised learning finds structure without labels.
- Every project follows the same loop: frame, get data, split, train, evaluate, iterate.
- In scikit-learn, every model shares `fit`, `predict`, and `score`.

## Try it

Load the built-in iris dataset with `from sklearn.datasets import load_iris`. Print how many examples and how many features it has (hint: `X.shape`). Then answer on paper: is predicting the flower species a regression or a classification problem, and why? What would the features and the label be?
