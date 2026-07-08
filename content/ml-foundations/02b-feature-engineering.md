# 02b — Feature Engineering

You have your data split correctly and no leaks (module 02). Before you can fit the models in the next module, there's a gap nobody warns beginners about: raw data almost never arrives in a form a model can actually use. There are text columns, missing cells, and numbers on wildly different scales. This module is about getting from a messy table to clean, model-ready features — the exact preprocessing the end-to-end capstone (module 08) leans on.

## Why raw data isn't model-ready

Almost every classical ML model is, under the hood, doing arithmetic on numbers. That imposes three quiet demands on your data:

- **Everything has to be a number.** A column like `city = "Lahore"` means nothing to a matrix multiply. Text categories have to be turned into numbers first.
- **Scales should be comparable.** If `income` runs into the hundreds of thousands and `age` runs 0–100, many models will treat income as overwhelmingly more important simply because its numbers are bigger — not because it matters more.
- **No missing values.** A single `NaN` will make most sklearn models throw an error and refuse to fit at all.

Feature engineering is the work of satisfying these demands — and, done well, of giving the model better features than it started with. Let's take the three problems in turn, then wire the fixes together.

## Numeric scaling

Consider two features, `income` (say, 20,000–200,000) and `age` (18–90). To a model that measures distance between points or sums up weighted features, `income` dominates purely by magnitude. **Scaling** puts every numeric feature on a comparable footing.

The workhorse is **standardization** — subtract the mean and divide by the standard deviation, so each feature ends up centered at 0 with a spread of about 1:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std on TRAIN
X_test_scaled = scaler.transform(X_test)          # only apply
```

The alternative is **min-max scaling** (`MinMaxScaler`), which squeezes each feature into a fixed range, usually 0–1. Rough guidance:

- `StandardScaler` is the sensible default. It handles outliers more gracefully and suits most linear models.
- `MinMaxScaler` is handy when you specifically need bounded values (some neural nets, image pixel intensities).

Why bother? Three families of model genuinely care about scale:

- **Linear and logistic regression** with regularization penalize large coefficients; unscaled features make that penalty unfair.
- **Distance-based models** (k-nearest neighbors, k-means) measure closeness — a big-magnitude feature swamps the distance.
- **Gradient descent** (the optimizer behind most models) converges far faster when features share a scale.

One important exception: **tree-based models — decision trees, random forests, gradient boosting — don't care about scale at all.** They split on thresholds ("is income > 50,000?"), and a threshold works identically whether or not the column was scaled. If your whole pipeline is trees, you can skip scaling. It won't hurt, but it won't help either.

Notice the `fit_transform` on train, `transform` on test pattern — the same discipline from module 02. The scaler learns the mean and standard deviation from the training set only. Computing them over the full dataset would leak test information into training.

## Encoding categoricals

This is the wall beginners hit on day one: a column of strings. You have to turn categories into numbers, and *how* you do it depends on whether the categories have an inherent order.

**Nominal** categories have no order — `city`, `color`, `payment_method`. For these, use **one-hot encoding**: create one new 0/1 column per category. A `color` column becomes `color_red`, `color_green`, `color_blue`, with a single 1 marking the right one.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X_train[["city"]])
```

Why not just map red→0, green→1, blue→2? Because that invents an order (blue > green > red) and a spacing (blue is "twice" green) that don't exist. A linear model would take that fake ordering literally. One-hot avoids it by giving each category its own independent column.

`handle_unknown="ignore"` is quietly essential. If a category shows up in the test set (or in production) that wasn't in the training data — a new city, say — the default behavior is to raise an error and crash. With `"ignore"`, the encoder simply outputs all zeros for that row's category and moves on. This is exactly the train-only-fitting discipline again: the encoder locks in the categories it saw during `fit`, and anything new later is handled gracefully.

**Ordinal** categories *do* have a meaningful order — `size` (small < medium < large), `education` (high school < bachelor's < master's). Here you *want* the numeric order, so use `OrdinalEncoder` and spell out the ranking yourself:

```python
from sklearn.preprocessing import OrdinalEncoder

size_encoder = OrdinalEncoder(categories=[["small", "medium", "large"]])
```

Passing `categories` explicitly matters — left to itself, `OrdinalEncoder` orders alphabetically, which would put "large" before "small."

**A caution on high cardinality.** One-hot encoding a column with thousands of distinct values (like `zip_code` or `user_id`) explodes your feature count into thousands of mostly-zero columns — slow to fit and easy to overfit. For high-cardinality columns, consider grouping rare categories into an "other" bucket, or reach for target/frequency encoding (a more advanced topic). The rule of thumb: one-hot is great up to a few dozen categories, questionable beyond that.

## Missing values

Real data has holes — a sensor dropped out, a form field was left blank. Since models choke on `NaN`, you have to fill them in, a step called **imputation**. sklearn's `SimpleImputer` covers the common strategies:

- `strategy="mean"` — fill numeric gaps with the column mean (sensitive to outliers).
- `strategy="median"` — fill with the median (a safer default for skewed numeric data).
- `strategy="most_frequent"` — fill with the most common value; works for categoricals.
- `strategy="constant", fill_value=...` — fill with a value you choose, e.g. the string `"missing"` so "was it absent?" becomes its own signal.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

And here is the leakage trap, straight out of module 02. **The imputation value must be learned from the training set only.** If you compute the median over the entire dataset before splitting, the test set's values have quietly influenced a number that gets baked into training — a leak. `fit` on train, `transform` on test. Every preprocessing step in this module obeys the same rule, which is precisely why we bundle them into a pipeline rather than run them by hand.

## Putting it together with ColumnTransformer

Here's the practical problem: numeric columns need imputing then scaling, while categorical columns need imputing then one-hot encoding. Different columns, different treatment. `ColumnTransformer` lets you route each group of columns through its own mini-pipeline, and `Pipeline` chains those steps in order. Wrapped together, the whole thing fits on train and applies consistently to everything else — with zero leakage, because `ColumnTransformer` handles the fit/transform split for you.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    "age":     [25, 42, 37, None, 29, 51, 33, 46],
    "income":  [40_000, 85_000, 62_000, 50_000, None, 120_000, 47_000, 90_000],
    "city":    ["Lahore", "Karachi", "Lahore", "Multan",
                "Karachi", "Lahore", None, "Multan"],
    "churned": [0, 1, 0, 0, 1, 1, 0, 1],
})

X = df.drop(columns="churned")
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

numeric_features = ["age", "income"]
categorical_features = ["city"]

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)          # fits ALL preprocessing on train only
print(model.score(X_test, y_test))    # transforms test with train-learned stats
```

That single `model.fit(X_train, y_train)` call learns the medians, the scaling, the categories, *and* the classifier — all from the training data. `model.score` (or `model.predict`) then applies the exact same transformations to the test set. There is no way to accidentally leak, no separate objects to keep in sync, and the whole thing saves and deploys as one unit. This is the backbone of the capstone in module 08, and it's the pattern you should reach for every time.

## Feature engineering beats tuning

One last idea, and it's the one that separates competent practitioners from beginners: **creating better features usually beats tuning the model.** Squeezing another 0.5% out of hyperparameters is a grind; handing the model a feature that captures the real signal is often a leap.

A few reliable moves:

- **Ratios and combinations.** `debt / income` may predict default far better than either column alone.
- **Binning.** Turning a continuous `age` into buckets (child / adult / senior) can expose a non-linear pattern a linear model can't otherwise see.
- **Interactions.** A product like `price × quantity` captures "total spend," which neither factor shows on its own.
- **Dates.** Explode a timestamp into day-of-week, month, or is-weekend — the raw datetime is nearly useless, the parts are gold.

Every such feature comes from *understanding the problem*, not from the algorithm. Spend your time here before you spend it tuning.

## Key takeaways

- Models need all-numeric, comparably-scaled, gap-free features — raw data rarely qualifies.
- Scale numeric features with `StandardScaler` (default) for linear, distance-based, and gradient-descent models; tree models don't need it.
- One-hot encode nominal categories, ordinal-encode ordered ones, and always set `handle_unknown="ignore"` so unseen categories don't crash prediction.
- Impute missing values with `SimpleImputer` — and learn the fill value from the training set only, or you'll leak.
- Wrap everything in `ColumnTransformer` + `Pipeline` so preprocessing is fit on train and applied consistently — this is exactly what the capstone uses.
- Good engineered features usually beat model tuning.

## Try it

Take the small dataframe above and add a categorical `plan` column with values `"basic"`, `"pro"`, `"enterprise"` (an *ordered* tier). Route it through an `OrdinalEncoder` with the correct order instead of one-hot, adding a third branch to the `ColumnTransformer`. Then deliberately put a brand-new city (e.g. `"Quetta"`) only in the test set and confirm the pipeline still predicts without error — that's `handle_unknown="ignore"` earning its keep.
