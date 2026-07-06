# 05 — Type Hints and Static Analysis

Python does not force you to declare the types of your variables, and for small scripts that freedom is pleasant. But as programs grow, the lack of type information becomes a liability: you pass a list where a dict was expected, forget that a function can return `None`, or call it with the arguments in the wrong order — and Python happily runs until it crashes somewhere far from the mistake. **Type hints** let you annotate what your code expects, turning that documentation into something a tool can check before you ever run the program. This lesson shows you how to add hints and how to catch bugs with a type checker.

## Why bother with types

A type hint is a promise about what a value is. Writing `features: list[float]` says "this should be a list of floats." Three benefits follow:

- **Enforced documentation.** Unlike a comment, a type hint cannot drift out of date without a checker complaining. It states intent precisely, and the machine holds you to it.
- **Catch bugs before running.** A type checker reads your annotations and flags mismatches — a `None` you forgot to handle, a wrong argument, a function that promises to return a number but sometimes returns nothing.
- **Better tooling.** Editors use hints to autocomplete, to warn you inline, and to navigate your code.

Crucially, type hints do not change how your program runs. Python ignores them at runtime; they exist for you and for the checker.

## Basic annotations

You annotate a parameter with `name: type` and a return value with `-> type` after the parentheses:

```python
def predict(features: list[float]) -> dict[str, float]:
    total = sum(features)
    return {"score": total / len(features)}
```

This says `predict` takes a list of floats and returns a dict mapping strings to floats. Note the built-in generics: `list[float]`, `dict[str, float]`. Since Python 3.10 you write these directly with the lowercase built-in types — you no longer need `List` and `Dict` imported from the `typing` module, which you may still see in older code.

Variables can be annotated too, though it is usually only necessary when the type isn't obvious:

```python
threshold: float = 0.5
labels: list[str] = []
```

## When a value might be missing

Real code often has values that are sometimes absent. A function might return a result or `None`. You express "X or None" with `X | None`:

```python
def find_user(user_id: int) -> dict[str, str] | None:
    if user_id in database:
        return database[user_id]
    return None
```

The `| None` is the modern spelling of what older code writes as `Optional[dict]`. The same `|` syntax expresses "one of several types" — `int | str` means "an int or a string" (older code writes `Union[int, str]`). Making `None` explicit is one of the highest-value things type hints do, because forgetting to handle a missing value is such a common bug, and a checker will now force you to.

## Aliases and forward references

When a type expression gets long or repeats, give it a name with a `TypeAlias`:

```python
from typing import TypeAlias

FeatureVector: TypeAlias = list[float]

def predict(features: FeatureVector) -> float:
    return sum(features) / len(features)
```

And if you reference a type that isn't defined yet (a class further down the file, for instance), add this line at the top of the module so all annotations are treated as text and resolved lazily:

```python
from __future__ import annotations
```

This is a common, harmless line to include; it sidesteps a whole category of ordering problems in annotations.

## A few more tools from typing

The `typing` module has a handful of constructs worth knowing:

- **`Literal`** restricts a value to specific constants: `def load(mode: Literal["r", "w"])` accepts only `"r"` or `"w"`.
- **`TypedDict`** describes the shape of a dict with known keys: a `dict` that always has `name: str` and `age: int`.
- **`Protocol`** describes a type by what it can do rather than what it inherits from — "anything with a `.predict()` method" — which is how you type duck-typed code.

```python
from typing import Literal, TypedDict


class Prediction(TypedDict):
    label: str
    confidence: float


def format_output(mode: Literal["json", "text"], result: Prediction) -> str:
    ...
```

You do not need to master these on day one — reach for them as the need arises.

## Pydantic: types that validate at runtime

There is a popular library, **Pydantic**, that takes typed classes one step further: it uses your annotations to *validate real data at runtime*. You declare a model as a class with typed fields, and Pydantic enforces those types when you build one:

```python
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float]
    threshold: float = Field(ge=0, le=1)
```

If someone hands this model a string where a float belongs, or a threshold above 1, Pydantic raises a clear validation error. This bridges the gap between "hints the checker reads" and "guarantees enforced on live data," and it is the backbone of request validation in web APIs — which you will use heavily in the APIs course.

## Running a type checker

Annotations only pay off when a tool checks them. The established default is **mypy**. Point it at your source and it reports mismatches:

```bash
mypy src/
```

A faster, stricter alternative is **pyright**, which also powers the excellent type-checking built into VS Code — many developers get pyright's feedback live as they type, with no separate command. Either tool is a fine choice; pick one and configure it in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
strict = true
```

Do not try to type an entire existing codebase overnight. **Gradual typing** is the whole point: start by annotating your function signatures — the parameters and return types — and the checker immediately catches the most common mistakes at your code's boundaries. Add internal annotations over time where they earn their keep. A partially typed codebase is strictly better than an untyped one.

*(A footnote for the curious: Astral, the makers of uv and ruff, are building a new type checker called `ty` that aims to be extremely fast. As of 2026 it is too new to recommend for real projects — stick with mypy or pyright for now.)*

## Key takeaways

- Type hints are enforced, checkable documentation of what your code expects; they don't affect how the program runs.
- Annotate parameters with `name: type` and returns with `-> type`, using built-in generics like `list[float]` and `dict[str, float]` (3.10+), not `List`/`Dict`.
- Express "maybe missing" with `X | None` and "one of several" with `A | B`; making `None` explicit prevents a whole class of bugs.
- `TypeAlias` names repeated types; `from __future__ import annotations` avoids forward-reference ordering issues.
- `Literal`, `TypedDict`, and `Protocol` cover constants, dict shapes, and duck-typed interfaces when you need them.
- Pydantic models turn type annotations into runtime validation — the foundation of API request checking (covered in the APIs course).
- Check your types with `mypy src/` (the default) or pyright (faster, great in VS Code); type gradually, starting with function signatures.

## Try it

Take a small module you have written — the `math_utils.py` from earlier lessons is perfect — and add type hints to every function signature: annotate the parameters and the return types, using built-in generics and `| None` where a function can return nothing. Add `from __future__ import annotations` at the top. Install mypy (`uv add --dev mypy`), add a `[tool.mypy]` section to your `pyproject.toml`, and run `mypy src/`. Fix anything it flags. Then deliberately introduce a bug — pass a `str` to a function expecting an `int` somewhere — and confirm mypy catches it before you ever run the code.
