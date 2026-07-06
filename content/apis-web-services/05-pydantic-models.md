# 05 — Request and Response Models with Pydantic

In Lesson 04 your routes accepted simple path and query parameters, but a real ML endpoint receives a *structured body* — a batch of features, a set of options, a nested payload — and you need to be sure that body is well-formed before your model ever sees it. This is what **Pydantic** does, and it is the engine underneath FastAPI's validation. You declare the shape of your data as a class, and Pydantic enforces it, converting good input and rejecting bad input with precise error messages. This lesson uses Pydantic v2 throughout, which is the current version and differs in important ways from the old v1.

## BaseModel as a request body

A Pydantic model is a class that inherits from `BaseModel` and declares fields as annotated attributes. When you use one as a function parameter in FastAPI (reusing the `app` from Lesson 04), it is automatically read from the JSON request body, validated, and handed to you as a fully-typed object:

```python
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    model_name: str
    features: list[float]

@app.post("/predictions", status_code=201)
def create_prediction(request: PredictionRequest) -> dict:
    return {"model": request.model_name, "n_features": len(request.features)}
```

FastAPI sees that `request` is a `BaseModel` subclass, so it parses the JSON body into a `PredictionRequest`. If the caller omits `model_name` or sends `features` as strings that cannot become floats, FastAPI returns a `422` with a detailed explanation — your function only ever runs on valid data, and you access fields as normal attributes: `request.model_name`.

## Field types and constraints

Beyond basic types, `Field()` lets you attach **constraints** that Pydantic enforces. Common ones are `ge`/`le` (greater/less than or equal) for numbers and `min_length`/`max_length` for strings and lists:

```python
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(ge=1, default=256)
```

Now `temperature=3.0` is rejected before your code sees it, and `name` cannot be empty; `default=256` makes `max_tokens` optional with a fallback. In general, a field is required unless it has a default. To make one genuinely optional — absent or `null` — annotate it `T | None` and default it to `None`:

```python
class ModelUpdate(BaseModel):
    description: str | None = None
    accuracy: float | None = None
```

This is exactly what a PATCH body (Lesson 02) looks like: every field optional, so the caller sends only what they want to change.

## The Annotated pattern

The modern, recommended way to attach constraints is `Annotated`, which pairs a type with its metadata. It reads cleanly and, when named, is reusable across models:

```python
from typing import Annotated
from pydantic import BaseModel, Field

Probability = Annotated[float, Field(ge=0.0, le=1.0)]

class Prediction(BaseModel):
    label: str
    confidence: Probability
```

`Annotated[float, Field(ge=0, le=1)]` says "a float that must be between 0 and 1." Naming it (`Probability`) lets you reuse the same constrained type everywhere. This is the preferred v2 style over putting `Field(...)` as a default value.

## Validators for custom rules

Constraints cover ranges and lengths, but some rules need logic. Use `@field_validator` for a rule about a **single field**:

```python
from pydantic import field_validator

class PredictionRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("features must not be empty")
        return v
```

The method (a `@classmethod` in v2) receives the field's value and either returns it, possibly transformed, or raises `ValueError`, which Pydantic converts into a validation error.

For a rule that spans **several fields**, use `@model_validator(mode="after")`, which runs after the whole model is built so you can compare fields against each other:

```python
from pydantic import model_validator

class Range(BaseModel):
    low: float
    high: float

    @model_validator(mode="after")
    def check_order(self) -> "Range":
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self
```

With `mode="after"`, the validator receives the fully-constructed instance (`self`), so both `self.low` and `self.high` are available and already type-checked.

## Response models — preventing data leaks

Just as you validate what comes *in*, you should control what goes *out*. Passing `response_model=` to a route tells FastAPI to filter the return value through that model, so **only the declared fields are returned** — anything extra is silently dropped. This is your safeguard against leaking internal or sensitive fields:

```python
class ModelPublic(BaseModel):
    id: int
    name: str
    accuracy: float

@app.get("/models/{model_id}", response_model=ModelPublic)
def get_model(model_id: int) -> dict:
    # internal_path is returned here but dropped from the response
    return {"id": model_id, "name": "resnet50", "accuracy": 0.94, "internal_path": "/secrets/weights.pt"}
```

Even though the function returns `internal_path`, the caller never sees it — `ModelPublic` does not declare it, so FastAPI strips it out. This separation of input models, output models, and internal data is a core habit of safe API design.

## Nested models

Models compose: a field can be another model, and Pydantic validates the whole tree, mirroring how JSON nests (Lesson 02):

```python
class Feature(BaseModel):
    name: str
    value: float

class PredictionRequest(BaseModel):
    model_name: str
    features: list[Feature]
```

A request body like `{"model_name": "x", "features": [{"name": "age", "value": 30}]}` is validated all the way down — each entry in `features` must be a valid `Feature`.

## model_dump, not dict

When you need a plain dict from a model — to log it, store it, or send it onward — call `model_dump()`. In Pydantic v2 the old `.dict()` method is deprecated:

```python
request = PredictionRequest(model_name="x", features=[Feature(name="age", value=30)])
payload = request.model_dump()     # -> a plain dict, ready to serialize
```

If you see `.dict()` in a tutorial, it is Pydantic v1 and out of date — use `model_dump()` everywhere.

## Key takeaways

- A Pydantic `BaseModel` used as a FastAPI parameter is auto-parsed from the JSON body, validated, and given to you fully typed.
- `Field()` adds constraints like `ge`, `le`, `min_length`, and `max_length` that produce automatic `422`s on bad input.
- Make a field optional by annotating it `T | None` with a `None` default.
- Prefer the `Annotated[float, Field(ge=0, le=1)]` pattern for constraints, and name reusable constrained types.
- Use `@field_validator` for single-field rules and `@model_validator(mode="after")` for cross-field rules that need the whole instance.
- Set `response_model=` on a route so only declared fields are returned — your defense against leaking internal data.
- Models nest to validate arbitrarily deep JSON; convert a model to a dict with `model_dump()`, never the deprecated `.dict()`.

## Try it

Extend the model-registry API from Lesson 04. Define a `RegisterModel` request model with a non-empty `name`, a `framework` string, and an `accuracy` field constrained to the range 0–1 using the `Annotated` + `Field` pattern. Add a `@model_validator(mode="after")` that rejects any model whose name contains a space. Then define a separate `ModelPublic` response model that omits an internal field (say, a storage path your handler includes), wire it up with `response_model=`, and confirm through `/docs` that the secret field never appears in the response. Finally, send an accuracy of `1.5` and read the `422` Pydantic returns.
