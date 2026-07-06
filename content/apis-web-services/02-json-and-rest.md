# 02 — JSON and REST: Designing Resource APIs

In Lesson 01 you saw that most API bodies are JSON and that requests name a method and a URL. This lesson turns those raw ingredients into a design. First we look closely at JSON — the data format that flows in and out of nearly every API — and then at REST, the widely-shared convention for organizing an API so that other developers can guess how it works without reading a manual. We will design a small ML model-registry API as we go, so the ideas stay concrete.

## JSON syntax and types

JSON (JavaScript Object Notation) is a plain-text format for structured data. It has just six types, and you already know all of them from Python:

```json
{
  "name": "resnet50",
  "version": 3,
  "accuracy": 0.94,
  "is_production": true,
  "tags": ["vision", "classification"],
  "owner": null
}
```

The types are: **object** (`{...}`, key-value pairs), **array** (`[...]`, ordered list), **string** (always double-quoted — single quotes are invalid JSON), **number** (integer or float, no distinction), **boolean** (`true`/`false`, lowercase), and **null**. That is the entire language. Its power is that it composes: values inside an object can themselves be objects or arrays, nested as deeply as you like.

## JSON maps onto Python

The reason JSON feels natural in Python is that the two line up almost exactly. A JSON object is a Python `dict`, an array is a `list`, and the scalars map to `str`, `int`/`float`, `bool`, and `None`. The standard library's `json` module does the translation both ways:

```python
import json

data = {"name": "resnet50", "version": 3, "tags": ["vision"]}
text = json.dumps(data)          # dict -> JSON string
back = json.loads(text)          # JSON string -> dict
```

`dumps` (dump-string) serializes a Python object to JSON text you can send over HTTP; `loads` (load-string) parses text you received back into Python objects. In practice the `requests` and `httpx` libraries do this for you, as you will see in Lesson 03, but it helps to know what is happening underneath.

## REST is a convention, not a protocol

**REST** (Representational State Transfer) is not a piece of software you install or a wire format like HTTP. It is a set of *conventions* for structuring an HTTP API so that it is predictable. The central idea is to model your API as a collection of **resources** — the nouns your system is about — and to act on them using the HTTP methods you already learned.

For our example, the resources are **models** and **predictions**. Notice those are nouns, not actions.

## Nouns as URLs, verbs as methods

The single most important REST rule: **URLs name resources (nouns), and the HTTP method says what to do to them (the verb).** You do not put the verb in the URL. So instead of `/getModel` or `/createPrediction`, you design paths like this:

```
GET    /models              # list all models
GET    /models/{id}         # fetch one model by id
POST   /models              # register a new model
PATCH  /models/{id}         # update part of a model
DELETE /models/{id}         # remove a model
POST   /predictions         # run a new prediction
```

A few conventions worth internalizing. Use **plural nouns** for collections (`/models`, not `/model`). A bare collection path (`/models`) with GET lists everything and with POST creates a new member. A path with an identifier (`/models/42`) addresses one specific member. And you never write a verb like `/models/42/delete` — the DELETE method already carries that meaning.

## Path params vs query params

There are two ways to put information into a URL, and they have different jobs.

A **path parameter** identifies *which* resource you mean. It is part of the resource's address:

```
GET /models/42          # 42 is a path param — the identity of the model
```

A **query parameter** modifies *how* you view a collection — filtering, sorting, or paginating. It comes after a `?` as `key=value` pairs joined by `&`:

```
GET /models?framework=pytorch&page=2&limit=20
```

Here `framework` filters the list down to PyTorch models, and `page`/`limit` control **pagination** — asking for the second page of 20 results rather than all of them at once. Pagination exists because a collection might have ten thousand entries and no one wants them in a single response. The rule of thumb: path params for *identity*, query params for *options on a collection*.

## Idempotency and safety in design

The method properties from Lesson 01 shape your design directly. **POST is not idempotent**, so it is the right verb for "create a new prediction" — each call should produce a new one. **PUT is idempotent** because it replaces the whole resource with exactly what you send: run it once or five times and the resource ends up identical. This is the real difference between the two:

```
POST /predictions        # each call creates a NEW prediction
PUT  /models/42          # sets model 42 to exactly this state, repeatable
```

Choosing correctly means callers (and retry logic) can reason about what happens when a request is repeated. If registering a model must not create duplicates when retried, you might design it as a PUT keyed on the model's name; if every call genuinely means "make a new thing," POST is right.

## Pragmatic RESTful design

Purists have long arguments about what counts as "truly RESTful." For building ML services, you do not need to win those arguments. Aim for **pragmatic REST**: plural noun collections, the right method for each action, path params for identity, query params for filtering and pagination, and JSON bodies. An API that follows those few rules is one that any developer — or any large language model wiring up a client — can navigate on the first try. That predictability is the entire point.

## Key takeaways

- JSON has six types (object, array, string, number, boolean, null) and maps almost directly onto Python's dict, list, str, number, bool, and None.
- `json.dumps` serializes Python to JSON text; `json.loads` parses it back.
- REST is a convention for organizing an HTTP API, not a protocol or a library.
- Model your API as resources (nouns) in URLs, and use HTTP methods (verbs) to act on them — never put verbs in paths.
- Use plural nouns for collections; a path with an id addresses one member.
- Path params identify *which* resource; query params filter, sort, and paginate a collection.
- POST creates (not idempotent); PUT replaces (idempotent) — choose based on how repeats should behave.

## Try it

Design, on paper, a small REST API for a dataset registry. Decide on your resources (datasets, and perhaps versions within a dataset) and write out the full set of endpoints — method plus path — for listing, fetching one, creating, updating, and deleting. Add at least one query-parameter example for filtering (say, by format) and one for pagination. For each endpoint, note whether it is safe, idempotent, or neither, and justify your choice of POST versus PUT for creation. Then write a sample JSON request body and a sample JSON response body for the "create a dataset" endpoint.
