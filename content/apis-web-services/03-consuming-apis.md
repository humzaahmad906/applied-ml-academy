# 03 — Consuming APIs from Python

You have seen HTTP on the wire with `curl` and learned how REST APIs are shaped. Now you will call them from Python, which is what your data pipelines and clients actually do. This lesson covers the `requests` library — the long-standing default — and `httpx`, the modern alternative, and it spends real time on the things that separate a toy script from production code: timeouts, retries, and pagination. These are the habits that keep an overnight data-fetching job from silently hanging or losing half its results.

## The basics with requests

`requests` gives you one function per HTTP method. A GET returns a response object you can inspect:

```python
import requests

resp = requests.get("https://api.github.com/users/torvalds")
print(resp.status_code)   # 200
data = resp.json()        # parse the JSON body into a dict
print(data["public_repos"])
```

`resp.status_code` is the number you learned about in Lesson 01. `resp.json()` parses the JSON body into Python objects — it is the `json.loads` step done for you. A POST works the same way, and you pass a body with the `json=` argument, which serializes a dict and sets `Content-Type: application/json` automatically:

```python
resp = requests.post(
    "https://httpbin.org/post",
    json={"model": "resnet50", "input": [0.1, 0.4, 0.9]},
)
```

## Check the status — raise_for_status

A common mistake is assuming a response succeeded. A 404 or 500 still comes back as a normal response object; `resp.json()` will happily parse an error body or throw a confusing parse error. Guard every call with `raise_for_status()`, which turns any 4xx or 5xx into an exception you can catch:

```python
resp = requests.get("https://api.github.com/users/torvalds")
resp.raise_for_status()   # raises HTTPError on 4xx/5xx
data = resp.json()
```

The rule: after every request, either check `resp.ok`/`status_code` yourself or call `raise_for_status()`. Never parse a body you have not confirmed is a success.

## Headers and authentication

Most real APIs require a token. You pass headers as a dict, and the standard pattern for API keys is the `Authorization: Bearer <token>` header from Lesson 01:

```python
import os

headers = {"Authorization": f"Bearer {os.environ['API_TOKEN']}"}
resp = requests.get("https://api.example.com/v1/models", headers=headers)
```

Read the token from the environment, never hard-code it into source — secrets in code end up in git history. (Config and secrets get their own treatment in the Software Engineering course.)

## Always set a timeout

By default, `requests` will wait *forever* for a response. If the server hangs, your script hangs with it — the classic cause of a data job that is "still running" at 3 a.m. but has actually been frozen for hours. Always pass a `timeout`:

```python
resp = requests.get(url, timeout=(3.05, 30))
```

The tuple is `(connect timeout, read timeout)`. The **connect timeout** caps how long to wait to establish the connection; the **read timeout** caps how long to wait between bytes once connected. A single number applies to both. There is no sensible default of "wait forever," so treat `timeout` as mandatory on every call.

## Retry logic

Transient failures happen — a 503, a dropped connection, a rate-limit 429. Rather than let one blip kill the job, retry with backoff. `requests` does this cleanly by mounting an `HTTPAdapter` configured with `urllib3`'s `Retry`:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

retry = Retry(
    total=5,
    backoff_factor=0.5,                     # sleeps 0.5s, 1s, 2s, 4s...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "PUT", "DELETE"],  # idempotent only, by default
)
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry))

resp = session.get("https://api.example.com/v1/models", timeout=(3.05, 30))
```

Two things to notice. `backoff_factor` spaces retries out exponentially so you do not hammer a struggling server. And `allowed_methods` defaults to idempotent methods only — this is exactly the idempotency lesson from Lesson 01 in action: retrying a POST could create duplicate resources, so it is excluded unless you explicitly opt in.

## Pagination — draining all pages

APIs return long collections in pages (Lesson 02). To get everything, you loop until there are no more pages. The most common style is **offset/limit**, where you advance a page number until you get an empty result:

```python
def fetch_all_repos(user: str) -> list[dict]:
    repos: list[dict] = []
    page = 1
    while True:
        resp = requests.get(
            f"https://api.github.com/users/{user}/repos",
            params={"per_page": 100, "page": page},
            timeout=(3.05, 30),
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:            # empty page means we are done
            break
        repos.extend(batch)
        page += 1
    return repos
```

`params=` builds the query string (`?per_page=100&page=1`) for you. The loop keeps asking for the next page and stops when a page comes back empty. The other common style is **cursor** pagination, where each response includes a token pointing at the next page; you pass it back until the token is absent:

```python
cursor = None
while True:
    params = {"limit": 100}
    if cursor:
        params["cursor"] = cursor
    resp = requests.get(url, params=params, timeout=(3.05, 30))
    resp.raise_for_status()
    body = resp.json()
    process(body["items"])
    cursor = body.get("next_cursor")
    if not cursor:
        break
```

Cursor pagination is more robust when the underlying data is changing, because it does not rely on stable numeric offsets.

## httpx — the modern alternative

`httpx` is a newer client with an API deliberately close to `requests`, plus support for HTTP/2 and async. For synchronous code the switch is nearly cosmetic:

```python
import httpx

resp = httpx.get("https://api.github.com/users/torvalds", timeout=10.0)
resp.raise_for_status()
data = resp.json()
```

The real payoff is `httpx.AsyncClient`, whose `await client.get(...)` lets you fire many requests concurrently — valuable when fetching thousands of records. For everyday scripts, `requests` is perfectly fine and ubiquitous; reach for `httpx` when you want async concurrency or HTTP/2. Either way the mental model — status check, timeout, retry, paginate — is identical.

## Key takeaways

- `requests.get`/`.post` return a response; use `.json()` to parse the body and `.status_code` to inspect the result.
- Call `raise_for_status()` (or check the code yourself) before trusting any response body.
- Pass authentication as an `Authorization: Bearer <token>` header, with the token read from the environment.
- Always set a `timeout` — the tuple form is `(connect, read)`; the default is to wait forever, which is a bug.
- Add retries with `urllib3`'s `Retry` on an `HTTPAdapter`, using exponential backoff and idempotent methods only.
- Drain paginated collections with a loop: offset/limit advances a page number, cursor pagination follows a next-page token.
- `httpx` mirrors the `requests` API and adds async and HTTP/2 for high-concurrency fetching.

## Try it

Write a script that fetches *all* public repositories for a GitHub user, following the offset/limit pagination loop above, and prints how many it found along with the name of the one with the most stars. Add `raise_for_status()` and a `timeout` to every call. Then, deliberately point it at a username that does not exist and confirm your error handling surfaces a clear message rather than a raw traceback. As a stretch, rewrite the same fetch using `httpx` and compare how little had to change.
