# 04b — Docstrings and Test Doubles

Two skills separate code that others can use from code that only you can babysit: documenting what a function does, and testing it without dragging in the whole world. This lesson pairs them. First, **docstrings** — the in-code documentation that explains a function's intent, its arguments, and what it hands back. Then **test doubles** — the mocks and fakes that let you test code touching networks, clocks, or paid APIs without actually calling them. Both are about drawing clean boundaries: a docstring documents the contract at a boundary, and a mock stands in for whatever sits on the other side of it.

## Why docstrings matter

A type hint tells you a parameter is a `float`. It does not tell you the float is a probability between 0 and 1, that passing a negative raises `ValueError`, or that the return value is rounded to two places. That intent lives in the **docstring** — the string literal directly under a `def` or `class`. It is not a comment; Python stores it as `func.__doc__`, `help(func)` prints it, IDEs surface it on hover, and documentation generators read it to build your API reference. A comment rots in the file; a docstring is part of the callable's interface.

The rule of thumb: **type hints say what the shape is, docstrings say what it means.** Together they are the documentation. Neither alone is enough.

## The Google style

Two docstring conventions dominate. The **Google style** is the more compact and, for most people, the more readable. Sections are named headers followed by indented entries:

```python
def normalize(scores: list[float], cap: float = 1.0) -> list[float]:
    """Scale scores so the largest equals ``cap``.

    Args:
        scores: Raw non-negative scores. Must contain at least one value.
        cap: The value the maximum score maps to. Defaults to 1.0.

    Returns:
        A new list of scaled scores, same length as the input.

    Raises:
        ValueError: If ``scores`` is empty or all zeros.
    """
    peak = max(scores)
    if peak == 0:
        raise ValueError("scores must contain a non-zero value")
    return [s / peak * cap for s in scores]
```

The opening line is a one-sentence summary in the imperative mood ("Scale...", not "Scales..."). Then `Args`, `Returns`, and `Raises` document the contract. Notice you do **not** repeat the types — the signature already has them. You document meaning: "non-negative", "at least one value", "defaults to 1.0".

## The NumPy style

The **NumPy style** uses underlined section headers and is favored across the scientific Python stack (NumPy, SciPy, pandas, scikit-learn). It is more verbose but scans well for functions with many parameters:

```python
def normalize(scores, cap=1.0):
    """Scale scores so the largest equals ``cap``.

    Parameters
    ----------
    scores : list of float
        Raw non-negative scores. Must contain at least one value.
    cap : float, optional
        The value the maximum score maps to (default is 1.0).

    Returns
    -------
    list of float
        A new list of scaled scores, same length as the input.

    Raises
    ------
    ValueError
        If ``scores`` is empty or all zeros.
    """
    ...
```

Pick one style per project and stay consistent — mixing them makes doc tooling produce ragged output. Google style pairs naturally with type hints (types in the signature, not the docstring); NumPy style predates ubiquitous hints and often restates the type. For a modern, hinted codebase, Google style is the lighter fit.

## From docstrings to published docs

Docstrings are not just for `help()`. Documentation generators harvest them into browsable sites. The two mainstays are **Sphinx** (with the `napoleon` extension, which teaches it to read Google and NumPy styles) and **MkDocs** with the **mkdocstrings** plugin, which renders a Markdown site straight from your docstrings and signatures. The payoff is concrete: write the docstring once, next to the code, and your published API reference stays in sync automatically. That is only true if the docstring is accurate — a stale docstring is worse than none, because it lies with authority.

## Why you mock

Now the second half. A good unit test is fast, deterministic, and isolated. Real-world dependencies are none of those. If your function calls a payment API, hits a database, reads the system clock, or draws a random number, a test that runs it for real is slow, flaky, and possibly expensive. You mock to cut those dependencies at the boundary:

- **Network** — don't make a real HTTP call; the endpoint may be down, rate-limited, or paid.
- **Filesystem / database** — don't depend on files or rows that may not exist on CI.
- **Time** — freeze `datetime.now()` so a test is not different at midnight.
- **Randomness** — pin the value so "pick a random winner" is checkable.
- **Paid or side-effecting APIs** — never charge a card or send an email from a test.

A **test double** is any stand-in for a real dependency. A **mock** is a double that also records how it was called so you can assert on it. This picks up where Lesson 03's unit tests and Lesson 04's fixtures left off: fixtures give you sample data, mocks give you sample *behavior*.

## unittest.mock: Mock and MagicMock

The standard library ships `unittest.mock`. A `Mock` is an object that invents attributes and methods on access and remembers every call. `MagicMock` is the same but also supports dunder methods (`__len__`, `__iter__`, `__enter__`), so it works as a context manager or iterable. Use `MagicMock` unless you have a reason not to.

You configure a return value and then assert on the calls:

```python
from unittest.mock import MagicMock


def test_mock_records_calls():
    client = MagicMock()
    client.fetch.return_value = {"status": "ok"}

    result = client.fetch("/users", limit=10)

    assert result == {"status": "ok"}
    client.fetch.assert_called_once_with("/users", limit=10)
    # output: passes
```

`assert_called_once_with` is the workhorse: it checks the mock was called exactly once **and** with exactly those arguments. Its relatives are `assert_called_with` (last call), `assert_not_called`, and the `call_count` / `call_args` attributes for finer inspection.

## patch: swap the real thing for a mock

`Mock` is only useful once it replaces the real dependency. `patch` does the swap, temporarily, and puts the original back afterward — as a decorator or a context manager. The critical rule: **patch where the name is looked up, not where it is defined.** If `weather.py` does `import requests` and calls `requests.get`, you patch `"weather.requests.get"`, not `"requests.get"`.

Say `weather.py` contains:

```python
import requests


def current_temp(city: str) -> float:
    """Return the current temperature in Celsius for ``city``.

    Args:
        city: City name to look up.

    Returns:
        Temperature in degrees Celsius.
    """
    resp = requests.get("https://api.example.com/weather", params={"city": city})
    resp.raise_for_status()
    return resp.json()["temp_c"]
```

The test never touches the network:

```python
from unittest.mock import patch, MagicMock
from myapp import weather


@patch("myapp.weather.requests.get")
def test_current_temp(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        **{"json.return_value": {"temp_c": 21.5}},
    )

    temp = weather.current_temp("Lahore")

    assert temp == 21.5
    mock_get.assert_called_once_with(
        "https://api.example.com/weather", params={"city": "Lahore"}
    )
    # output: passes, no real request made
```

The same swap as a context manager, when you want it scoped to a few lines:

```python
def test_current_temp_ctx():
    with patch("myapp.weather.requests.get") as mock_get:
        mock_get.return_value = MagicMock(**{"json.return_value": {"temp_c": 5.0}})
        assert weather.current_temp("Oslo") == 5.0
```

## monkeypatch: pytest's built-in for env vars and attributes

pytest ships a `monkeypatch` fixture for the common, lighter cases: setting environment variables, swapping an attribute, changing the working directory. Its advantage over `patch` is that pytest undoes every change automatically at the end of the test, so you never leak state:

```python
def get_api_key() -> str:
    """Return the API key from the environment, or raise if unset."""
    import os

    key = os.environ.get("API_KEY")
    if not key:
        raise RuntimeError("API_KEY not set")
    return key


def test_get_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key-123")
    assert get_api_key() == "test-key-123"
    # output: passes

    monkeypatch.setattr("random.random", lambda: 0.42)
    import random
    assert random.random() == 0.42
```

Reach for `monkeypatch` for env vars and simple attribute swaps; reach for `patch` when you want a full `Mock` with call assertions.

## Faking HTTP: responses and respx

For code that makes HTTP calls, hand-building mock response objects gets tedious. Two libraries do it cleanly by intercepting at the transport layer. **`responses`** targets the `requests` library; **`respx`** targets `httpx` (which underlies most modern async clients, including the OpenAI SDK). They let you register a URL and the canned response it should return, so your code under test runs unmodified:

```python
import responses
import requests


@responses.activate
def test_with_responses():
    responses.add(
        responses.GET,
        "https://api.example.com/weather",
        json={"temp_c": 18.0},
        status=200,
    )

    resp = requests.get("https://api.example.com/weather", params={"city": "Paris"})

    assert resp.json()["temp_c"] == 18.0
    assert len(responses.calls) == 1
    # output: passes
```

The `respx` equivalent for an `httpx` client uses the `respx_mock` pytest fixture:

```python
import httpx


def test_with_respx(respx_mock):
    respx_mock.get("https://api.example.com/weather").mock(
        return_value=httpx.Response(200, json={"temp_c": 18.0})
    )

    resp = httpx.get("https://api.example.com/weather")

    assert resp.json()["temp_c"] == 18.0
    # output: passes
```

These beat raw `patch` for HTTP because you describe the *wire response* (status, headers, JSON body) instead of reverse-engineering the client's internal object graph. That makes the test robust to how the client is implemented.

## The discipline: mock at the boundary, and no further

The most common mocking mistake is mocking too much. Two rules keep you honest.

**Mock at the boundary.** Replace the thing that leaves your process — the HTTP call, the DB cursor, the clock — and nothing inside it. If you mock your own business logic, you are no longer testing it; you have tested that your mock returns what you told it to, which proves nothing.

**Test behavior, not implementation.** A test that asserts "method A called helper B which called helper C" breaks the moment you refactor the internals, even when the observable behavior is identical. That is a brittle test. Assert on the *result* and on the *calls that cross the boundary* — did we hit the right URL with the right payload — not on private wiring. `assert_called_with` is the right tool for boundary calls; using it on every internal call is a smell. When a test needs a dozen mocks to run, that is usually the code telling you the function has too many dependencies, not that you need more mocks.

## Key takeaways

- A docstring is part of a callable's interface (`__doc__`, `help()`, doc generators) — not a comment. Type hints give the shape; docstrings give the meaning.
- Google style (`Args:` / `Returns:` / `Raises:`) is compact and pairs well with type hints; NumPy style (underlined headers) suits the scientific stack. Pick one per project.
- Sphinx (`napoleon`) and MkDocs (`mkdocstrings`) harvest docstrings into a published API reference that stays in sync with the code.
- Mock to isolate tests from network, filesystem, time, randomness, and paid APIs — anything slow, flaky, or side-effecting.
- `MagicMock` records calls and invents attributes; `patch` swaps the real dependency for it — patch where the name is *looked up*, not where it is defined.
- Use pytest's `monkeypatch` fixture for env vars and attribute swaps; it auto-reverts.
- `responses` (for `requests`) and `respx` (for `httpx`) fake HTTP at the transport layer by describing the wire response.
- Mock at the boundary only; test observable behavior with `assert_called_with` on boundary calls, not private wiring. Needing many mocks signals too many dependencies.

## Try it

Take a function that calls an external HTTP API (write a small one if you don't have it, like the `current_temp` example above). First, give it a complete Google-style docstring with `Args`, `Returns`, and `Raises`, and confirm `help(your_func)` prints it. Then write three tests: one using `@patch` to mock the HTTP client with a `MagicMock` and `assert_called_once_with` on the URL and params; one using `monkeypatch.setenv` to supply a fake API key; and one using `responses` (or `respx` if your code uses `httpx`) to fake the wire response. Finally, look at your `@patch` test and check you are asserting on the boundary call and the returned result — not on any internal helper. See Lessons 03 and 04 for the unit-testing and fixture foundations these build on.
