# 04 — Fixtures, Parametrize, and Test Organization

Once you have written a handful of tests (Lesson 03), patterns start to repeat. Several tests need the same sample data. The same logic needs checking against a dozen different inputs. Some tests are slow and you only want to run them sometimes. pytest has clean answers for all three: **fixtures** for shared setup, **parametrize** for many inputs, and **markers** for selective runs. This lesson shows you how to keep a growing test suite organized and free of copy-paste.

## Fixtures: shared setup

When multiple tests need the same starting object — a sample dataset, a configured client, a temporary file — you do not want to build it inside every test. A **fixture** is a function that produces that object, marked with `@pytest.fixture`. Any test that names the fixture as a parameter receives its value:

```python
import pytest


@pytest.fixture
def sample_scores():
    return [90, 85, 72, 100, 60]


def test_average(sample_scores):
    assert sum(sample_scores) / len(sample_scores) == 81.4


def test_max(sample_scores):
    assert max(sample_scores) == 100
```

pytest sees that both tests ask for `sample_scores`, calls the fixture, and passes the result in. The setup is written once and reused everywhere it is named.

## Cleanup with yield fixtures

Some fixtures set up something that must be torn down afterward — a temporary file, a database connection. For these, use `yield` instead of `return`. Everything before the `yield` is setup; everything after runs as cleanup once the test finishes, pass or fail:

```python
import pytest


@pytest.fixture
def temp_data_file(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("1,2,3")
    yield path
    # cleanup runs here after the test
    path.unlink()
```

Here `tmp_path` is a built-in pytest fixture that hands you a unique temporary directory, so you never litter your real filesystem. Our fixture writes a file, yields its path to the test, and deletes it afterward. (With `tmp_path`, pytest cleans up the directory itself, but the pattern is what matters.)

## Fixture scopes

By default a fixture runs once per test that uses it — that is `function` scope. If a fixture is expensive to build and safe to share, you can widen its scope so it is created once and reused:

```python
@pytest.fixture(scope="session")
def big_model():
    return load_expensive_model()
```

The common scopes are:

- **function** (default) — rebuilt for every test. Use when tests might mutate the object and must not affect each other.
- **module** — built once per test file. Good for something moderately expensive shared across a file's tests.
- **session** — built once for the entire test run. Reserve for genuinely expensive, read-only resources like a loaded model or a shared connection.

Wider scope is faster but riskier: if one test mutates a shared fixture, it can corrupt the next. Default to `function` and widen only when you have a reason.

## Parametrize: same logic, many inputs

When you find yourself copying a test and changing only the numbers, reach for `@pytest.mark.parametrize`. It runs the same test body once per set of inputs:

```python
import pytest
from mypackage.math_utils import add


@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

This is four tests in one definition. pytest runs the body four times, filling in `a`, `b`, and `expected` from each tuple, and reports each case separately — so if only the negative case fails, you see exactly that. Parametrize is the cure for copy-paste tests and makes adding a new case a one-line change.

## Markers: running a subset

Some tests are slow — they hit a network, load a large file, train something. You do not want them on every quick run. Tag them with a **marker**:

```python
import pytest


@pytest.mark.slow
def test_full_training_run():
    ...
```

Then select or skip by marker at the command line:

```bash
pytest -m slow          # only the slow tests
pytest -m "not slow"    # everything except the slow tests
```

This lets you run a fast subset constantly and the full suite occasionally. (Declare custom markers in `pyproject.toml` under `[tool.pytest.ini_options]` to avoid warnings.)

## Measuring coverage

**Coverage** tells you which lines of your code your tests actually exercise. The `pytest-cov` plugin adds it to pytest:

```bash
pytest --cov=src --cov-report=term-missing
```

`--cov=src` measures coverage of your `src/` code, and `--cov-report=term-missing` prints, per file, the percentage covered and the exact line numbers no test touched — which is far more useful than the percentage alone, because it tells you *what* to test next.

A caution: coverage is a guide, not a goal. Chasing 100% leads to hollow tests that call code without checking anything meaningful — coverage theater. Aim instead for coverage of the code that matters: your core logic, your edge cases, your error paths. A suite at 80% that tests the right things beats one at 100% that tests nothing well.

## Organizing fixtures with conftest.py

You met `conftest.py` in Lesson 03 as the home for shared fixtures. It works at every level of your tests folder. A `conftest.py` at the `tests/` root provides fixtures to the whole suite; a `conftest.py` inside `tests/integration/` provides fixtures only to the tests in that subfolder. This lets you scope fixtures to where they are relevant — broad, common setup at the top, specialized setup nearer the tests that need it — without importing anything by hand. pytest finds and wires them up automatically.

## Key takeaways

- A `@pytest.fixture` function produces shared setup; tests receive it by naming it as a parameter.
- Use `yield` in a fixture to run cleanup after the test; the built-in `tmp_path` fixture gives you a throwaway directory.
- Fixture scope (function / module / session) trades safety for speed — default to function, widen only for expensive, read-only resources.
- `@pytest.mark.parametrize` runs one test body over many inputs, replacing copy-pasted near-duplicate tests.
- Markers like `@pytest.mark.slow` let you run subsets with `pytest -m slow` or `pytest -m "not slow"`.
- `pytest --cov=src --cov-report=term-missing` shows which lines are untested; aim for meaningful coverage, not 100% theater.
- `conftest.py` at different folder levels scopes shared fixtures to where they are needed.

## Try it

Take a test file from Lesson 03 and refactor it. First, pull any repeated sample data into a `@pytest.fixture` and have your tests receive it as a parameter. Next, convert a group of near-identical tests into a single `@pytest.mark.parametrize` case covering at least four input combinations, including an edge case. Add a `@pytest.mark.slow` marker to one test and confirm `pytest -m "not slow"` skips it. Finally, install `pytest-cov` and run `pytest --cov=src --cov-report=term-missing`; look at the missing lines and write one more test that covers a path you had left untested.
