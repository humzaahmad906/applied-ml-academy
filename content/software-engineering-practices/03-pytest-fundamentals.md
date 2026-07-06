# 03 — Testing with pytest: The Fundamentals

Writing code is only half the job; knowing it actually works is the other half. Tests are small programs that run your code and check that it does what you expect. They catch bugs before your users do, they let you change and improve code without fear of quietly breaking something, and they double as living documentation of how your code is meant to behave. This lesson introduces **pytest**, the standard testing tool in modern Python, and shows you how to write your first tests with almost no ceremony.

## Why write tests at all

It is tempting to test by hand — run the program, eyeball the output, move on. That works once. But every time you change the code, you would have to re-check everything by hand, and you won't. Automated tests do that checking for you, instantly, every time. Three concrete payoffs:

- **Catch bugs early.** A failing test tells you something broke the moment it breaks, not weeks later in production.
- **Refactor without fear.** With tests guarding behavior, you can restructure code confidently — if you break something, a test goes red.
- **Document behavior.** A well-named test is a precise, executable statement of what the code should do. New teammates read tests to understand intent.

## pytest basics

pytest is refreshingly low-ceremony. A test is just a function whose name starts with `test_`, containing a plain `assert`. No classes, no boilerplate, no special assertion methods to memorize.

Suppose you have this function in `src/mypackage/math_utils.py`:

```python
def add(a, b):
    return a + b
```

A test for it lives in `tests/test_math_utils.py`:

```python
from mypackage.math_utils import add


def test_add_positive_numbers():
    assert add(2, 3) == 5


def test_add_with_zero():
    assert add(5, 0) == 5
```

That is a complete test file. Each function checks one thing with `assert`. When the expression after `assert` is true, the test passes; when it is false, pytest reports a failure — and it is smart enough to show you both the expected and actual values, so you rarely need to add a message.

## Running your tests

You run tests by typing `pytest` at the terminal (or `uv run pytest`, as you saw in Lesson 02). From your project root:

```bash
pytest
```

pytest discovers every `test_*.py` file and every `test_` function inside them, runs them all, and prints a summary of passes and failures. For more detail, add `-v` (verbose), which lists each test by name:

```bash
pytest -v
```

To run a single test while you focus on it, name the file and the function:

```bash
pytest tests/test_math_utils.py::test_add_with_zero
```

## Where tests live

Put your tests in a `tests/` directory that mirrors your source layout. If your package has `math_utils.py` and `core.py`, your tests folder has `test_math_utils.py` and `test_core.py`. This one-to-one mirroring makes it obvious where the tests for any given module live, and where to add new ones.

When several tests need the same setup — a sample dataset, a temporary file — you put that shared setup in a special file called `conftest.py` at the root of your tests folder. pytest loads it automatically, and any fixtures defined there are available to all your tests. You will learn to write fixtures in Lesson 04; for now, just know that `conftest.py` is where shared test scaffolding goes.

## What to test

You cannot test everything, and you shouldn't try. Aim your effort where it pays off:

- **Pure functions first.** Functions that take inputs and return outputs with no side effects are the easiest and most valuable to test — start there.
- **Edge cases.** Empty lists, zero, negative numbers, very large values, `None`. Bugs love the boundaries.
- **Error paths.** Check that bad input raises the error you expect, not just that good input works.

A helpful way to structure each test is **given / when / then**: given some starting state, when you call the code, then a specific result should hold. You do not write those words as comments — you just let the test's shape follow that arc.

```python
import pytest
from mypackage.math_utils import safe_divide


def test_safe_divide_normal():
    # given two numbers, when we divide, then we get the quotient
    assert safe_divide(10, 2) == 5


def test_safe_divide_by_zero_raises():
    # given a zero divisor, when we divide, then it raises
    with pytest.raises(ZeroDivisionError):
        safe_divide(10, 0)
```

The `pytest.raises` context manager is how you test error paths: the test passes only if the code inside the `with` block raises the named exception.

## A word on test-driven development

Some developers write the test *before* the code: write a failing test that describes what you want, watch it fail, write just enough code to make it pass, then clean up. This cycle — red, green, refactor — is called **test-driven development (TDD)**. It is a useful discipline that keeps you focused on behavior rather than implementation, and it guarantees your code is testable by construction. Treat it as a tool you reach for when it helps, not a rule you must obey everywhere. Writing the test right after the code, or alongside it, is perfectly good practice too.

## Key takeaways

- Tests catch bugs early, let you refactor without fear, and document how your code should behave.
- In pytest a test is just a `test_`-prefixed function with a plain `assert` — no classes or boilerplate.
- Run everything with `pytest`, add `-v` for detail, or target one test with `path::test_name`.
- Keep tests in a `tests/` directory that mirrors your source; put shared setup in `conftest.py`.
- Test pure functions first, then edge cases and error paths; structure each test as given / when / then.
- Use `with pytest.raises(SomeError):` to assert that bad input raises the expected exception.
- TDD (write the failing test first) is a helpful discipline, not a religion.

## Try it

Take the `add` function from Lesson 01's project (or write a small `math_utils.py` with `add` and a `safe_divide` that raises on a zero divisor). Create `tests/test_math_utils.py` and write four tests: two for `add` covering a normal case and an edge case (adding zero or a negative), and two for `safe_divide` — one checking a normal division and one using `pytest.raises` to confirm dividing by zero raises `ZeroDivisionError`. Run `pytest -v` and read the output. Then deliberately break `add` (make it subtract) and run the tests again to see exactly how pytest reports a failure.
