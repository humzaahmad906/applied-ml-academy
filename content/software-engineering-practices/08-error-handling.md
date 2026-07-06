# 08 — Error Handling and Exceptions

In the Python Foundations course you met `try`/`except` and learned that it stops a program from crashing when something goes wrong. That was the introduction. This lesson goes deeper, because *how* you handle errors is one of the clearest signals of whether code is production-grade. Good error handling is not about suppressing crashes — it is about failing in the right place, with the right information, so that a problem is easy to diagnose instead of silently corrupting your results.

## Never write a bare except

The single most damaging habit in Python error handling is the bare `except`:

```python
# never do this
try:
    value = compute()
except:
    value = None
```

This catches *everything* — not just the error you were worried about, but also typos in your own code, keyboard interrupts, and out-of-memory conditions. It hides bugs. If `compute` fails because you misspelled a variable name, this code quietly hands you `None` and marches on, and you spend an afternoon wondering why your results are wrong.

Always name the exception you actually expect:

```python
try:
    value = int(user_input)
except ValueError:
    value = 0
```

You can catch several related types in one tuple:

```python
try:
    result = data[key] / count
except (KeyError, ZeroDivisionError) as err:
    logger.warning("Could not compute result: %s", err)
    result = None
```

The rule is: catch the *specific* thing you know how to handle, and let everything else propagate. An exception you did not anticipate is a message from your program telling you something is wrong — do not gag it.

## The exception hierarchy

Exceptions in Python form a tree. At the very top sits `BaseException`. Below it sits `Exception`, and below *that* sit all the everyday errors: `ValueError`, `KeyError`, `TypeError`, `FileNotFoundError`, and so on.

```
BaseException
├── SystemExit          (raised by sys.exit)
├── KeyboardInterrupt   (raised by Ctrl-C)
└── Exception
    ├── ValueError
    ├── KeyError
    ├── TypeError
    ├── OSError
    │   └── FileNotFoundError
    └── ... everything you normally deal with
```

This shape matters. `SystemExit` and `KeyboardInterrupt` deliberately sit *outside* `Exception`, so that catching `Exception` does not accidentally trap a user pressing Ctrl-C or a clean shutdown. This is exactly why `except Exception:` is acceptable as a last resort but bare `except:` (which catches `BaseException`) is not. **Only ever catch `Exception` or its subclasses.**

Because the hierarchy is a tree, catching a parent catches its children too. `except OSError:` will also catch `FileNotFoundError`. Catch as narrowly as the situation allows.

## Custom exceptions

The built-in exceptions cover generic failures, but your program has failures that are specific to *it*. When a model file is missing, `FileNotFoundError` is technically true but tells the caller nothing about your domain. Define your own exception type:

```python
class ModelNotFoundError(Exception):
    """Raised when a requested model cannot be located or loaded."""


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise ModelNotFoundError(f"No model named {name!r} in {MODELS_DIR}")
    return _deserialize(path)
```

A custom exception is just a class inheriting from `Exception`; often the whole body is a docstring. The value is that it gives callers *something meaningful to catch*:

```python
try:
    model = load_model("sentiment-v3")
except ModelNotFoundError:
    model = load_model("sentiment-default")
```

Create custom exceptions when an error crosses a module boundary — when code in one part of your program needs to react specifically to a failure raised elsewhere. Do not invent a new exception type for every conceivable error; invent one when a caller genuinely needs to distinguish *your* failure from all others.

## Fail loud

Beginners often "handle" errors by swallowing them: catching a problem and returning `None` so the program keeps running. This feels safe and is usually a trap. A `None` returned from deep inside your code travels far before it causes a visible failure, and by then the original cause is long gone.

The healthier instinct is to **fail loud**: when something unexpected happens, `raise` immediately, close to where the problem occurred, with a clear message. Handle the errors you genuinely know how to recover from; for everything else, let the exception fly. A program that crashes with a precise traceback at the true source is far easier to fix than one that limps along producing wrong answers.

## try / except / else / finally

The full statement has four blocks, and each has a distinct job:

```python
try:
    f = open(path)
    data = json.load(f)
except FileNotFoundError:
    logger.error("Config file missing: %s", path)
    raise
except json.JSONDecodeError as err:
    logger.error("Config is not valid JSON: %s", err)
    raise
else:
    logger.info("Config loaded with %d keys", len(data))
finally:
    f.close()
```

`try` holds the risky code. Each `except` handles one specific failure. The `else` block runs *only if no exception was raised* — put the "success path" work here, so it is clearly separated from the risky part and is not itself wrapped by the `try`. The `finally` block runs *no matter what* — exception or not — and is for cleanup that must always happen, like closing a file or releasing a lock.

## Context managers do finally for you

The `finally`-for-cleanup pattern is so common that Python has dedicated syntax: the `with` statement, backed by context managers.

```python
with open(path) as f:
    data = json.load(f)
# f is closed automatically, even if json.load raises
```

The file is guaranteed to close whether the block finishes normally or blows up. Any resource that needs releasing — files, database connections, network sockets, locks — should be managed with `with` rather than manual `try`/`finally`. It is shorter and impossible to forget.

## Logging exceptions

When you catch an exception and want a record of it, do not just log the message — log the full traceback. The logging module has a method built for exactly this:

```python
try:
    run_pipeline()
except Exception:
    logger.exception("Pipeline failed")
    raise
```

`logger.exception(...)` logs at ERROR level *and* attaches the complete traceback automatically. It only works inside an `except` block. Notice the `raise` on the last line: this logs the failure and then re-raises it so callers still see it — recording an error is not the same as pretending it did not happen.

## Connection to the APIs course

This "catch a specific type, respond meaningfully" pattern reappears when you build web APIs. In FastAPI, you signal an error to a client by raising `HTTPException(status_code=404, detail="Model not found")`. That is the same idea as a custom exception: a specific, catchable, meaningful failure — here translated into an HTTP response. You will see this in the APIs course; the discipline you build now carries straight over.

## Key takeaways

- Never use a bare `except:` — it hides bugs and even traps Ctrl-C. Catch the specific type(s) you expect.
- Exceptions form a tree under `BaseException`; only ever catch `Exception` or its subclasses, and catch as narrowly as you can.
- Define custom exceptions (subclassing `Exception`) when a failure crosses a module boundary and callers need something meaningful to catch.
- Fail loud: `raise` on the unexpected instead of silently returning `None`.
- `else` runs only on success; `finally` always runs — use it (or better, a `with` context manager) for cleanup.
- Use `logger.exception(...)` inside an `except` block to capture the full traceback, and re-`raise` if you are not truly recovering.

## Try it

Write a function `load_config(path)` that opens a JSON file and returns its contents as a dict. Define a custom exception `ConfigError(Exception)`. Inside the function, use `try`/`except`/`else`/`finally` (or a `with` block) to handle two cases distinctly: a missing file and a file that contains invalid JSON. In each case, log the problem with `logger.exception(...)` and then `raise ConfigError(...)` with a clear message that names the path. Call it three times — once with a valid file, once with a path that does not exist, once with a file containing broken JSON — and observe how each failure produces a precise, informative traceback instead of a silent `None`.
