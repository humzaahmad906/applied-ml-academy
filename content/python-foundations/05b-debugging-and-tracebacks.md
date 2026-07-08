# 05b — Debugging and Reading Tracebacks

Every working programmer spends more time diagnosing broken code than writing new code, and beginners spend the most of all. The previous lesson showed how to *handle* errors you expect with `try` and `except`. This lesson is about the other half: what to do when an error you did *not* expect stops your program cold. When that happens, Python hands you a traceback and, later, a debugger. Learning to read one and drive the other is the single most valuable day-to-day skill you will build.

## A traceback is a map, not noise

When an unhandled error stops your program, Python prints a block of text called a traceback. It looks intimidating, so beginners often skim it, panic, and start changing random lines. That is the worst thing you can do. A traceback is a precise map of exactly where and why your program failed. You just have to read it in the right direction.

The trick: **read a traceback from the bottom up.** The last line tells you *what* went wrong. The lines above it tell you *where*, tracing the chain of function calls that led to the failure, most recent last.

Here is a real one. Suppose you run this program:

```python
def get_price(item, prices):
    return prices[item]

def total_cost(cart, prices):
    total = 0
    for item in cart:
        total += get_price(item, prices)
    return total

prices = {"apple": 30, "bread": 25}
cart = ["apple", "bread", "milk"]
print(total_cost(cart, prices))
```

Running it produces:

```
Traceback (most recent call last):
  File "shop.py", line 12, in <module>
    print(total_cost(cart, prices))
  File "shop.py", line 7, in total_cost
    total += get_price(item, prices)
  File "shop.py", line 2, in get_price
    return prices[item]
KeyError: 'milk'
```

Read the last line first: `KeyError: 'milk'`. So a dictionary lookup failed on the key `'milk'`. Now walk *up* the stack to find where. The bottom frame, `line 2, in get_price`, is where the error actually fired: `return prices[item]`. The frame above shows who called it (`total_cost`, line 7), and the top frame shows where that started (`line 12`, the `print`). The culprit line is `return prices[item]`, and the root cause is that `cart` contains `"milk"`, which is not in `prices`. You did not have to guess at all; the map told you.

The phrase "most recent call last" at the top is your reminder: the deepest, most immediate cause is at the bottom, right above the error message.

## The common exceptions, decoded

Most errors you hit as a beginner are a handful of types repeating. Knowing what each one usually means turns a scary message into a quick diagnosis.

- **`NameError`** — you used a name Python has never seen. Usually a typo (`prnt` instead of `print`), or using a variable before you assigned it, or forgetting to import something.
- **`TypeError`** — you did an operation on the wrong kind of value, like `"3" + 5` (string plus int) or calling something that is not a function. The message often says exactly which types clashed.
- **`KeyError`** — you asked a dictionary for a key it does not have, as in the example above.
- **`IndexError`** — you asked a list for a position that does not exist, like `items[5]` on a list of three things.
- **`AttributeError`** — you used `.something` on an object that has no such attribute or method. Often the object is `None` when you expected a real value, giving `'NoneType' object has no attribute ...`.
- **`ValueError`** — the type is right but the value is not acceptable, like `int("hello")`. The value cannot be turned into what you asked for.

When you see one of these, read it as a sentence: "`TypeError`: I tried to combine two incompatible things." That framing points you at the fix.

## Print-debugging, done sanely

Before reaching for a debugger, the fastest tool is almost always a well-placed `print`. When you are not sure what a variable holds at some point, print it. But print it *well*: show the value **and** its type, because half of all bugs are "I thought this was a number and it was actually a string."

```python
def total_cost(cart, prices):
    total = 0
    for item in cart:
        print(f"item={item!r}  type={type(item)}")   # what am I looking at?
        total += get_price(item, prices)
    return total
```

The `!r` in the f-string prints the *repr* of the value, so a string shows with its quotes (`'milk'`) and you can tell `"5"` from `5` at a glance. For lists, dictionaries, or arrays, printing the length or shape is often more useful than the whole contents:

```python
print(f"cart has {len(cart)} items")
```

Print-debugging is a first resort because it needs no tools, works everywhere, and answers the most common question directly: "what is actually in this variable right now?" Its downside is that you have to edit the code, rerun, and then remember to delete the prints. When that loop gets slow, step up to the real debugger.

## The real debugger: breakpoint() and pdb

Python ships with an interactive debugger called **pdb**. Since Python 3.7 you drop into it by writing `breakpoint()` on the line where you want to pause:

```python
def total_cost(cart, prices):
    total = 0
    for item in cart:
        breakpoint()                 # pause here, every loop
        total += get_price(item, prices)
    return total
```

When execution reaches `breakpoint()`, the program freezes and you get a `(Pdb)` prompt. Now you are *inside* the running program and can inspect anything. The essential commands are short:

- **`p expr`** — print the value of an expression, e.g. `p item` or `p prices`.
- **`l`** (list) — show the source lines around where you are paused, so you can see the context.
- **`n`** (next) — run the current line and stop at the next one in this same function.
- **`s`** (step) — like `n`, but if the current line calls a function, step *into* it.
- **`c`** (continue) — resume running until the next breakpoint or the end.
- **`q`** (quit) — stop the program and leave the debugger.

A live session on the buggy shop program might look like this. You add `breakpoint()` at the top of `get_price`, run, and:

```
(Pdb) p item
'milk'
(Pdb) p prices
{'apple': 30, 'bread': 25}
(Pdb) p item in prices
False
```

Three commands and the bug is obvious: `item` is `'milk'`, and it is not in `prices`. Use `n` to walk forward one line at a time watching values change, `s` when you want to follow a call into another function, and `c` to fly to the next pause. The debugger beats print-debugging when the bug is buried in a loop or deep in nested calls, because you inspect anything you like without editing and rerunning.

## A systematic method

Random poking wastes time. Good debuggers follow a loop, whether they realize it or not:

1. **Reproduce.** Make the bug happen on demand. A bug you cannot trigger reliably you cannot fix reliably. Find the exact input that breaks it.
2. **Isolate.** Narrow down *where* it happens. The traceback gives you the line; a `print` or a `breakpoint()` a few lines earlier tells you the last point where things were still correct.
3. **Hypothesize.** Form one specific guess: "I bet `item` is a string when the code expects it in the dict." A vague "something is wrong here" is not a hypothesis.
4. **Check.** Test that one guess with a `print` or a `p` in pdb. If you were right, fix it. If not, discard the guess and form a new one. Change one thing at a time so you always know what caused what.

A classic companion to this loop is **rubber-duck debugging**: explain your code, line by line, out loud to an inanimate object (traditionally a rubber duck). It sounds silly and it works, because the act of stating what each line *should* do forces you to notice the line that does not. If no duck is handy, a patient coworker or a written note works just as well.

## Looking ahead: from print to logging

Print-debugging is perfect for a quick "what is in here?" while you are actively working. But scattering `print` calls through code that ships to real users is a mess: the output is noisy, you cannot turn it off, and you always forget to remove some. For real programs, Python's `logging` module replaces `print` with something you can switch on and off, tag by severity, and route to a file. You will meet it in the software-engineering-practices lessons. For now, reach for `print` and `breakpoint()` while you learn; just know that a cleaner tool is waiting when your code grows up.

## Key takeaways

- Read tracebacks **bottom-up**: the last line is the error type and message; the lines above trace the call stack, with the immediate cause at the bottom.
- Learn the common exceptions on sight: `NameError` (unknown name), `TypeError` (wrong kind of value), `KeyError`/`IndexError` (missing dict key / list position), `AttributeError` (no such attribute, often on `None`), `ValueError` (right type, bad value).
- Print-debug sanely: print the value **and** its `type`, and use `!r` so strings show their quotes.
- Use `breakpoint()` to drop into pdb; the core commands are `p` (print), `l` (list), `n` (next), `s` (step in), `c` (continue), `q` (quit).
- Debug systematically: reproduce, isolate, hypothesize, check — one change at a time. Rubber-ducking surfaces bugs by making you explain the code aloud.
- Use `print` while learning, but reach for `logging` in real code you ship.

## Try it

Take the buggy shop program from this lesson and get to the bottom of it yourself. First run it and read the traceback out loud, naming the error type and the exact culprit line. Then add a `print` inside the loop that shows each `item` and its `type`, and confirm which value breaks the lookup. Finally, replace the print with `breakpoint()`, rerun, and at the `(Pdb)` prompt use `p item`, `p prices`, and `p item in prices` to prove the cause. As a last step, fix the bug two ways and decide which you prefer: skip missing items, or give them a default price.
