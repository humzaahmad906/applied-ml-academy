# 04 — Functions and Modules

As programs grow, you find yourself writing the same steps again and again. Functions let you name a piece of work once and reuse it everywhere. Modules let you organize functions into files and borrow code that others have already written. Together they are how programs stay manageable as they scale. This lesson shows you how to define your own functions and how to bring in code from elsewhere.

## Defining a function

A function is a named block of code that you can run whenever you like. You define one with the `def` keyword, a name, parentheses, and a colon, then indent the body:

```python
def greet():
    print("Hello there!")
```

Defining a function does not run it. To run, or "call," the function, write its name followed by parentheses:

```python
greet()   # Hello there!
greet()   # Hello there!
```

The value of a function is that you write the logic once and reuse it as many times as you need.

## Arguments

Most functions need information to do their job. You pass that information through arguments, which the function receives as parameters, the names in the parentheses:

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Ada")    # Hello, Ada!
greet("Alan")   # Hello, Alan!
```

You can accept several arguments, separated by commas:

```python
def describe(name, age):
    print(f"{name} is {age} years old.")

describe("Ada", 30)
```

You can also give a parameter a default value, used when the caller does not supply one:

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Ada")              # Hello, Ada!
greet("Ada", "Welcome")   # Welcome, Ada!
```

Passing arguments by name makes calls clearer and lets you skip over defaults:

```python
greet("Ada", greeting="Hi")
```

## Returning values

Printing shows something on screen, but often you want a function to hand a value back so the rest of your program can use it. That is what `return` does:

```python
def add(a, b):
    return a + b

total = add(3, 4)
print(total)          # 7
print(add(10, 20))    # 30
```

The difference between printing and returning matters. A function that prints shows you something; a function that returns gives you a value you can store, combine, and pass along. Most useful functions return.

A function can return more than one value by separating them with commas, which Python bundles into a tuple. This pairs neatly with the unpacking you saw earlier:

```python
def min_and_max(numbers):
    return min(numbers), max(numbers)

low, high = min_and_max([4, 1, 9, 2])
print(low, high)   # 1 9
```

Once a `return` runs, the function stops immediately. This is handy for guard clauses that exit early:

```python
def safe_divide(a, b):
    if b == 0:
        return None
    return a / b
```

## Why functions help

Functions do more than save typing. They give a name to an idea, so `safe_divide(x, y)` reads more clearly than the raw arithmetic and its guard. They let you fix a bug in one place instead of many. And they let you test a piece of logic on its own. A good habit is to make each function do one clear thing, and to name it after that thing.

## Modules and imports

A module is simply a Python file full of functions and values. Python ships with a large standard library of modules covering math, dates, randomness, file paths, and much more. You bring a module into your program with `import`:

```python
import math

print(math.sqrt(16))    # 4.0
print(math.pi)          # 3.141592653589793
```

After importing, you reach the module's contents with a dot: `math.sqrt`. If you only need one or two things, you can import them by name:

```python
from math import sqrt, pi

print(sqrt(25))   # 5.0
```

You can also give an import a shorter nickname, which is common with larger libraries you will meet later:

```python
import statistics as stats

print(stats.mean([2, 4, 6]))   # 4
```

Here are a couple more useful modules from the standard library:

```python
import random
print(random.randint(1, 6))    # a random dice roll

import datetime
print(datetime.date.today())   # today's date
```

## Your own modules

Any script you write is itself a module. If you save some functions in a file named `tools.py`, another script in the same folder can use them:

```python
# in tools.py
def double(x):
    return x * 2
```

```python
# in another file
import tools
print(tools.double(21))   # 42
```

This is how programs grow beyond a single file: you split related functions into modules, then import what you need. It keeps each file focused and your code easy to find.

## Key takeaways

- Define functions with `def name(parameters):` and run them by calling `name(arguments)`.
- Parameters can have default values, and arguments can be passed by name for clarity.
- `return` hands a value back to the caller; a function that only prints gives nothing back.
- Returning several comma-separated values produces a tuple you can unpack.
- `import` brings in modules; reach their contents with a dot, or import specific names directly.
- Any `.py` file is a module, so you can split your own code across files and import between them.

## Try it

Write a function `celsius_to_fahrenheit(temp)` that converts a temperature and returns the result (the formula is `temp * 9 / 5 + 32`). Call it with a few values and print each. Then write a second function that takes a list of Celsius temperatures and returns a new list of the converted values. Finally, import the `random` module and use it to generate five random temperatures between -10 and 40, convert them with your functions, and print both versions side by side.
