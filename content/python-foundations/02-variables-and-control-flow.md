# 02 — Variables, Types, and Control Flow

Now that Python runs on your machine, we can start giving it real work to do. This lesson covers the building blocks every program is made of: how to store information in variables, the basic kinds of information Python understands, and how to make your program decide and repeat. These ideas are small on their own but combine into everything.

## Variables

A variable is a name that points at a value. You create one with a single equals sign, and from then on the name stands for that value.

```python
age = 30
name = "Ada"
price = 19.99
```

The name goes on the left, the value on the right. You can use the name anywhere you would use the value, and you can change what it points to whenever you like:

```python
score = 10
score = score + 5
print(score)   # 15
```

Choose names that describe what they hold. `total_price` tells a reader far more than `t` or `x`. Good names make code read like a description of what it does.

## Types

Every value in Python has a type, which describes what kind of thing it is and what you can do with it. The three types you meet first are numbers, text, and booleans.

Numbers come in two flavors. Integers are whole numbers like `42` or `-7`. Floats have a decimal point, like `3.14` or `19.99`. Python handles both naturally:

```python
whole = 42          # an integer
fraction = 3.5      # a float
print(whole + fraction)   # 45.5
```

Text is called a string, because it is a string of characters. You write strings inside quotes, single or double, as long as they match:

```python
greeting = "Hello"
place = 'World'
message = greeting + ", " + place + "!"
print(message)   # Hello, World!
```

A handy trick is the f-string, which lets you drop variables straight into text by putting an `f` before the quote and wrapping names in curly braces:

```python
name = "Ada"
age = 30
print(f"{name} is {age} years old.")   # Ada is 30 years old.
```

Booleans are the simplest type of all: they are either `True` or `False`. They appear whenever you ask a yes-or-no question:

```python
is_adult = age >= 18
print(is_adult)   # True
```

If you ever want to know the type of a value, ask Python directly:

```python
print(type(42))       # <class 'int'>
print(type("hi"))     # <class 'str'>
print(type(True))     # <class 'bool'>
```

## Comparisons and logic

Comparisons produce booleans. The common ones are `==` for equal, `!=` for not equal, and `<`, `>`, `<=`, `>=` for ordering. Note the double equals for comparison, which is different from the single equals used to assign a variable.

```python
print(5 == 5)    # True
print(5 != 3)    # True
print(2 < 1)     # False
```

You can combine conditions with `and`, `or`, and `not`:

```python
temperature = 22
print(temperature > 15 and temperature < 25)   # True
```

## Making decisions with if

An `if` statement runs a block of code only when a condition is true. Python knows which lines belong to the block by their indentation, the spaces at the start of the line. This is not decoration; it is how Python reads structure.

```python
temperature = 30

if temperature > 25:
    print("It is warm.")
elif temperature > 10:
    print("It is mild.")
else:
    print("It is cold.")
```

Python checks each condition top to bottom. The first one that is true runs, and the rest are skipped. `elif` (short for "else if") lets you test more conditions, and `else` catches everything that was left over.

## Repeating with loops

Often you want to do something many times. A `for` loop repeats once for each item in a collection. The `range` function is a convenient way to loop a fixed number of times:

```python
for i in range(3):
    print("Hello number", i)
# Hello number 0
# Hello number 1
# Hello number 2
```

Notice that counting starts at zero, which is standard in Python. You can also loop directly over items:

```python
for fruit in ["apple", "banana", "cherry"]:
    print(fruit)
```

A `while` loop repeats as long as a condition stays true. Use it when you do not know in advance how many times you will loop:

```python
count = 3
while count > 0:
    print(count)
    count = count - 1
print("Lift off!")
```

Be careful with `while`: something inside the loop must eventually make the condition false, or the loop runs forever. Here, `count` shrinks each pass until it reaches zero.

## Putting it together

These pieces already let you write useful programs. Here is a small one that classifies a list of scores:

```python
scores = [95, 42, 78, 88, 30]

for score in scores:
    if score >= 50:
        print(f"{score}: pass")
    else:
        print(f"{score}: retake")
```

The loop visits each score, the `if` decides its fate, and the f-string reports the result. Read it slowly and you will see the whole lesson in five lines.

## Key takeaways

- A variable is a name pointing at a value, assigned with a single `=`.
- Core types are integers, floats, strings, and booleans; `type(x)` reveals any value's type.
- f-strings let you embed variables in text with `f"{name}"`.
- Comparisons like `==` and `>` produce booleans; combine them with `and`, `or`, `not`.
- `if` / `elif` / `else` choose which block runs, guided by indentation.
- `for` loops repeat over a collection or a `range`; `while` loops repeat while a condition holds.

## Try it

Write a script that stores a person's age in a variable and prints whether they are a child (under 13), a teenager (13 to 19), or an adult (20 and over), using `if` / `elif` / `else`. Then add a `for` loop that counts from 1 to 10 and prints "fizz" for every number divisible by 3 (test this with `number % 3 == 0`) and the number itself otherwise. Run it and check the output matches what you expect by hand.
