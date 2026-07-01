# 07 — Comprehensions and Iterators

You already know how to loop over a collection and build up a result. Python offers a more compact and expressive way to do the most common version of that: comprehensions. It also has a deeper idea running underneath all looping, called iteration, along with generators that produce values lazily, one at a time. These tools make your code shorter, clearer, and often more efficient. This lesson brings them together.

## The pattern comprehensions replace

A very common task is to take one list and build a new list from it. With an ordinary loop it looks like this:

```python
numbers = [1, 2, 3, 4, 5]
squares = []
for n in numbers:
    squares.append(n * n)
print(squares)   # [1, 4, 9, 16, 25]
```

That is three lines of bookkeeping around one idea: square each number. A list comprehension expresses the same idea in a single readable line.

## List comprehensions

A list comprehension puts the loop inside the square brackets, with the expression you want first:

```python
numbers = [1, 2, 3, 4, 5]
squares = [n * n for n in numbers]
print(squares)   # [1, 4, 9, 16, 25]
```

Read it left to right as "n times n, for each n in numbers." The result is a brand-new list. You can transform items any way you like:

```python
words = ["hello", "world"]
shouted = [w.upper() for w in words]
print(shouted)   # ['HELLO', 'WORLD']
```

You can also filter, keeping only some items, by adding an `if` at the end:

```python
numbers = range(10)
evens = [n for n in numbers if n % 2 == 0]
print(evens)   # [0, 2, 4, 6, 8]
```

And you can combine transform and filter in one expression:

```python
squares_of_evens = [n * n for n in range(10) if n % 2 == 0]
print(squares_of_evens)   # [0, 4, 16, 36, 64]
```

A word of judgment: comprehensions are wonderful when they stay simple and readable. If one grows long, with several conditions or nested loops, a plain `for` loop is often clearer. Favor readability over cramming everything into one line.

## Dictionary and set comprehensions

The same syntax works for dictionaries and sets. A dictionary comprehension uses curly braces with a key-value pair:

```python
numbers = [1, 2, 3, 4]
squares_map = {n: n * n for n in numbers}
print(squares_map)   # {1: 1, 2: 4, 3: 9, 4: 16}
```

A set comprehension uses curly braces with single values, and gives you uniqueness for free:

```python
words = ["apple", "banana", "cherry"]
lengths = {len(w) for w in words}
print(lengths)   # {5, 6}  (apple and cherry both 6)
```

## Iteration, the idea underneath

Every time you write `for item in something`, Python is asking that something for its items one at a time. Anything that can be looped over this way is called iterable. Lists, dictionaries, sets, strings, files, and ranges are all iterable. This shared behavior is why the same `for` loop works on all of them.

You do not usually need to think about the machinery, but knowing it exists explains something important: not every iterable holds all its items in memory at once. A `range(1000000)` does not build a million-item list; it produces numbers on demand as you loop. This is the door to generators.

## Generators

A generator produces values one at a time, only as they are needed, rather than building a whole collection up front. The simplest way to make one is a generator expression, which looks exactly like a list comprehension but with parentheses instead of square brackets:

```python
squares = (n * n for n in range(1000000))
```

This line runs instantly and uses almost no memory, because no squares have been computed yet. They are produced one at a time as you loop:

```python
for s in squares:
    if s > 50:
        break
    print(s)   # 0 1 4 9 16 25 36 49
```

The payoff is efficiency. If you only need to scan through values once, or you are working with a huge or even endless sequence, a generator avoids building a giant list you would immediately throw away.

You can also write a generator as a function using the `yield` keyword. Each `yield` hands back one value and pauses the function until the next value is requested:

```python
def count_up_to(limit):
    n = 1
    while n <= limit:
        yield n
        n += 1

for number in count_up_to(4):
    print(number)   # 1 2 3 4
```

Unlike `return`, which ends a function, `yield` suspends it and remembers where it left off, resuming from that spot when the next value is asked for. This lets a generator produce a long or unbounded stream of values without ever holding them all at once.

## Choosing between them

Use a list comprehension when you want all the results at once and will use them more than once, or need to index into them. Use a generator when you will pass through the values a single time, especially if the sequence is large, because it saves memory. For everyday small collections, a list comprehension is usually the natural choice; reach for generators when scale or streaming makes them worthwhile.

## Key takeaways

- A list comprehension builds a new list in one line: `[expr for item in iterable]`.
- Add `if condition` to filter, and combine transforming and filtering in the same expression.
- Dictionary and set comprehensions use the same idea with curly braces.
- Anything you can loop over is iterable; the same `for` loop works across lists, strings, files, and more.
- A generator produces values lazily, one at a time, saving memory for large or single-pass sequences.
- Generator expressions use parentheses; generator functions use `yield`, which pauses and resumes.

## Try it

Given a list of words, use a list comprehension to build a new list containing only the words longer than four letters, each converted to uppercase. Next, build a dictionary comprehension mapping each word to its length. Then write a generator function that yields the squares of numbers from 1 up to a given limit, and use it in a loop to print the first few squares. Finally, compare the memory idea for yourself: create a list comprehension and a generator expression over `range(1000000)` and notice how one takes a moment while the other returns instantly.
