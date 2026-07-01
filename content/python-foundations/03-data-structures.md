# 03 — Data Structures

So far you have stored one value at a time in a variable. Real programs work with collections: a list of prices, a table of user records, a set of unique tags. Python gives you four built-in ways to hold groups of values, each suited to a different job. This lesson introduces all four and, just as importantly, teaches you how to choose between them.

## Lists

A list is an ordered collection that you can change. You write one with square brackets, separating items with commas:

```python
fruits = ["apple", "banana", "cherry"]
```

Because a list is ordered, each item has a position, called its index, starting from zero. You reach an item with square brackets:

```python
print(fruits[0])    # apple
print(fruits[2])    # cherry
```

Negative indexes count from the end, so `fruits[-1]` is the last item. Lists are changeable, which means you can add, replace, and remove items:

```python
fruits.append("date")     # add to the end
fruits[0] = "apricot"     # replace an item
fruits.remove("banana")   # remove by value
print(fruits)             # ['apricot', 'cherry', 'date']
print(len(fruits))        # 3
```

The `len` function gives the number of items. You can also take a slice, a sub-list, using a colon:

```python
numbers = [10, 20, 30, 40, 50]
print(numbers[1:4])   # [20, 30, 40]
```

Reach for a list whenever you have a sequence of items that might grow, shrink, or change, and where order matters.

## Dictionaries

A dictionary stores pairs: a key and the value it maps to. Think of a real dictionary, where you look up a word (the key) to find its definition (the value). You write one with curly braces and colons:

```python
person = {
    "name": "Ada",
    "age": 30,
    "city": "London",
}
```

You look up a value by its key, not by position:

```python
print(person["name"])   # Ada
```

Dictionaries are changeable too. You add or update a value by assigning to a key:

```python
person["age"] = 31          # update
person["email"] = "a@x.io"  # add a new pair
```

Looking up a key that does not exist raises an error, so a safe way to fetch is `get`, which returns a fallback instead:

```python
print(person.get("phone", "unknown"))   # unknown
```

You can loop over a dictionary's keys, values, or pairs:

```python
for key, value in person.items():
    print(key, "->", value)
```

Reach for a dictionary when you want to look things up by a meaningful label rather than a position: a username to a profile, a product code to a price.

## Sets

A set is an unordered collection with no duplicates. It is the right tool when you care only about membership: is this thing present or not, and what are the unique items? You write one with curly braces, but with single values rather than pairs:

```python
tags = {"python", "data", "python", "ml"}
print(tags)   # {'python', 'data', 'ml'}  (duplicate removed)
```

Sets make membership tests fast and duplicate removal effortless:

```python
print("data" in tags)     # True
numbers = [1, 2, 2, 3, 3, 3]
unique = set(numbers)
print(unique)             # {1, 2, 3}
```

Sets also support real set operations like union and intersection:

```python
a = {1, 2, 3}
b = {2, 3, 4}
print(a & b)   # {2, 3}  (in both)
print(a | b)   # {1, 2, 3, 4}  (in either)
```

Reach for a set when order does not matter and you want either uniqueness or fast "is it in here?" checks.

## Tuples

A tuple looks like a list but is written with parentheses and cannot be changed once created:

```python
point = (3, 4)
print(point[0])   # 3
```

Because a tuple is fixed, it signals "these values belong together and will not change," like the x and y of a coordinate or the day, month, and year of a date. Tuples are also how Python bundles multiple return values, which you will see in a later lesson. A neat trick is unpacking, assigning each part to its own name in one line:

```python
x, y = point
print(x, y)   # 3 4
```

Reach for a tuple when you have a small, fixed group of related values that should not be modified.

## Choosing the right one

A short guide to keep in mind:

- Use a **list** for an ordered sequence you expect to change.
- Use a **dictionary** to look values up by a meaningful key.
- Use a **set** for uniqueness or fast membership checks, when order is irrelevant.
- Use a **tuple** for a small, fixed group of values that belong together.

These four cover the vast majority of everyday needs, and they nest freely: a list of dictionaries is a common way to hold a table of records, and you will meet it constantly in data work.

```python
people = [
    {"name": "Ada", "age": 30},
    {"name": "Alan", "age": 41},
]
print(people[1]["name"])   # Alan
```

## Key takeaways

- Lists are ordered and changeable; access items by index, slice with a colon, and grow with `append`.
- Dictionaries map keys to values; look up by key, and use `get` to avoid errors on missing keys.
- Sets are unordered with no duplicates; ideal for uniqueness and fast membership tests.
- Tuples are fixed groups of related values and support convenient unpacking.
- `len(...)` counts items in any of these collections.
- These structures nest, and a list of dictionaries is a standard way to hold tabular data.

## Try it

Build a small phone book. Start with a dictionary mapping three names to phone numbers. Add a fourth entry, update one existing number, and print each name-and-number pair using a loop over `.items()`. Then make a list of the fruits people mentioned, deliberately including some duplicates, and use a set to print only the unique ones along with how many unique fruits there were. Predict each output before you run it, then check.
