# 04b — Flexible Functions and Dataclasses

The last lesson gave you functions with a fixed set of parameters. Real programs are messier: sometimes you don't know in advance how many values a caller will hand you, sometimes you want to bundle related data into a tidy object, and sometimes you need to read a file without wrestling with clumsy path strings. This lesson covers the tools that handle all three. Along the way you will meet one of the most famous beginner traps in Python and learn the one-line habit that avoids it.

## Accepting any number of arguments with `*args`

Suppose you want a function that adds up however many numbers it is given: two, five, or twenty. You cannot write a parameter for each. Instead you put a `*` before a parameter name, and Python collects all the extra positional arguments into a tuple:

```python
def add_all(*numbers):
    total = 0
    for n in numbers:
        total += n
    return total

print(add_all(1, 2, 3))        # output: 6
print(add_all(10, 20, 30, 40)) # output: 100
print(add_all())               # output: 0
```

The name `numbers` is ordinary; the `*` is what does the work. By convention people write `*args` ("arguments") when the name does not matter, but a descriptive name like `*numbers` reads better.

## Accepting named extras with `**kwargs`

The sibling of `*args` uses two stars and collects extra *keyword* arguments into a dictionary:

```python
def make_profile(**details):
    for key, value in details.items():
        print(f"{key}: {value}")

make_profile(name="Ada", role="engineer", city="London")
# output: name: Ada
# output: role: engineer
# output: city: London
```

Here `details` is a normal dict: `{"name": "Ada", "role": "engineer", "city": "London"}`. The conventional name is `**kwargs` ("keyword arguments"). You can combine ordinary parameters, `*args`, and `**kwargs` in one signature, and they must appear in that order:

```python
def report(title, *values, **options):
    print(title)
    print("values:", values)
    print("options:", options)

report("Sales", 100, 200, currency="USD", year=2026)
# output: Sales
# output: values: (100, 200)
# output: options: {'currency': 'USD', 'year': 2026}
```

## Passing arguments through with `*` and `**`

The same two stars work in reverse. When placed in front of a list or dict *at the call site*, they unpack it into separate arguments. This is how you forward arguments from one function to another:

```python
def greet(greeting, name):
    print(f"{greeting}, {name}!")

args = ["Hello", "Ada"]
greet(*args)                    # output: Hello, Ada!

kwargs = {"greeting": "Hi", "name": "Alan"}
greet(**kwargs)                 # output: Hi, Alan!
```

You will see this constantly in libraries: a wrapper function accepts `*args, **kwargs` and passes them straight to the function it wraps, without needing to know their exact shape. Reach for `*args`/`**kwargs` when the number of inputs genuinely varies. If a function always takes three things, name those three things.

## The mutable default argument trap

Here is a bug that catches almost every Python beginner. You want a function that appends an item to a list, creating a fresh list if none is given:

```python
def add_item(item, basket=[]):   # <-- looks fine, is broken
    basket.append(item)
    return basket

print(add_item("apple"))    # output: ['apple']
print(add_item("bread"))    # output: ['apple', 'bread']   <-- surprise!
```

The second call was supposed to start empty, but "apple" is still there. The reason: **default values are created once, when the function is defined, not each time it is called.** That single list is shared by every call that relies on the default. Every append piles onto the same object.

The fix is a fixed habit. Use `None` as the default, a sentinel meaning "nothing was passed," and build the real value inside the function:

```python
def add_item(item, basket=None):
    if basket is None:
        basket = []
    basket.append(item)
    return basket

print(add_item("apple"))    # output: ['apple']
print(add_item("bread"))    # output: ['bread']   <-- fresh each time
```

The rule is simple: **never use a mutable object (list, dict, set) as a default value.** Use `None` and create the object inside the body. Immutable defaults like numbers, strings, and `True`/`False` are perfectly safe because they cannot be changed in place.

## Keyword-only arguments

Long argument lists get hard to read at the call site. What does `create_user("Ada", True, False)` mean? You can force certain arguments to be passed by name by putting a bare `*` in the signature. Everything after the `*` must be given as a keyword:

```python
def create_user(name, *, is_admin=False, verified=False):
    print(f"{name}: admin={is_admin}, verified={verified}")

create_user("Ada", is_admin=True)          # output: Ada: admin=True, verified=False
# create_user("Ada", True)                 # TypeError: too many positional arguments
```

Now the call *must* spell out `is_admin=True`, so the meaning is obvious and you cannot accidentally swap two booleans. Keyword-only arguments are the recommended way to expose optional flags and settings.

## Dataclasses: clean containers for data

Often you just want an object that holds a few related fields: a point, a config, a record. Written by hand, that is a lot of boilerplate:

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
```

The `@dataclass` decorator writes all of that for you. You declare the fields with type hints and Python generates `__init__`, `__repr__`, and equality checks automatically:

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(3, 4)
print(p)              # output: Point(x=3, y=4)
print(p.x)            # output: 3
print(p == Point(3, 4))  # output: True
```

That last line is a real convenience: two hand-written objects would compare as unequal by default, but dataclasses compare field by field.

Fields can have defaults, just like function parameters, and the mutable-default rule applies here too. For lists or dicts you must use `field(default_factory=...)`, which calls the factory fresh for each instance:

```python
from dataclasses import dataclass, field

@dataclass
class Config:
    name: str
    epochs: int = 10
    layers: list = field(default_factory=list)

c = Config("baseline")
print(c)   # output: Config(name='baseline', epochs=10, layers=[])
```

If you had written `layers: list = []` instead, every `Config` would share one list, the same trap as before, and modern Python actually raises an error to stop you.

You can also freeze a dataclass so its fields cannot be changed after creation. This is useful for values that should never mutate, like settings:

```python
@dataclass(frozen=True)
class Settings:
    learning_rate: float = 0.001
    batch_size: int = 32

s = Settings()
# s.batch_size = 64   # FrozenInstanceError: cannot assign to field
```

Dataclasses matter beyond convenience. As your projects grow you will meet config objects and validation libraries like Pydantic that build directly on this idea of "a class that describes structured data." Getting comfortable with dataclasses now is a direct stepping stone.

## `pathlib`: the modern way to handle files

Older code builds file paths by gluing strings together or calling functions from `os.path`. The `pathlib` module is cleaner: it gives you a `Path` object with methods for everything you need. The star feature is the `/` operator, which joins path pieces correctly on any operating system:

```python
from pathlib import Path

data_dir = Path("data")
file_path = data_dir / "raw" / "sample.txt"
print(file_path)          # output: data/raw/sample.txt
print(file_path.name)     # output: sample.txt
print(file_path.suffix)   # output: .txt
```

Reading and writing text is a single call, no need to open and close a file by hand:

```python
notes = Path("notes.txt")
notes.write_text("hello\nworld\n")
content = notes.read_text()
print(content)            # output: hello
                          # output: world
print(notes.exists())     # output: True
```

To find files matching a pattern, use `glob`. The `*` means "anything," so `*.csv` matches every CSV file in a folder:

```python
for csv_file in Path("data").glob("*.csv"):
    print(csv_file)
```

Prefer `Path` over string paths and over `os.path` in new code. It is shorter, harder to get wrong across operating systems, and reads like the operations you actually mean.

## Key takeaways

- `*args` gathers extra positional arguments into a tuple; `**kwargs` gathers extra keyword arguments into a dict.
- At a call site, `*list` and `**dict` unpack values back into separate arguments, the standard way to pass arguments through.
- Never use a mutable object (`[]`, `{}`) as a default argument; default to `None` and build the object inside the function.
- A bare `*` in a signature makes the following parameters keyword-only, which keeps calls readable.
- `@dataclass` generates `__init__`, `__repr__`, and equality from typed fields; use `field(default_factory=...)` for mutable defaults and `frozen=True` for read-only objects.
- `pathlib.Path` joins paths with `/`, reads and writes text in one call, and finds files with `glob`, preferred over raw strings and `os.path`.

## Try it

Write a function `average(*numbers)` that returns the mean of however many numbers it is given, returning `0` when called with none. Next, write a function `log_event(message, tags=None)` that appends `message` to the `tags` list (defaulting to a fresh list) and returns it, then call it several times to confirm each call starts empty. Then define a `@dataclass` called `Experiment` with fields `name: str`, `seed: int = 0`, and `metrics: list` using `default_factory`, and print an instance. Finally, use `pathlib` to write three numbers to a file called `scores.txt`, read them back, and print the total.
