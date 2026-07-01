# 06 — Object-Oriented Basics

You have been using objects all along without naming them. A string knows how to strip itself, a list knows how to append to itself, a file object knows how to read itself. Each of those is an object: a bundle of data together with the actions that operate on that data. In this lesson you learn to build your own objects using classes. This unlocks a powerful way of organizing programs around the things they model.

## What is an object

An object combines two things: data (called attributes) and behavior (called methods, which are just functions that belong to the object). A string object holds its characters and offers methods like `strip` and `upper`. A list object holds its items and offers `append` and `sort`. When you write `text.upper()`, you are asking the string object `text` to run its `upper` method on its own data.

Building your own objects lets you model the things in your program directly. If your program is about bank accounts, you can have account objects. If it is about students, student objects. The data and the actions that belong together stay together.

## Defining a class

A class is the blueprint from which objects are made. You define one with the `class` keyword, and by convention class names use CapitalizedWords:

```python
class Dog:
    def bark(self):
        print("Woof!")
```

From the blueprint you create objects, called instances, by calling the class like a function:

```python
rex = Dog()
rex.bark()   # Woof!
```

`rex` is an instance of `Dog`. You can make as many as you like, and each is independent. Notice the `self` parameter in `bark`. Every method automatically receives the instance it was called on as its first argument, and by convention we name it `self`. You do not pass it yourself; Python fills it in when you write `rex.bark()`.

## Attributes and __init__

Objects become useful when they carry data. The special method `__init__` runs automatically each time you create an instance, and it is where you set up the object's attributes. You attach data to the instance by assigning to `self`:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        print(f"{self.name} says Woof!")

rex = Dog("Rex", 3)
fido = Dog("Fido", 5)

print(rex.name)    # Rex
print(fido.age)    # 5
rex.bark()         # Rex says Woof!
```

When you write `Dog("Rex", 3)`, Python creates a fresh object and calls `__init__` with `self` set to that new object and the other arguments filled in. Inside, `self.name = name` stores the name on the object. Each instance keeps its own attributes, which is why `rex` and `fido` have different names and ages.

Methods can use those attributes freely, as `bark` does with `self.name`. This is the heart of the idea: the data and the behavior live together in one object.

## Methods that do real work

Methods can read and change an object's data. Here is an account that tracks a balance and offers actions to change it:

```python
class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds.")
            return
        self.balance -= amount

    def summary(self):
        return f"{self.owner}: {self.balance}"

acc = Account("Ada", 100)
acc.deposit(50)
acc.withdraw(30)
print(acc.summary())   # Ada: 120
```

Everything about an account, its data and the rules for changing that data, is gathered in one place. The `withdraw` method even enforces a rule: you cannot withdraw more than you have. Bundling the rules with the data like this helps keep an object in a valid state.

## A friendlier printout

By default, printing an object shows something unhelpful like `<__main__.Account object at 0x...>`. You can define a `__str__` method to control what `print` shows:

```python
class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def __str__(self):
        return f"Account({self.owner}, {self.balance})"

acc = Account("Ada", 100)
print(acc)   # Account(Ada, 100)
```

Methods whose names begin and end with double underscores, like `__init__` and `__str__`, are special hooks that Python calls at particular moments. You do not call them directly; you define them and Python uses them.

## When objects help

Classes are not always the right tool, and reaching for them too early can overcomplicate simple code. A single function is often clearer than a class with one method. Objects earn their place when you have data and behavior that clearly belong together, when you need many independent copies of the same kind of thing, or when you want to enforce rules about how some data can change. Accounts, users, game characters, and documents are natural objects. A one-off calculation is not.

As a rule of thumb: if you find yourself passing the same cluster of related values into many functions, those values and functions probably want to become an object. Until then, plain functions and the data structures you already know will serve you well.

## Key takeaways

- An object bundles data (attributes) with behavior (methods); you have used objects like strings and lists all along.
- A class is a blueprint; calling it creates instances, each with its own attributes.
- `__init__` runs on creation and sets up attributes via `self`.
- Every method receives the instance as its first parameter, named `self` by convention.
- Methods can read and modify the object's data, and can enforce rules about valid states.
- Define `__str__` for a readable printout; double-underscore methods are hooks Python calls for you.

## Try it

Design a `Rectangle` class. Its `__init__` should take a width and a height and store them as attributes. Add an `area` method that returns width times height, and a `perimeter` method that returns twice the sum of the sides. Add a `__str__` method so printing a rectangle shows its dimensions nicely. Create two rectangles of different sizes, print each, and print their areas. Then reflect: would a plain function have been simpler here, or does bundling the data and calculations together feel cleaner?
