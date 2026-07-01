# 05 — Files and Error Handling

Programs become far more useful once they can read information from files and save results back out. And the moment a program touches the outside world, things can go wrong: a file might be missing, a number might be malformed, a calculation might divide by zero. This lesson teaches you both halves of robust programming: working with files, and handling errors gracefully so a small problem does not crash everything.

## Reading a file

To read a file, you open it, read from it, and close it. Python gives you a clean way to do all three with the `with` statement, which closes the file for you automatically when the block ends:

```python
with open("notes.txt") as f:
    contents = f.read()
print(contents)
```

The `open` function returns a file object, and `read` pulls in the whole file as one string. The `with` block guarantees the file is closed even if something goes wrong inside it, which is why it is the recommended way to work with files.

Often you want the file line by line rather than as one blob. Looping over a file object gives you one line at a time:

```python
with open("notes.txt") as f:
    for line in f:
        print(line.strip())
```

The `strip` method removes the newline character at the end of each line, along with any surrounding spaces, so your output does not have blank gaps.

## Writing a file

To write, open the file in write mode by passing `"w"` as the second argument. Then use the file object's `write` method:

```python
with open("output.txt", "w") as f:
    f.write("First line\n")
    f.write("Second line\n")
```

Note two things. First, write mode replaces the file's contents entirely, so anything already there is erased. If you want to add to the end instead, use append mode with `"a"`. Second, `write` does not add line breaks for you, so include `\n` yourself where you want new lines.

Here is a small end-to-end example that reads a file of numbers, adds them up, and writes the total to a new file:

```python
total = 0
with open("numbers.txt") as f:
    for line in f:
        total += int(line.strip())

with open("total.txt", "w") as f:
    f.write(f"Total: {total}\n")
```

## When things go wrong

The example above assumes every line is a clean number and the file exists. In the real world, neither is guaranteed. When Python hits a problem it cannot handle, it raises an error (also called an exception) and, unless you intervene, the program stops and prints a message.

You have probably already seen a few. A missing file raises `FileNotFoundError`. Converting bad text to a number raises `ValueError`. Dividing by zero raises `ZeroDivisionError`. Looking up a missing dictionary key raises `KeyError`. Each error has a name that tells you what kind of thing went wrong.

## try and except

To handle an error rather than crash, wrap the risky code in a `try` block and describe what to do in an `except` block:

```python
try:
    number = int("not a number")
except ValueError:
    print("That was not a valid number.")
```

The `try` block runs normally. If an error of the named type occurs, Python jumps straight to the matching `except` block instead of crashing. If no error occurs, the `except` block is skipped entirely.

Always name the specific error you expect. Catching a bare exception with no type hides bugs you did not anticipate and makes problems hard to find. Be precise about what you are prepared to handle:

```python
try:
    with open("data.txt") as f:
        value = int(f.read())
except FileNotFoundError:
    print("The file is missing.")
except ValueError:
    print("The file did not contain a number.")
```

Here two different problems get two different, helpful responses. You can list as many `except` blocks as you need.

## else and finally

Two optional companions round out the pattern. An `else` block runs only if the `try` block succeeded with no error, which keeps your success path clearly separated from the risky part:

```python
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero.")
else:
    print(f"Result is {result}")   # Result is 5.0
```

A `finally` block runs no matter what, whether or not an error occurred. It is the place for cleanup that must always happen:

```python
try:
    risky_operation()
except ValueError:
    print("Handled the error.")
finally:
    print("This always runs.")
```

## Handling errors well

The goal is not to wrap every line in `try` and `except`. That clutters your code and hides real problems. Instead, handle the specific errors you can reasonably expect and recover from, such as a missing input file or malformed data, and let genuinely unexpected errors surface so you can see and fix them. A robust program is not one that never fails; it is one that fails clearly and recovers where it sensibly can.

Bringing the ideas together, here is a reader that skips bad lines gracefully instead of crashing on the first one:

```python
total = 0
try:
    with open("numbers.txt") as f:
        for line in f:
            try:
                total += int(line.strip())
            except ValueError:
                print(f"Skipping bad line: {line.strip()!r}")
except FileNotFoundError:
    print("No numbers file found; total stays 0.")

print(f"Total: {total}")
```

## Key takeaways

- Use `with open(...) as f:` to read or write files; it closes the file automatically.
- Read a whole file with `read`, or loop over the file object for one line at a time; `strip` trims newlines.
- Open with `"w"` to overwrite, `"a"` to append; add `\n` yourself for line breaks.
- Errors (exceptions) stop a program unless handled; each has a name like `ValueError` or `FileNotFoundError`.
- Wrap risky code in `try` and handle specific exceptions in `except`; avoid catching everything blindly.
- `else` runs on success and `finally` always runs, useful for cleanup.

## Try it

Create a text file with several lines, where most are numbers but a couple are words. Write a program that reads the file line by line, adds up the numbers, and skips the non-numbers without crashing, printing a note for each line it skips. Then write the final total to a second file. As a final check, run your program after renaming the input file so it is missing, and confirm it prints a friendly message instead of a crash.
