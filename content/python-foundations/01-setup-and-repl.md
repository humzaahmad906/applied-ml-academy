# 01 — Setup and the REPL

Welcome. This is the very first step of your journey into programming with Python, and we are going to take it slowly and carefully. By the end of this lesson you will have Python running on your own computer, you will have typed real code and watched it respond, and you will understand the two main ways people write Python programs. No prior experience is assumed. If you have never written a line of code before, you are exactly who this lesson is for.

## Why Python

Python is a programming language: a way of writing instructions that a computer can follow. People choose Python because it reads almost like plain English, it forgives small mistakes gracefully, and it powers an enormous amount of the modern world, from websites to scientific research to the machine learning systems this platform teaches. Learning Python well means everything else you learn later rests on solid ground.

## Installing Python

The goal here is to get a working copy of Python onto your machine. There are several ways to do this, and any of them is fine.

The simplest path for most beginners is to download the official installer. Visit the official Python website, choose the latest stable version for your operating system, and run the installer. On Windows, make sure to tick the box that says "Add Python to PATH" during installation. This lets your computer find Python from anywhere.

To confirm it worked, open a terminal (called Command Prompt or PowerShell on Windows, Terminal on macOS and Linux) and type:

```
python --version
```

You should see something like `Python 3.12.1`. The exact numbers do not matter much, as long as it starts with a 3. If you see a version number, Python is installed and ready.

## What is the REPL

The REPL is the friendliest place to start. The name stands for Read, Evaluate, Print, Loop. That describes exactly what it does: it reads a line you type, evaluates it (works out what it means), prints the result, and loops back to wait for your next line. It is a live conversation with Python.

Start it by typing `python` on its own in your terminal and pressing Enter:

```
python
```

You will see a prompt made of three greater-than signs:

```
>>>
```

That prompt is Python waiting for you. Try typing some arithmetic:

```python
>>> 2 + 2
4
>>> 10 * 5
50
>>> 100 / 4
25.0
```

Notice that Python answered each line immediately. This instant feedback is what makes the REPL so good for learning. You have a question, you type it, you get an answer. Try a few more:

```python
>>> "hello" + " " + "world"
'hello world'
>>> 7 > 3
True
```

Python happily glued two pieces of text together and told you that seven is indeed greater than three. To leave the REPL, type `exit()` and press Enter, or press Ctrl-D.

## Scripts: saving your work

The REPL is wonderful for experiments, but everything vanishes when you close it. When you want to keep a program and run it again later, you write a script: a file containing Python code that you save to disk.

Create a plain text file named `hello.py`. The `.py` ending tells the computer this is Python code. Put one line inside it:

```python
print("Hello from my first script!")
```

The `print` function displays text on the screen. In the REPL, results appear automatically, but in a script you must ask for output explicitly with `print`. Save the file, then in your terminal run:

```
python hello.py
```

You should see your message appear. You just wrote and ran your first program. You can run this same file as many times as you like, share it with others, and build it up over time.

## Notebooks: a third way

There is a third environment worth knowing about, especially for data and machine learning work: the notebook. A notebook mixes runnable code, its output, and written notes in one document. You write code in small blocks called cells and run them one at a time, seeing each result appear directly below the cell. It feels like a blend of the REPL and a script.

Notebooks are excellent for exploring data, plotting charts, and explaining your thinking as you go. Many later lessons on this platform use them. To try one, you can install the notebook tools with your terminal:

```
pip install notebook
```

Then launch it:

```
jupyter notebook
```

Your web browser will open with a workspace where you can create a new notebook and start typing code into cells.

## Which should I use

Think of it this way. Reach for the REPL when you want to test a quick idea or check how something behaves. Reach for a script when you have a finished, repeatable program you want to save and run again. Reach for a notebook when you are exploring data and want your code, results, and explanations living together. You will use all three over time, and switching between them will soon feel natural.

For now, the important thing is that Python runs on your machine and you have seen it respond to you. Everything else builds from here.

## Key takeaways

- Python is a beginner-friendly language and the foundation for the rest of this platform.
- Confirm your install by running `python --version` in a terminal.
- The REPL reads, evaluates, prints, and loops, giving instant feedback one line at a time.
- Scripts are saved `.py` files you run with `python filename.py`, and they keep your work.
- Notebooks combine code, output, and notes, and shine for data exploration.
- Use `print(...)` to show output from a script.

## Try it

Open the REPL and use it as a calculator: work out how many minutes are in a week by multiplying the numbers together. Then create a script named `about_me.py` that uses two `print` lines to display your name and your favorite number. Run the script from your terminal and confirm both lines appear. Finally, close the REPL and reopen it, and notice that your earlier calculations are gone, while your saved script still runs. That contrast is the whole point of this lesson.
