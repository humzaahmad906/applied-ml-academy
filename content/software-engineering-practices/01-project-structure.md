# 01 — Project Structure and Python Packaging

When you first learn Python, a project is often just a folder full of `.py` files that you run one at a time. That works for a while, but the moment you want to install your code, share it, import it cleanly across files, or test it properly, you need real structure. This lesson shows you how modern Python projects are laid out, how the single configuration file `pyproject.toml` describes your project, and why a small amount of structure now saves you a great deal of confusion later.

## The one file that describes your project: pyproject.toml

Modern Python projects are configured through a single file called `pyproject.toml`. It lives at the root of your project and holds everything build tools and packaging tools need to know. TOML is a simple, readable configuration format — sections in square brackets, `key = value` pairs underneath.

Here is a minimal but complete `pyproject.toml`:

```toml
[project]
name = "mypackage"
version = "0.1.0"
description = "A small example package"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.32",
    "pydantic>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

The `[project]` table is your project's identity. `name` is how the package is installed and imported, `version` follows semantic versioning, `requires-python` states the minimum Python you support, and `dependencies` lists the other packages you need with version constraints.

The `[build-system]` table tells tools how to turn your source into an installable package. Here we use **hatchling**, a modern, low-config build backend. You will also see **setuptools** (the long-standing default) and **flit** (minimal, for pure-Python packages) in the wild — any of them works, and hatchling is a fine choice for new projects.

## src layout vs flat layout

There are two common ways to arrange your code. In the **flat layout**, your package folder sits directly at the project root:

```
myproject/
    mypackage/
        __init__.py
        core.py
    tests/
    pyproject.toml
```

In the **src layout**, your package lives inside a `src/` directory:

```
myproject/
    src/
        mypackage/
            __init__.py
            core.py
    tests/
    pyproject.toml
    README.md
```

Both work, but the src layout is recommended for new projects, and the reason is subtle but important. When your package sits at the root, Python can import it just because you happen to be standing in that directory — even if you never installed it. That means your tests might pass against the loose files on disk rather than against the package as it would actually be installed. With the src layout, the only way your tests can find `mypackage` is if it is properly installed, so you test the real thing. This catches "works on my machine" packaging bugs before your users hit them.

## Command-line entry points

If your package provides a command someone should be able to run in their terminal, you declare it under `[project.scripts]`:

```toml
[project.scripts]
mytool = "mypackage.cli:main"
```

This says: create a command named `mytool` that calls the `main` function in `mypackage/cli.py`. After installing the package, the user can type `mytool` at the shell and it just works — no `python -m` needed.

## Editable installs

While developing, you do not want to reinstall your package every time you change a line. An **editable install** links the installed package back to your source files, so edits take effect immediately:

```bash
pip install -e .
```

Or, with the faster tooling you will meet in Lesson 02:

```bash
uv pip install -e .
```

The `-e` means "editable" and the `.` means "the project in the current directory." This is the standard way to work on a package locally: install once in editable mode, then just keep editing.

## What not to use for new projects

You will still find old projects with `setup.py` and `setup.cfg` files. These predate `pyproject.toml` and carried the same information in older formats. For new work, you do not need them — `pyproject.toml` handles everything. Reach for `setup.py` only in the rare case where you need to run custom Python at build time (compiling C extensions, for example), and even then, prefer keeping metadata in `pyproject.toml`.

## Putting it together: a complete skeleton

Here is a clean starting point for a real project:

```
myproject/
    src/
        mypackage/
            __init__.py
            core.py
            cli.py
    tests/
        test_core.py
    pyproject.toml
    README.md
```

The `src/mypackage/__init__.py` file marks the folder as a package (it can be empty, or expose your public functions). The `tests/` folder mirrors your source — you will see why in Lesson 03. The `README.md` explains what the project is and how to use it, and packaging tools can display it on the Python Package Index. With this layout and a `pyproject.toml` like the one above, you have a project that installs cleanly, tests honestly, and is ready to share.

## Key takeaways

- `pyproject.toml` is the single configuration file for a modern Python project — `[project]` for identity and dependencies, `[build-system]` for how to build it.
- The src layout (`src/mypackage/`) is recommended because it forces your tests to run against the installed package, catching packaging bugs early.
- `[project.scripts]` turns a function into a command users can run from their terminal.
- An editable install (`pip install -e .` or `uv pip install -e .`) links the installed package to your source so edits take effect immediately.
- New projects do not need `setup.py` or `setup.cfg`; `pyproject.toml` replaces them.
- hatchling is a good modern build backend; setuptools and flit are common alternatives.

## Try it

Create a new folder for a small project and give it the src-layout skeleton shown above: a `src/` directory containing a package with `__init__.py` and a `core.py` holding one simple function (say, `def add(a, b): return a + b`), plus a `tests/` folder and a `README.md`. Write a `pyproject.toml` with a `[project]` table (pick a name, version, and `requires-python`) and a `[build-system]` table using hatchling. Run `pip install -e .` from the project root, then open a Python shell and confirm you can `from mypackage.core import add` and call it. Change the function, and confirm the change is visible without reinstalling.
