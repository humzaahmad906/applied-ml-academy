# 02 — Virtual Environments and Dependency Management

Every Python project depends on other packages, and different projects need different versions of them. One project wants an old version of a library; another needs the latest. If you install everything into a single shared Python, sooner or later two projects will demand incompatible versions and one of them breaks. Virtual environments solve this by giving each project its own isolated set of packages. This lesson explains why that isolation matters, then shows you the modern toolchain — **uv** — that makes creating environments and managing dependencies fast and reproducible.

## Why virtual environments exist

Your operating system ships with a Python, and other system tools may rely on it. If you `pip install` packages into that system Python, you are mixing your project's needs with the operating system's — a recipe for breakage. Worse, all your projects would share one pile of packages, so upgrading a library for one project silently changes it for every other.

The rule is simple: **one environment per project**. A virtual environment is just a private folder holding its own copy of Python and its own installed packages. Activate it, and `python` and `pip` point at that private world. Your projects stay independent, and you can delete an environment and rebuild it without touching anything else.

## uv: the modern default

For years the standard was `python -m venv` to create environments and `pip` to install into them, often with `pip-tools` bolted on for lockfiles. In 2026 the fast, unified default is **uv** — a single tool that replaces `pip`, `virtualenv`, and `pip-tools` at once, and runs dramatically faster.

Starting a new project takes one command:

```bash
uv init myproject
cd myproject
```

This creates a project with a `pyproject.toml` (the file you met in Lesson 01) already set up. To add a dependency, you do not edit the file by hand — you let uv do it:

```bash
uv add requests
uv add "pydantic>=2.0"
```

`uv add` records the dependency in your `pyproject.toml`, resolves compatible versions, installs them into the project's environment, and updates the lockfile — all in one step.

## The lockfile: your reproducibility contract

When uv resolves dependencies it writes a file called `uv.lock`. This lockfile records the exact version of every package in your dependency tree — not just the ones you asked for, but their dependencies too, pinned to precise versions with checksums. It is the difference between "install something compatible with pydantic 2" and "install exactly these versions that we know work together."

You **commit `uv.lock` to version control.** It is the contract that guarantees a teammate — or a server, or you in six months — rebuilds the identical environment. To materialize that environment from the lockfile:

```bash
uv sync
```

`uv sync` reads `uv.lock` and makes your environment match it exactly, installing or removing packages as needed. If you want to update the pinned versions later (say, to pick up security fixes), you run:

```bash
uv lock --upgrade
```

which re-resolves and rewrites the lockfile, after which `uv sync` applies it.

## Running code in the environment

You do not need to manually "activate" anything with uv. Prefix commands with `uv run` and they execute inside the project's environment:

```bash
uv run python main.py
uv run pytest
```

`uv run` ensures the environment is in sync first, so you never accidentally run against stale dependencies.

## Working with existing projects and plain pip

Not every project uses uv's full workflow. For an existing project, or when you just want an environment and pip-style installs, uv still helps:

```bash
uv venv
uv pip install -r requirements.txt
```

`uv venv` creates a virtual environment (like `python -m venv` but faster), and `uv pip install` is a drop-in replacement for `pip install`. This lets you adopt uv incrementally without restructuring anything.

It is worth knowing the baseline these tools build on. The standard-library approach is `python -m venv .venv` to create an environment, `source .venv/bin/activate` to activate it, and `pip install` to add packages — no lockfile unless you add one. **poetry** is another popular tool common in industry that also manages dependencies and lockfiles with its own workflow. uv covers the same ground as both, faster, which is why it is the recommended starting point today.

## Pinning your Python version

Projects should also pin which Python they run on, so everyone uses the same interpreter. A `.python-version` file at the project root does this — it simply contains a version string like `3.12`. uv can even install that Python for you:

```bash
uv python install 3.12
```

With a `.python-version` file present, uv will use that interpreter automatically when it creates the project's environment, so "which Python?" stops being a source of surprises.

## Key takeaways

- Never install project packages into your system Python; give each project its own isolated virtual environment.
- **uv** is the modern default: `uv init` to start, `uv add` to add dependencies, `uv sync` to build the environment, and `uv run` to execute code inside it.
- The `uv.lock` lockfile pins every package to an exact version — commit it, and everyone rebuilds an identical environment.
- `uv lock --upgrade` re-resolves versions when you want updates; `uv sync` then applies them.
- For existing projects, `uv venv` and `uv pip install` are fast drop-in replacements for `python -m venv` and `pip`.
- `python -m venv` + pip is the stdlib baseline and poetry is a common industry alternative; uv replaces both.
- A `.python-version` file plus `uv python install 3.12` pins the interpreter so everyone runs the same Python.

## Try it

Create a fresh project with `uv init`, then add two dependencies with `uv add` (try `requests` and `rich`). Open the generated `pyproject.toml` and confirm they appear under `dependencies`, then open `uv.lock` and notice how many packages it pins — far more than the two you named, because it locks their dependencies too. Write a tiny `main.py` that imports one of the libraries and prints something, and run it with `uv run python main.py`. Finally, add a `.python-version` file containing `3.12`, delete the `.venv` folder, run `uv sync`, and watch uv rebuild the exact same environment from the lockfile.
