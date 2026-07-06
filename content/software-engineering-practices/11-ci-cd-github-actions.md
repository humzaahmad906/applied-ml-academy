# 11 — CI/CD with GitHub Actions

You have learned to write tests, format code, and check types. The problem is that a human has to *remember* to run all of those before merging — and humans forget, especially under deadline. Continuous Integration solves this by running your checks automatically, on every change, on a neutral machine, so a broken change is caught before it ever reaches your main branch. This lesson shows you how to set up CI for a Python project using GitHub Actions, the built-in automation system for repositories hosted on GitHub.

## What CI and CD mean

**CI — Continuous Integration** — means every change is automatically built and tested the moment it is proposed. When you open a pull request, a server checks out your code, installs dependencies, and runs your tests and quality checks. If anything fails, the PR is flagged and cannot merge until it is fixed. The value is that "it works on my machine" stops being a valid excuse: the checks run on a clean, standard environment every time.

**CD — Continuous Delivery/Deployment** — is the next step: once the checks pass, the change is automatically shipped to staging or production. This lesson covers CI, which is the foundation; you cannot safely automate deployment until you trust that your checks are thorough and always run.

## A minimal workflow file

GitHub Actions reads instructions from YAML files in a special directory: `.github/workflows/`. Each file describes a *workflow* — a set of jobs that run when some event occurs. Here is a complete, working CI workflow for a `uv`-managed Python project:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync

      - name: Lint
        run: uv run ruff check

      - name: Check formatting
        run: uv run ruff format --check

      - name: Run tests
        run: uv run pytest
```

## Reading the YAML

Let us walk through the structure, because these few keys appear in every workflow you will ever write.

- **`on`** declares *what triggers the workflow*. Here it runs on every `push` to `main` and on every `pull_request`. This is the heart of CI: the PR trigger means your checks run automatically on every proposed change.
- **`jobs`** is a collection of independent units of work. We have one job called `test`.
- **`runs-on: ubuntu-latest`** picks the machine. GitHub gives you a fresh Linux virtual machine for each run — a clean, standard environment, which is exactly what makes CI trustworthy.
- **`steps`** is an ordered list of things to do. Each step either *uses* a pre-built action or *runs* a shell command.

The `uses` steps pull in reusable actions maintained by others: `actions/checkout@v4` fetches your repository's code onto the runner, `astral-sh/setup-uv@v5` installs the `uv` tool, and `actions/setup-python@v5` installs the Python interpreter. The `@v4`/`@v5` pins each action to a major version so a future release cannot silently break your build.

The `run` steps then execute the same commands you run locally: `uv sync` installs your dependencies from the lockfile, `ruff check` lints, `ruff format --check` verifies formatting without changing files (it fails if anything is unformatted), and `pytest` runs your tests. If any command exits with an error, the job fails and GitHub marks the change red.

## Testing across Python versions with a matrix

Libraries often need to work on several Python versions. Rather than copy the job three times, a **matrix** runs it once per version automatically:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync
      - run: uv run pytest
```

The `strategy.matrix` block lists the versions, and `${{ matrix.python-version }}` substitutes each one in turn. GitHub runs the whole job twice, in parallel, once per Python version — so you learn immediately if your code works on 3.11 but breaks on 3.12.

## Badges and protecting main

Once CI is running, two touches make it visible and enforced. A **status badge** in your README shows the current build state at a glance — a small green "CI: passing" image that GitHub generates for your workflow. And in your repository settings you can mark CI as a **required status check**, which means GitHub will physically block merging a pull request until the checks pass. This is what turns CI from a suggestion into a guarantee: main stays green because nothing red can get in.

## The connection to MLOps

Everything here is the foundation for CI in machine-learning projects, which you will meet in the **MLOps Engineer Nanodegree**. An ML pipeline adds more stages to this same skeleton: validating that a newly trained model beats the current baseline before it ships, checking incoming data for schema drift or bad distributions, and publishing model artifacts to a registry. But the shape is identical to what you have built here — an automated pipeline, triggered by change, that refuses to let bad work through. Master ordinary CI first and ML CI is a natural extension.

## A note on documentation

CI keeps code *working*; documentation keeps it *understandable* — the two together are what make a project maintainable. Aim for three habits. A **README** answers the essentials for anyone arriving at your project: what it does, how to install it, how to run it, and how to run the tests. **Docstrings** on your functions and classes describe purpose, parameters, and return values in a consistent style — Google style and NumPy style are the two common conventions; pick one and stay with it. And **comments** in the code should explain *why*, not *what* — the code already says what it does, so use comments for the reasoning that the code cannot express, like why a particular threshold was chosen or why an obvious-looking simplification would be wrong.

## Key takeaways

- CI automatically runs your tests and checks on every change on a clean machine, so broken code is caught before it merges; CD then automates shipping, and CI is its foundation.
- GitHub Actions workflows live in `.github/workflows/*.yml`, structured around `on` (triggers), `jobs`, `runs-on`, and `steps`.
- A steps list mixes reusable actions (`actions/checkout@v4`, `astral-sh/setup-uv@v5`, `actions/setup-python@v5`) with `run` commands (`uv sync`, `ruff check`, `ruff format --check`, `pytest`).
- A `strategy.matrix` runs the same job across multiple Python versions in parallel.
- A README status badge shows build health, and marking CI a *required status check* physically blocks merging until it passes.
- ML CI (in the MLOps Nanodegree) extends this same skeleton with model validation, data checks, and artifact publishing.

## Try it

In a small `uv`-managed project that has at least one test, create the file `.github/workflows/ci.yml` with the minimal workflow shown above. Push it to a GitHub repository and open a pull request; watch the Actions tab run checkout, `uv sync`, `ruff check`, and `pytest` on a clean Ubuntu runner. Deliberately introduce a failing test or an unformatted line, push again, and confirm the check turns red and the PR reports the failure. Then fix it and watch it go green. As a stretch, add a `strategy.matrix` for Python 3.11 and 3.12 and confirm the job now runs twice, and add the status badge to your README.
