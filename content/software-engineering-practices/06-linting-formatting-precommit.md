# 06 — Linting, Formatting, and Pre-commit Hooks

Code that works is not the same as code that is clean, consistent, and free of small mistakes. Two kinds of tools keep a codebase healthy: **linters**, which find likely bugs and bad patterns, and **formatters**, which enforce a consistent style automatically. In 2026 both jobs are handled by one remarkably fast tool, **ruff**. This lesson shows you how to lint and format with ruff, and then how to wire these checks into git with **pre-commit** so that unclean code simply cannot be committed.

## Linting versus formatting

These two words are often confused, so it helps to separate them clearly.

- **Formatting** is about appearance: indentation, line length, where spaces and blank lines go, how long expressions wrap. Formatting changes never alter what your code *does* — they make it look consistent so nobody argues about style in code review.
- **Linting** is about correctness and quality: an unused import, a variable you assigned but never used, a bare `except`, a likely bug. A linter reads your code and warns you about patterns that are wrong or risky.

You want both, and historically that meant juggling several tools: `flake8` for linting, `isort` for import ordering, `black` for formatting, `pyupgrade` for modernizing syntax. ruff replaces all of them.

## ruff: one fast tool

**ruff** is a single tool that does the work of that whole stack, and it runs roughly a hundred times faster than the tools it replaces — fast enough that checking a large project feels instant. It has two main commands.

To lint (find problems):

```bash
ruff check
```

This scans your project and reports issues — unused imports, undefined names, style-guide violations. Many can be fixed automatically:

```bash
ruff check --fix
```

To format (enforce consistent style):

```bash
ruff format
```

This rewrites your files into a consistent style, the same way `black` does — the output is near-identical, so if you know black you already know what ruff format produces. `black` is still widely used and perfectly good; ruff format simply folds that job into the same tool as your linter.

## Configuring ruff in pyproject.toml

ruff is configured in the same `pyproject.toml` you have been using throughout this course. You choose which rule sets to enable and which specific rules to ignore:

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
```

Under `[tool.ruff.lint]`, `select` turns on families of rules by their letter codes: `E` for style errors, `F` for likely bugs (from the old flake8), `I` for import sorting (replacing isort), `UP` for modernizing old syntax (replacing pyupgrade). `ignore` switches off specific rules you disagree with. Under `[tool.ruff.format]`, you tune formatting choices like quote style. The top-level `[tool.ruff]` sets shared options like line length and the Python version to target.

## pre-commit: enforce checks automatically

Running `ruff check` and `ruff format` by hand works only if you remember to. The reliable approach is to make git run them for you on every commit, so unclean code can never slip in. That is what **pre-commit** does: it manages "hooks" — checks that fire automatically at the moment you commit.

You configure it with a file named `.pre-commit-config.yaml` at your project root:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.0
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
```

A few things to notice. The hooks come from `astral-sh/ruff-pre-commit`, the official ruff hook repository. Two hooks run: `ruff-check` (linting) and `ruff-format` (formatting). Order matters — put `ruff-check` *before* `ruff-format`, so linting fixes are applied first and then the formatter tidies the final result. And `rev: v0.15.0` **pins the version**: everyone on the team runs the exact same ruff, so results are identical rather than drifting as versions change.

## Wiring it into git

The config file alone does nothing until you install the hooks into your local git:

```bash
pre-commit install
```

This one-time command tells git to run your hooks before each commit. From now on, when you `git commit`, pre-commit runs ruff on the files you are committing. If ruff finds a problem it cannot auto-fix, the commit is blocked and you see exactly what to fix. If it auto-fixes something, the commit stops so you can review and re-stage the change. Either way, code that violates your standards never makes it into history.

You can also run all the hooks manually across the whole project — useful the first time you set it up:

```bash
pre-commit run --all-files
```

These same hooks are what a continuous-integration pipeline runs on every pull request, so what passes locally passes in CI too. (You will set up GitHub Actions for that in a later lesson.)

## Key takeaways

- Formatting governs appearance (never behavior); linting finds likely bugs and bad patterns — you want both.
- **ruff** is a single, ~100x-faster tool that replaces flake8, isort, black, and pyupgrade.
- `ruff check` lints (add `--fix` to auto-fix) and `ruff format` formats; its output is near-identical to black, which remains common.
- Configure ruff in `pyproject.toml`: `select`/`ignore` rule sets under `[tool.ruff.lint]`, style under `[tool.ruff.format]`.
- **pre-commit** runs checks automatically at commit time via `.pre-commit-config.yaml` using the `astral-sh/ruff-pre-commit` hooks.
- Put `ruff-check` before `ruff-format` in the hook order, and pin the version with `rev:` so everyone runs the same ruff.
- `pre-commit install` wires the hooks into git so unclean code can't be committed; `pre-commit run --all-files` checks everything at once.

## Try it

In one of your existing projects, add ruff as a dev dependency (`uv add --dev ruff`) and add `[tool.ruff]`, `[tool.ruff.lint]`, and `[tool.ruff.format]` sections to your `pyproject.toml`, selecting the `E`, `F`, `I`, and `UP` rule sets. Run `ruff check` and read what it finds, then run `ruff check --fix` and `ruff format` and watch it clean up. Next, create a `.pre-commit-config.yaml` using the `astral-sh/ruff-pre-commit` hooks (check-before-format, version pinned), run `pre-commit install`, and then deliberately commit a file with an unused import. Confirm that the commit is blocked or the import is stripped automatically before your code reaches the repository.
