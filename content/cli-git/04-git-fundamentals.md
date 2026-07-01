# 04 — Git Fundamentals

You've been editing files. But files change, mistakes happen, and "final_v2_REALLY_final.txt" is not a backup strategy. **Git** is version control: it records snapshots of your project over time so you can see what changed, when, why, and roll back if needed. It's the tool that lets teams work on the same code without chaos. Let's learn the core loop.

## Setting up once

Before your first commit, tell Git who you are. This gets attached to every snapshot you make:

```bash
git config --global user.name "Ada Lovelace"
git config --global user.email "ada@example.com"
```

You only do this once per machine. The `--global` flag applies it everywhere.

## Creating a repository

A **repository** (repo) is a project folder that Git is tracking. Turn any folder into one with `git init`:

```bash
mkdir myproject
cd myproject
git init
```

```
Initialized empty Git repository in /Users/ada/myproject/.git/
```

That hidden `.git` folder is where Git stores all history. Don't touch it — just know it's the "brain" of the repo. To check the state of things at any time:

```bash
git status
```

```
On branch main

No commits yet

nothing to commit (working tree clean)
```

`git status` is the command you'll run most. Whenever you're unsure what Git thinks is going on, run it.

## The three areas

Git has a mental model worth internalizing early. A file lives in one of three places:

1. **Working directory** — the actual files you edit.
2. **Staging area** — a holding zone for changes you've marked as ready to save.
3. **Repository** — the permanent history of committed snapshots.

The flow is: edit files → **stage** the ones you want → **commit** them into history. Staging lets you commit *some* changes now and others later, so each snapshot tells a clean story.

## Staging and committing

Create a file, then watch Git notice it:

```bash
echo "# My Project" > README.md
git status
```

```
On branch main
No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        README.md
```

Git sees `README.md` but isn't tracking it yet — it's **untracked**. Move it to the staging area with `git add`:

```bash
git add README.md
git status
```

```
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   README.md
```

Now it's **staged**. Record it permanently with `git commit`, attaching a message that explains *why* the change was made:

```bash
git commit -m "Add project README"
```

```
[main (root-commit) a1b2c3d] Add project README
 1 file changed, 1 insertion(+)
```

That `a1b2c3d` is the **commit hash** — a unique ID for this snapshot. You've made your first commit.

To stage everything changed at once, use `git add .` (the `.` means "all files here"). But stage deliberately — a commit should be one logical change, not a dumping ground.

## Writing good commit messages

A message like "stuff" or "fix" helps no one, including future you. Write a short, clear summary in the present tense:

```bash
git commit -m "Fix login crash when email field is empty"
```

Good messages describe *what changed and why* so that reading the history later actually tells a story.

## Seeing what changed

Before committing, review exactly what you altered with `git diff`:

```bash
git diff
```

```
diff --git a/README.md b/README.md
index e69de29..b5f3a1c 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 # My Project
+A tool for tracking tasks.
```

Lines starting with `+` were added, `-` were removed. `git diff` shows unstaged changes; `git diff --staged` shows what's staged and ready to commit.

## Viewing history

Every commit is preserved. Browse them with `git log`:

```bash
git log
```

```
commit a1b2c3d4e5f6... (HEAD -> main)
Author: Ada Lovelace <ada@example.com>
Date:   Wed Jul 1 09:14:22 2026

    Add project README
```

For a compact overview, the one-line format is far more readable:

```bash
git log --oneline
```

```
9f8e7d6 Add task list feature
a1b2c3d Add project README
```

This is your project's timeline — every snapshot, newest at the top.

## Ignoring files with `.gitignore`

Some files should never be committed: secrets, huge build outputs, temporary junk, dependency folders. Create a file named `.gitignore` and list patterns to skip:

```bash
node_modules/
*.log
.env
.DS_Store
```

Git will now pretend those files don't exist for tracking purposes. `git status` won't nag about them, and `git add .` won't stage them. The `.gitignore` file itself *should* be committed, so the whole team ignores the same things:

```bash
git add .gitignore
git commit -m "Add .gitignore"
```

A well-tuned ignore list keeps your history clean and your secrets out of version control — which matters enormously.

## The everyday loop

Ninety percent of your Git usage is this rhythm:

```bash
git status              # what's going on?
git add <files>         # stage what's ready
git diff --staged       # double-check it
git commit -m "..."     # save the snapshot
git log --oneline       # confirm it landed
```

Internalize that loop and Git stops being scary.

## Key takeaways

- `git init` starts tracking a folder; `git status` tells you the current state (run it constantly).
- Files flow through three areas: **working directory** → **staging** (`git add`) → **repository** (`git commit`).
- Commit messages should explain *what and why*, in the present tense.
- `git diff` shows unstaged changes, `git diff --staged` shows staged ones; `git log --oneline` shows history.
- List files to never track in `.gitignore` — and commit that file so the whole team shares it.

## Try it

Build a repo from nothing:

1. Create a new folder, `cd` into it, and run `git init`.
2. Create a `README.md` with a title, check `git status`, then stage and commit it with a clear message.
3. Add a second line to the README, run `git diff` to see the change, then commit it separately.
4. Create a `.gitignore` that ignores `*.log`, then create a file `debug.log` and confirm `git status` does *not* show it.
5. Run `git log --oneline` and read your project's history back. You built that timeline yourself.
