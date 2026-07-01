# 05 — Branching and Merging

So far your commits form a single straight line. But real work is rarely linear — you want to try a new feature without breaking the working version, or two people need to build different things at once. **Branches** make that possible: they let you diverge, experiment freely, and later fold your work back in. This is where Git goes from "backup tool" to "collaboration engine."

## What a branch is

A **branch** is just a movable pointer to a commit. Your project starts on a branch called `main`. When you make a new branch, you get an independent line of development: commits on it don't touch `main` until you decide to combine them.

Think of it like a parallel timeline. You split off, do work, and either merge it back or throw it away — all without disturbing the original.

## Creating and switching branches

The modern command for this is `git switch`. Create a branch and move onto it in one step:

```bash
git switch -c add-search
```

```
Switched to a new branch 'add-search'
```

The `-c` flag means "create." To see all branches (the current one is marked with `*`):

```bash
git branch
```

```
  main
* add-search
```

Switch back to an existing branch without `-c`:

```bash
git switch main
```

Any commits you make now belong to whichever branch you're standing on. Nothing you do on `add-search` affects `main` until you merge.

## Working on a branch

Let's do real work on the feature branch:

```bash
git switch add-search
echo "def search(): pass" > search.py
git add search.py
git commit -m "Add search stub"
```

```
[add-search 7c3d1a9] Add search stub
 1 file changed, 1 insertion(+)
```

If you now switch back to `main` and run `ls`, `search.py` won't be there — it only exists on `add-search`. That's the point: your experiment is isolated. Switch to the feature branch and it reappears. Git swaps the files under you as you move between branches.

## Merging: bringing work back together

When the feature is ready, you **merge** it into `main`. First switch to the branch you want to merge *into*, then merge the other branch *in*:

```bash
git switch main
git merge add-search
```

```
Updating a1b2c3d..7c3d1a9
Fast-forward
 search.py | 1 +
 1 file changed, 1 insertion(+)
```

"Fast-forward" means `main` hadn't changed since you branched, so Git simply slid the pointer forward. Now `search.py` is on `main`. Once merged, you can delete the finished branch:

```bash
git branch -d add-search
```

```
Deleted branch add-search (was 7c3d1a9).
```

Deleting a merged branch is safe — the commits live on in `main`. Branches are cheap; make them freely and clean them up when done.

## When histories diverge

Fast-forward only happens when `main` didn't move. Often it did — someone else committed, or you committed on `main` yourself. Then Git creates a **merge commit** that ties the two lines together. That's normal and Git handles it automatically... *unless* both branches changed the same lines. Then you get a **conflict**.

## Resolving conflicts

A conflict means Git can't decide which version wins, so it asks you. Say both `main` and a branch edited the same line of `greeting.txt`:

```bash
git merge add-greeting
```

```
Auto-merging greeting.txt
CONFLICT (content): Merge conflict in greeting.txt
Automatic merge failed; fix conflicts and then commit the result.
```

Don't panic — this is routine. Open the conflicted file and you'll see Git's markers:

```text
<<<<<<< HEAD
Hello, world!
=======
Hi there, world!
>>>>>>> add-greeting
```

Read this as: everything between `<<<<<<< HEAD` and `=======` is what's currently on your branch (`main`); everything between `=======` and `>>>>>>>` is what's coming from `add-greeting`. Your job is to edit the file into the version you actually want, deleting all three marker lines:

```text
Hi there, world!
```

Then stage the resolved file and commit to complete the merge:

```bash
git add greeting.txt
git commit -m "Merge add-greeting, keep friendlier greeting"
```

```
[main 5e6f7a8] Merge add-greeting, keep friendlier greeting
```

That's it — conflict resolved. The key insight: a conflict isn't an error, it's Git asking a question it can't answer for you. You decide, remove the markers, add, commit.

## Checking your work mid-merge

If you get lost during a conflict, `git status` guides you — it lists which files are still "unmerged":

```bash
git status
```

```
You have unmerged paths.
  (fix conflicts and run "git commit")

Unmerged paths:
  (use "git add <file>..." to mark resolution)
        both modified:   greeting.txt
```

And if you decide the merge was a mistake and want to bail out entirely before finishing:

```bash
git merge --abort
```

That returns you to exactly where you were before the merge started. A safe escape hatch worth remembering.

## Why branch at all?

The habit is: `main` always stays working. Every new feature, fix, or experiment gets its own branch. You break things freely there, and only merge back when it's solid. This keeps the shared version stable and makes it easy to abandon ideas that don't pan out — just delete the branch. It's the same discipline whether you work alone or on a team.

## Key takeaways

- A **branch** is a movable pointer to a commit — an isolated line of work that doesn't affect `main` until merged.
- `git switch -c name` creates and moves to a branch; `git switch name` moves to an existing one; `git branch` lists them.
- `git merge name` folds a branch into your current one; a clean case is a **fast-forward**.
- A **conflict** appears when both sides changed the same lines. Edit the file to the version you want, delete the `<<<<<<<` / `=======` / `>>>>>>>` markers, then `git add` and `git commit`.
- `git status` guides you through an in-progress merge; `git merge --abort` cancels it safely.

## Try it

Create a conflict on purpose and resolve it:

1. In a repo, commit a file `notes.txt` containing the single line `original`.
2. Create a branch `edit-a`, change the line to `version A`, and commit.
3. Switch back to `main`, change the same line to `version B`, and commit.
4. Merge `edit-a` into `main`. When the conflict appears, open the file and read the markers.
5. Edit it to a final version of your choosing, remove all markers, then `git add` and `git commit` to finish. Confirm with `git log --oneline` that the merge landed.
