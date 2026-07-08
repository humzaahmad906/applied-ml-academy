# 04b — Undoing Changes and GitHub Authentication

You've learned the everyday loop: edit, stage, commit. But two things trip up nearly every beginner. The first is panic: "I broke something — how do I undo this?" The second is the wall you hit on your first `git push`: GitHub asks for a password, you type it, and it gets rejected. Both have clean answers once you understand what's actually happening. Let's fix both.

## The mental model for undoing

Remember the three areas from the last lesson: the **working directory** (files you're editing), the **staging area** (changes marked ready), and the **repository** (committed history). Almost every "undo" question is really one question: *which area do I want to rewind, and how far back?*

- Changed a file but haven't staged it? You're rewinding the working directory.
- Ran `git add` too early? You're rewinding the staging area.
- Already committed and regret it? You're rewinding history.

Match the command to the area and undoing stops feeling like a coin flip.

## Discarding uncommitted changes

Say you edited `train.py`, the change was a bad idea, and you haven't committed yet. To throw away those edits and restore the last committed version:

```bash
git restore train.py
```

This overwrites your working copy with what's in the last commit. The edits are **gone** — `git restore` does not back them up, so only run it when you're sure you want the changes discarded.

If you already staged a file and just want to *unstage* it (move it out of staging, but keep your edits), use:

```bash
git restore --staged train.py
```

`git status` will show the file back as modified-but-unstaged. Nothing is destroyed here — you're only undoing the `git add`.

> **A note on `git checkout`.** Older tutorials use `git checkout -- train.py` to discard changes and `git checkout <branch>` to switch branches. Git 2.23 split that one overloaded command into two clearer ones: `git restore` for files and `git switch` for branches. `git checkout` still works, but `restore` and `switch` say what they do. Prefer them.

## Stashing: park your work and come back

Sometimes you're mid-change when something urgent comes up — a colleague needs you to look at the `main` branch *now*, but your work isn't ready to commit. `git stash` tucks your uncommitted changes away and gives you a clean working directory:

```bash
git stash
```

```
Saved working directory and index state WIP on main: 9f8e7d6 Add task list feature
```

Your edits are safely parked. Do your urgent task, then bring the work back exactly as it was:

```bash
git stash pop
```

`pop` reapplies the most recent stash and removes it from the stash list. You can stash multiple times; list what you've got with:

```bash
git stash list
```

```
stash@{0}: WIP on main: 9f8e7d6 Add task list feature
stash@{1}: WIP on main: a1b2c3d Add project README
```

Think of the stash as a clipboard for work-in-progress. It's the safe, non-destructive way to switch context without committing half-finished code.

## Fixing commits

### Amend the last commit

Made a commit and immediately noticed a typo in the message, or forgot to include a file? Fix the most recent commit in place:

```bash
git commit --amend -m "Fix login crash when email field is empty"
```

If you forgot a file, stage it first with `git add`, then run `git commit --amend`. Only amend commits you **haven't pushed yet** — amending rewrites the commit, and rewriting shared history causes problems for teammates.

### Reset: move the branch pointer backward

`git reset` moves your branch back to an earlier commit. It comes in three flavors, and the difference between them is exactly *how much they touch your files* — so read carefully:

- `git reset --soft HEAD~1` — undo the last commit but **keep all its changes staged**. Use this to recombine or rewrite a commit. Nothing is lost.
- `git reset --mixed HEAD~1` — undo the last commit and unstage its changes, but **keep them in your working directory**. This is the default. Nothing is lost; you just go back to before you staged.
- `git reset --hard HEAD~1` — undo the last commit **and permanently delete its changes from your working directory.**

That last one deserves a loud warning. **`git reset --hard` destroys uncommitted work with no confirmation and no trash can.** If you have edits you haven't committed and you run `--hard`, they are gone. If you reset past commits you meant to keep, those changes disappear from your working files too. Before you ever run `--hard`, ask yourself: is there anything here I haven't committed that I'd cry about losing? If the answer is anything but a confident "no," use `--soft` or `--mixed` instead, or stash first. `HEAD~1` means "one commit back"; `HEAD~2` means two, and so on.

### Revert: the safe undo for pushed commits

Once a commit is pushed and other people may have pulled it, rewriting history (with `reset` or `amend`) is rude and dangerous — it desyncs everyone. The safe tool is `git revert`:

```bash
git revert a1b2c3d
```

Instead of deleting the commit, `revert` creates a **new** commit that undoes the changes from `a1b2c3d`. History stays intact and moves forward, so nobody's clone breaks. This is the correct way to undo something that's already shared. Rule of thumb: **`reset` for local, unpushed mistakes; `revert` for anything already public.**

## The safety net: reflog

Here's the reassuring part. Even after a scary `reset --hard`, Git usually still remembers where you were. The **reflog** records every place `HEAD` has pointed — every commit, reset, and checkout — even ones no longer reachable by any branch:

```bash
git reflog
```

```
9f8e7d6 (HEAD -> main) HEAD@{0}: reset: moving to HEAD~1
3c4d5e6 HEAD@{1}: commit: Add experimental feature
9f8e7d6 HEAD@{2}: commit: Add task list feature
```

See that `3c4d5e6` you thought you destroyed? Bring it back:

```bash
git reset --hard 3c4d5e6
```

The reflog is your undo-of-the-undo. It's why "I lost a commit" is almost never true for committed work — Git keeps reflog entries for about 90 days by default. (It does *not* rescue uncommitted edits wiped by `reset --hard`, which is why committing often matters.)

## GitHub authentication: why your first push failed

You created a repo on GitHub, ran `git push`, GitHub prompted for a username and password, you typed your account password — and it was rejected. This is not a bug. **GitHub removed password authentication for Git operations in 2021.** Typing your login password will never work. You need one of the methods below.

### Option A: gh CLI (the easy path)

If you install the official GitHub CLI (`gh`), authentication is one command:

```bash
gh auth login
```

It walks you through a browser login and configures Git to use HTTPS with a token automatically — no keys or tokens to manage by hand. For most beginners on a personal machine, this is the least painful route. Start here if you're unsure.

### Option B: SSH keys (great once set up)

SSH uses a key pair: a private key that stays on your machine and a public key you hand to GitHub. Generate a modern `ed25519` key:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to accept the default location (`~/.ssh/id_ed25519`). Setting a passphrase is optional but recommended. Now copy the **public** key — the `.pub` file, never the private one:

```bash
cat ~/.ssh/id_ed25519.pub
```

On GitHub, go to **Settings → SSH and GPG keys → New SSH key**, give it a label like "Work laptop," and paste the key. Then test the connection:

```bash
ssh -T git@github.com
```

```
Hi ada! You've successfully authenticated, but GitHub does not provide shell access.
```

That "successfully authenticated" line means you're set. Push and pull over SSH now work without ever typing a password. Make sure your remote uses the SSH URL (`git@github.com:user/repo.git`), not the HTTPS one.

### Option C: HTTPS with a Personal Access Token

If you prefer HTTPS, you authenticate with a **Personal Access Token (PAT)** in place of a password. Create one at **Settings → Developer settings → Personal access tokens → Fine-grained tokens**. Fine-grained tokens are the modern choice: you scope them to specific repositories and specific permissions (e.g. read/write contents), and they expire on a date you set. Generate the token, copy it (you only see it once), and paste it when Git prompts for a password on your next push. A credential helper can cache it so you don't paste it every time.

## Key takeaways

- Every "undo" maps to an area: `git restore <file>` discards working-dir edits, `git restore --staged <file>` unstages.
- `git stash` / `git stash pop` parks work-in-progress safely so you can switch tasks.
- `git commit --amend` fixes the last (unpushed) commit; `reset --soft` keeps changes staged, `--mixed` keeps them unstaged, and **`--hard` permanently deletes uncommitted work — treat it with caution.**
- Use `reset` for local unpushed mistakes; use `git revert` for anything already pushed, because it undoes safely with a new commit.
- `git reflog` recovers "lost" committed work — Git remembers where HEAD has been.
- GitHub killed password auth. Use `gh auth login` (easiest), an `ed25519` SSH key, or a fine-grained Personal Access Token over HTTPS.

## Try it

Practice undoing in a throwaway repo so mistakes cost nothing:

1. In a test repo, edit a tracked file, then run `git restore <file>` and confirm the edit is gone.
2. Edit the file again, `git add` it, then `git restore --staged <file>` — check `git status` shows it unstaged but still modified.
3. Make a change, run `git stash`, confirm the working directory is clean, then `git stash pop` to bring it back.
4. Commit something, run `git reset --soft HEAD~1`, and observe the change sitting back in staging (nothing lost).
5. Run `git reflog` and read the trail of everywhere HEAD has been — that's your safety net.
6. Set up authentication for real: run `gh auth login`, or generate an `ed25519` key and add it to GitHub, then `git push` and watch it succeed without a password prompt.
