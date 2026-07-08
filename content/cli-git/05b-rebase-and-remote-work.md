# 05b — Rebase and Working on Remote Machines

You know how to branch and merge. This lesson adds two things you'll reach for constantly as an ML engineer: **rebase**, an alternative way to combine branches that keeps history tidy, and the everyday skills of **working on a remote GPU box** — logging in, moving files, and keeping a long training job alive after you close your laptop. We'll finish with **environment variables**, the glue that makes your tools and CUDA find each other.

## Rebase: replaying commits for a straight line

Merging and rebasing solve the same problem — integrating changes from one branch into another — but they tell the story differently. A merge ties two lines together with a merge commit, preserving the exact shape of what happened. A **rebase** instead lifts your branch's commits off and *replays* them one by one on top of the latest `main`, as if you'd started your work from there in the first place.

Say you branched off `main` to build a feature, and meanwhile `main` moved ahead. To rebase your feature branch onto the new `main`:

```bash
git switch add-search
git rebase main
```

```
Successfully rebased and updated refs/heads/add-search.
```

Now your commits sit neatly on top of the current `main`, and the history reads as one clean, straight line — no merge commit, no fork in the graph. When you later merge `add-search` into `main`, it fast-forwards.

## Merge or rebase — which to use

Both are correct; they optimize for different things.

- **Merge** keeps a true record. The merge commit shows exactly when two lines came together. History is honest but can look tangled when many branches merge.
- **Rebase** keeps history linear and easy to read with `git log`, as if development happened in sequence. The cost: it *rewrites* commits — each replayed commit gets a brand-new ID, because its parent changed.

A common workflow blends them: use rebase privately to tidy your own feature branch before anyone sees it, then use merge to fold the polished branch into shared `main`. That gives you a readable feature and an honest integration point.

## The golden rule: never rebase shared history

This is the one rule you must not break, so read it slowly.

**Never rebase commits that you have already pushed and that other people may have based work on.**

Here's why it matters. Rebasing doesn't move your commits — it *replaces* them with new copies that have new IDs. If those original commits only ever lived on your machine, no harm done: you're rewriting a story nobody else has read. But if you already pushed them, your teammates' clones still point at the *old* commits. When you force your rewritten branch up, their history and yours no longer agree on what happened. Git sees two divergent sets of commits, people start merging the duplicates back in, and you get a tangled mess that's genuinely painful to untangle.

The safe boundary is simple: rebase freely while your work is **local and unpushed**. Once a branch is **pushed and shared**, integrate with `merge`, not rebase. If you're ever unsure whether others have your commits, assume they do and merge.

### A quick word on interactive rebase

Rebase has a cleanup mode, `git rebase -i` (interactive), for grooming *your own local* commits before sharing — squashing five messy "wip" commits into one clean commit, fixing a typo in a commit message, or reordering. You'll open it like this:

```bash
git rebase -i HEAD~3
```

That lets you edit the last three commits. It's a genuinely useful tool, but it rewrites history, so the golden rule applies with full force: interactive-rebase only commits you haven't pushed. We won't go deeper here — just know it exists for tidying local work.

## Working on a remote GPU machine

Most real training doesn't happen on your laptop — it happens on a rented or shared box with a real GPU. The daily loop is: connect, move your code over, launch the job, and make sure it survives when your connection drops.

### Connecting with SSH

`ssh` opens a secure shell on the remote machine. You give it a username and an address:

```bash
ssh humza@203.0.113.42
```

Now your terminal is *on the server*. Commands you type run there, against its GPU. Type `exit` to return to your laptop.

### Moving files: scp and rsync

To copy a single file up to the server, `scp` (secure copy) works like `cp` but across the network:

```bash
scp train.py humza@203.0.113.42:~/project/
```

For anything bigger — a whole project folder, or syncing changes repeatedly — `rsync` is the workhorse. It transfers only what changed, so re-syncing after a small edit is fast:

```bash
rsync -avz --progress ./project/ humza@203.0.113.42:~/project/
```

Here `-a` preserves file structure and permissions, `-v` is verbose, `-z` compresses in transit, and `--progress` shows a live bar. To pull results (like checkpoints or logs) back down, just swap the order:

```bash
rsync -avz humza@203.0.113.42:~/project/checkpoints/ ./checkpoints/
```

### Keeping a job alive: tmux

Here's the trap that catches everyone once. You SSH in, start a job that'll run for six hours, and close your laptop — or your WiFi hiccups. The SSH session dies, and every process tied to it dies with it. Your six-hour run is gone.

The fix is a **terminal multiplexer** like `tmux` (or the older `screen`). It creates a session that lives *on the server itself*, independent of your connection. You start your job inside it, detach, and even if your laptop disconnects, the job keeps running. Reconnect later and pick up exactly where you left off.

Start a named session:

```bash
tmux new -s train
```

Your prompt now runs inside tmux. Launch the job, and it's good practice to tee the output to a log file so you have a record:

```bash
python train.py 2>&1 | tee train.log
```

Now **detach**: press `Ctrl-b`, release, then press `d`. You're back to the plain server shell, and tmux reports the session is detached — but your job is still running inside it. At this point you can safely `exit` the SSH connection and close your laptop.

When you come back, SSH in again and **reattach**:

```bash
tmux attach -t train
```

You're back in the session, watching your job as if you never left. To list sessions if you forget the name, run `tmux ls`. This one habit — always launch long jobs inside tmux — will save you from losing training runs.

## Environment variables and PATH

An **environment variable** is a named value the shell and the programs it launches can read. You set one with `export`:

```bash
export WANDB_API_KEY=abc123
python train.py
```

Now `train.py` (and anything it calls) can read `WANDB_API_KEY`. This is how you pass secrets and config to tools without hard-coding them.

The most important variable is **`PATH`** — a colon-separated list of directories the shell searches when you type a command. When you run `python`, the shell walks through `PATH` in order and runs the first `python` it finds. That's why *which* Python runs depends entirely on `PATH`. You can see it:

```bash
echo $PATH
```

```
/usr/local/cuda/bin:/home/humza/miniconda3/bin:/usr/bin:/bin
```

To add a directory — say, CUDA's tools so `nvcc` is findable — prepend it (prepending wins over existing entries):

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

The `:$PATH` on the end matters: it keeps everything that was already there. Drop it and you'd wipe out your PATH, and suddenly even `ls` won't be found.

### Making it stick: .bashrc / .zshrc

An `export` only lasts for the current shell. Close the terminal and it's gone. To set something every time you open a shell, add the line to your shell's startup file — `~/.bashrc` for bash, `~/.zshrc` for zsh (macOS default). Anything in there runs automatically on each new shell.

This is exactly how the pieces connect. Activating a conda or venv environment works by putting its `bin/` directory at the front of `PATH`, so its Python and packages shadow the system ones. CUDA setup on a GPU box is usually two lines in `~/.bashrc` — one adding `/usr/local/cuda/bin` to `PATH`, one adding its libraries to `LD_LIBRARY_PATH` — so every new shell can find the GPU toolkit without you thinking about it.

## Key takeaways

- **Rebase** replays your commits on top of another branch for a clean, linear history; **merge** ties branches together with a merge commit that preserves the true shape.
- **Golden rule:** rebase only local, unpushed commits. Never rebase history that's been pushed and shared — it rewrites commit IDs and breaks everyone else's copy.
- `git rebase -i` cleans up your own local commits (squash, reorder, reword) — same golden rule applies.
- On a remote box: `ssh` to connect, `scp`/`rsync` to move files, and **`tmux`** to keep long jobs alive across disconnects (`Ctrl-b` `d` to detach, `tmux attach -t name` to reattach).
- **Environment variables** (`export NAME=value`) pass config to programs; **`PATH`** decides which command runs. Add to `PATH` with `export PATH=/new/dir:$PATH`, and make it permanent in `~/.bashrc` or `~/.zshrc`. This is the same mechanism behind activating envs and finding CUDA.

## Try it

Practice the remote-work reflexes locally (no server needed for the tmux part):

1. Run `echo $PATH` and read the list of directories out loud — that's your command search order.
2. Set a variable: `export GREETING=hello`, then `echo $GREETING`. Open a new terminal and run `echo $GREETING` again — notice it's empty, because `export` didn't persist.
3. Start a tmux session with `tmux new -s practice`, run a slow command like `sleep 300`, then detach with `Ctrl-b` `d`. Run `tmux ls` to see it still alive, then `tmux attach -t practice` to reattach.
4. In a scratch repo, make two commits on a branch, then run `git rebase -i HEAD~2` and squash them into one. Confirm with `git log --oneline` that two commits became one — all while the work is still local and unpushed.
