# 06 — GitHub and Pull Requests

Everything so far has lived on your own machine. But code is meant to be shared, backed up, and worked on by teams. **GitHub** is a website that hosts Git repositories online, and it's where most collaboration happens. In this final piece you'll connect your local repo to a remote one, push and pull changes, and learn the workflow that teams use every day: the **pull request**.

## Remotes: your repo, online

A **remote** is a copy of your repository hosted somewhere else — usually on GitHub. Your local repo and the remote sync back and forth. The default remote is conventionally named `origin`.

If you created a repo locally, you connect it to a new empty GitHub repo like this:

```bash
git remote add origin https://github.com/ada/myproject.git
git remote -v
```

```
origin  https://github.com/ada/myproject.git (fetch)
origin  https://github.com/ada/myproject.git (push)
```

`git remote add` links the name `origin` to that URL. `git remote -v` confirms the connection.

## Cloning: getting an existing repo

More often, the repo already exists online and you want a local copy. `git clone` downloads the entire repo — all history included — and wires up the remote automatically:

```bash
git clone https://github.com/ada/myproject.git
cd myproject
```

```
Cloning into 'myproject'...
remote: Enumerating objects: 24, done.
Receiving objects: 100% (24/24), done.
```

After cloning, `origin` is already set up. You're ready to work immediately.

## Pushing: sending commits up

Once you've committed locally, `git push` uploads those commits to the remote. The first push of a branch sets its upstream with `-u`:

```bash
git push -u origin main
```

```
Enumerating objects: 5, done.
To https://github.com/ada/myproject.git
 * [new branch]      main -> main
```

The `-u` tells Git to remember that your local `main` tracks `origin/main`. After that first time, you can just run `git push` — Git knows where it goes.

## Pulling: getting others' changes

When teammates push their work, you need it locally. `git pull` fetches new commits from the remote and merges them into your branch:

```bash
git pull
```

```
Updating a1b2c3d..9f8e7d6
Fast-forward
 app.py | 12 ++++++++++++
 1 file changed, 12 insertions(+)
```

Get in the habit of pulling *before* you start work each session. It keeps you current and reduces conflicts. If both you and a teammate changed the same lines, `git pull` can raise a merge conflict — you resolve it exactly as you learned: edit the file, remove the markers, `git add`, `git commit`.

## The pull request workflow

Here's the heart of team collaboration. You generally do **not** push straight to `main`. Instead:

1. Pull the latest `main`.
2. Create a feature branch for your work.
3. Commit your changes on that branch.
4. Push the branch to GitHub.
5. Open a **pull request** (PR) asking to merge your branch into `main`.
6. A teammate reviews it, suggests changes, and approves.
7. The branch is merged, then deleted.

A **pull request** is a proposal: "here are my changes — please review and merge them." It's where discussion, review, and quality checks happen before code reaches the shared `main`.

## Walking through a PR

Start from an up-to-date `main`, branch, and work:

```bash
git switch main
git pull
git switch -c add-export
echo "def export(): pass" > export.py
git add export.py
git commit -m "Add export function"
```

Now push the branch up:

```bash
git push -u origin add-export
```

```
To https://github.com/ada/myproject.git
 * [new branch]      add-export -> add-export
remote: Create a pull request for 'add-export' on GitHub by visiting:
remote:   https://github.com/ada/myproject/pull/new/add-export
```

Notice GitHub even prints a link to open the PR. Follow it (or click "Compare & pull request" on the GitHub site), give the PR a clear title and description of *what* changed and *why*, and submit it.

## What happens during review

Once the PR is open, a reviewer looks at the diff — exactly the `+`/`-` view you saw with `git diff`, but on the web with the ability to comment on individual lines. They might:

- **Approve** it if it looks good.
- **Request changes** with specific comments.

If changes are requested, you don't open a new PR. You just make more commits on the same branch and push again:

```bash
# edit files based on feedback
git add export.py
git commit -m "Handle empty export case"
git push
```

The PR updates automatically with your new commits. Once approved, someone clicks **Merge** and your work lands in `main`. Then delete the branch — both on GitHub (a button) and locally:

```bash
git switch main
git pull
git branch -d add-export
```

That final `git pull` brings the freshly merged code down to your machine, and you're ready for the next task.

## Why this workflow wins

It might feel like a lot of steps for a small change, but each one earns its place. Branches keep `main` stable. Pull requests create a moment for a second set of eyes to catch bugs before they ship. The discussion on a PR becomes a record of *why* a decision was made. And because everything is on the remote, your work is backed up and visible to the team. This exact loop — branch, commit, push, PR, review, merge — is the daily rhythm of nearly every software team, and you now know all of it.

## Key takeaways

- A **remote** (usually `origin`) is your repo hosted online; `git remote add` links one, `git clone` downloads an existing one.
- `git push` uploads commits (`-u origin branch` the first time); `git pull` downloads and merges others' commits — pull before you start work.
- Don't commit straight to `main`. Branch, commit, push the branch, then open a **pull request**.
- A PR is a reviewable proposal to merge; respond to feedback by pushing **more commits to the same branch** — the PR updates itself.
- After merge, pull the latest `main` and delete the finished branch locally and on GitHub.

## Try it

If you have a GitHub account, run the full loop on a practice repo:

1. Create an empty repo on GitHub, then `clone` it to your machine.
2. On `main`, create a `README.md`, commit it, and `git push -u origin main`.
3. Create a branch `add-feature`, add a small file, commit, and push the branch with `-u`.
4. Open a pull request on GitHub from `add-feature` into `main`; write a title and a short description of the change.
5. Merge the PR on GitHub, then locally switch to `main`, `git pull`, and delete the merged branch. Confirm the new file is now on `main`.
