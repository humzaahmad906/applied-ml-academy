# 10 — Code Review and Pull-Request Discipline

Writing code is only half of professional software work; the other half is getting that code reviewed and merged by a team. The pull request — a proposal to merge your changes, opened for others to read before it lands — is where that happens. This lesson is about the *practice* and *culture* of pull requests and code review: what makes a change easy to review, what makes a review useful, and how teams keep the whole thing collaborative rather than combative. It is one of the highest-leverage skills you can build, because it is how your work becomes the team's work.

A note on scope: the git mechanics — creating branches, committing, pushing, opening the PR — are taught in the **Command Line & Git course**. This lesson deliberately teaches zero git commands. It is about human judgment, not tooling.

## What a good pull request looks like

The single most important property of a pull request is that it is **small and focused**. A reviewer can hold about a screen's worth of change in their head at once. A 60-line PR gets a careful, thoughtful review in ten minutes; a 2,000-line PR gets a rubber-stamp "looks good" because no human can actually verify it. Small PRs are not a nicety — they are the difference between real review and theatre.

Focused means **one concern per PR**. If you are fixing a bug, fix the bug; do not also rename three variables and upgrade a dependency along the way. When a PR does one thing, its title says what that thing is, its history is clean, and if it needs to be reverted later, reverting it does not undo unrelated work.

A good PR communicates in three places:

- **The title** is a short, descriptive summary — "Add retry logic to model download," not "fixes" or "updates."
- **The body explains *why*, not just *what*.** The diff already shows what changed. What the reviewer cannot see is the reasoning: what problem this solves, what approach you chose, what alternatives you rejected, and anything you are unsure about. "The download failed intermittently on slow networks; this adds three retries with backoff. I considered raising the timeout instead but that just made failures slower" — that paragraph makes review effortless.
- **The linked context** — the issue or ticket this addresses, so the change has a traceable reason for existing.

Here is what that looks like in practice — a body a reviewer can act on without asking a single follow-up question:

```markdown
## What
Add three retries with exponential backoff to `download_model()`.

## Why
Model downloads fail intermittently on slow networks (see #214), and a
single failure aborts the whole training run. Retrying recovers cleanly.

## Alternatives considered
Raising the timeout — rejected, because a truly slow link would just
fail more slowly rather than succeed.

## Testing
Added `test_download_retries_then_succeeds`; ran the full suite locally.
```

Contrast that with a title of "fixes" and an empty body: the reviewer would have to reverse-engineer every one of those points from the diff, which is exactly the work a good description saves them.

## What a good review looks like

Reviewing well is a skill in its own right. Start by **reading the whole diff** before commenting on any single line — a comment about line 12 may be answered by line 40. Then evaluate on two axes:

- **Correctness.** Does it do what it claims? Are the edge cases handled? Are the errors caught specifically (Lesson 08)? Is anything missing?
- **Maintainability.** Will the next person understand this? Are the names clear? Is it as simple as it can be, or is there speculative abstraction nobody asked for? Is it tested?

A crucial habit: **ask questions rather than issue commands.** "What happens here if the list is empty?" invites the author to think and often reveals something you missed, whereas "handle the empty list" assumes you are right and shuts down the conversation. The reviewer is not the author's boss; you are a second pair of eyes, and phrasing that reflects genuine curiosity produces better code and better relationships. Compare these two comments on the same line:

```text
Bad:  This is wrong, wrap it in a try/except.
Good: If `models_dir` doesn't exist yet, will `iterdir()` raise here?
      Should we create it first, or is that guaranteed upstream?
```

The second version teaches, invites a real answer, and is right even in the cases where the reviewer's hunch turns out to be wrong.

Most review tools give you three verdicts. **Comment** leaves feedback without a decision. **Request changes** says "I think something here must change before this merges." **Approve** says "I am satisfied; this can land." Reserve *request changes* for genuine problems, not stylistic preferences, and be generous with *approve* once the substance is sound.

## A review checklist

When you review (or when you self-review before opening a PR — always do this first), run down a mental checklist:

- **Tests** — is the new behaviour covered? Do existing tests still pass?
- **Types** — are functions annotated, and would the type checker be happy?
- **Error handling** — specific exceptions, no bare `except`, failures raised not swallowed?
- **Logging** — meaningful log lines at sensible levels, and no stray `print()`?
- **Configuration** — no hard-coded URLs, keys, or paths; config read from the environment?
- **Dependencies** — is every new dependency actually justified, or could the standard library do it?

Self-reviewing against this list before you ask anyone else catches most issues while they are still cheap to fix.

## Branch strategy in practice

Teams keep a stable main branch that is always deployable, and do their work on short-lived **feature branches** — one branch per PR, merged and deleted when done. The shorter a branch lives, the less it drifts from main and the less painful the merge. (Creating and merging those branches is the git course's territory; here we care about the *strategy*.)

When a PR merges, many teams **squash-merge**: all the messy work-in-progress commits on the branch are collapsed into a single clean commit on main. This keeps the main history readable — one commit per landed change. An alternative philosophy is **trunk-based development**, where everyone integrates tiny changes into main many times a day behind feature flags, keeping branches almost non-existent; it is common at larger scale and worth knowing the name of.

## Drafts, templates, and the social contract

Two conventions smooth collaboration. **Draft pull requests** let you open a PR that is explicitly not ready to merge, to get early feedback on direction before you have polished everything — far cheaper than discovering after a day of work that you took the wrong approach. And a **PR template** — a file at `.github/pull_request_template.md` — pre-fills the description box with prompts ("What does this do? Why? How was it tested?"), so every PR arrives with the context reviewers need.

Underlying all of this is a social contract: **code review is collaborative, not adversarial.** The author and reviewer are on the same side, both trying to make the change as good as it can be. Critique the code, never the person. When a suggestion is genuinely optional — a preference rather than a problem — prefix it with **"Nit:"** (short for nitpick), which signals "take it or leave it, this won't block approval." Small signals like that keep review a place people want to participate in rather than dread.

## Key takeaways

- Keep PRs small and focused — one concern each — so review can be real rather than a rubber stamp. Git mechanics live in the Command Line & Git course.
- A good PR has a descriptive title and a body that explains *why*, not just what, with links to the issue it addresses.
- Review by reading the whole diff first, checking both correctness and maintainability, and asking questions instead of issuing commands.
- Run a checklist: tests, types, error handling, logging, config-not-hardcoded, justified dependencies — and self-review before asking others.
- Use short-lived feature branches with squash-merge (or trunk-based development at scale) to keep main clean and deployable.
- Draft PRs get early feedback; a `.github/pull_request_template.md` prompts for context; and "Nit:" marks optional suggestions in a collaborative, not adversarial, culture.

## Try it

Take a change you have made recently — even a small one — and write the pull-request description you *would* open for it, without opening anything. Give it a precise title, then a body that explains the why: what problem it solves, the approach you took, one alternative you considered and rejected, and how you tested it. Then put on the reviewer's hat and read your own diff against the checklist above (tests, types, error handling, logging, config, dependencies), writing at least two review comments phrased as genuine questions and marking one of them "Nit:". Notice how much clearer the change becomes once you have had to justify it to an imagined reader.
