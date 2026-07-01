# 06 — IAM and Security

Once you have machines, storage, and networks in the cloud, one question towers over the rest: **who is allowed to do what?** Getting this wrong is how accounts get taken over, data gets leaked, and cloud bills explode from a leaked key. The system that answers that question is called **IAM** — Identity and Access Management. This lesson covers the core ideas and the habits that keep you safe, in plain language.

## The two halves: authentication and authorization

Security splits into two questions that are easy to confuse:

- **Authentication** — *who are you?* Proving your identity, usually with a password plus a second factor, or with a cryptographic key.
- **Authorization** — *what are you allowed to do?* Deciding, once your identity is known, which actions you may perform on which resources.

IAM handles both, but most of the interesting design work is in authorization.

## Identities: users, groups, and roles

An **identity** is anything that can be granted access. There are a few kinds:

- A **user** represents a person with long-lived credentials (a login, and often access keys for programmatic use).
- A **group** is a bundle of users that share the same permissions — for example, all your data scientists. You grant permissions to the group once instead of to each person.
- A **role** is an identity that isn't tied to one person and is meant to be **assumed temporarily**. A VM can assume a role to get short-lived credentials, or a person can switch into a role to do a specific job. Roles are the preferred way to hand out access because the credentials are temporary and automatically expire.

## Policies: the rules of access

Permissions are written as **policies** — documents that spell out allowed (or denied) actions on specific resources. A policy attaches to an identity and answers, for every request, "is this allowed?" A single readable policy might say, in effect: *this role may read objects from the `training-data` bucket, and nothing else.*

The power of policies is precision. You can allow reading but not deleting, restrict access to one bucket rather than all of them, and even limit actions to certain conditions. That precision is what makes the next principle possible.

## Least privilege: the golden rule

The single most important idea in cloud security is **least privilege**: give every identity the *minimum* permissions it needs to do its job, and nothing more.

It's tempting to hand out broad "admin" access to make things "just work." Resist it. If a service only needs to read from one bucket, don't give it write access to everything. The reason is blast radius: if that identity is ever compromised — a leaked key, a bug, a phished password — the damage is limited to what it could do. A narrow permission set turns a catastrophe into an inconvenience.

A practical way to apply this: start with no permissions, then add exactly what breaks. It's slower up front but far safer than starting with everything and trying to take things away later.

## Secrets: passwords, keys, and tokens

A **secret** is any credential that grants access: a database password, an API key, an access token, a private key. Mishandling secrets is one of the most common ways systems get breached. A few firm rules:

- **Never hard-code secrets in source code.** They end up in version control, shared, and forgotten. Committing a key to a public repository can lead to a compromised account within minutes.
- **Never commit secrets to git.** Use a `.gitignore` and keep credentials out of the repository entirely.
- **Use a secrets manager.** Cloud providers offer dedicated services (and there are standalone tools) that store secrets encrypted and hand them to your application at runtime. Your code asks for the secret when it needs it rather than carrying it around.
- **Prefer roles over long-lived keys.** When a VM or service can assume a role for temporary credentials, you avoid storing a permanent key at all. This is the single best habit for machine-to-machine access.
- **Rotate credentials.** Change passwords and keys periodically, and immediately if one might have leaked.

In your local environment, secrets typically come from **environment variables** rather than being written in files:

```bash
export DB_PASSWORD="..."     # set a secret in the environment, not in code
# the application reads it via the environment at runtime
```

## Extra layers worth having

Beyond IAM policies, a few defenses pay for themselves many times over:

- **Multi-factor authentication (MFA).** Require a second factor (a phone app or hardware key) on top of passwords, especially for the root/owner account. A stolen password alone then isn't enough to get in.
- **Encryption.** Encrypt data **at rest** (stored on disk or in a bucket) and **in transit** (moving over the network, which is what HTTPS does). Most cloud storage can encrypt at rest with a single setting.
- **Audit logs.** Providers can record who did what and when. Turn this on early; when something goes wrong, the log is how you find out what happened.
- **Guard the root account.** The all-powerful owner account should be used almost never — set up MFA on it, then create limited users for day-to-day work.

## Key takeaways

- IAM answers two questions: **authentication** (who are you?) and **authorization** (what can you do?).
- Identities come as **users**, **groups**, and **roles**; roles give temporary, auto-expiring credentials and are the safest way to grant access.
- **Policies** define allowed actions on specific resources with fine precision.
- **Least privilege** — grant the minimum needed and nothing more — is the golden rule; it shrinks the damage of any breach.
- Never hard-code or commit **secrets**; use a secrets manager, prefer roles over long-lived keys, and rotate credentials.
- Add MFA, encrypt data at rest and in transit, enable audit logs, and lock down the root account.

## Try it

You don't need to configure a live account to practice the thinking:

1. Pick a small system — say, a web app that reads user photos from a bucket and writes records to a database. List every distinct identity involved (the app, the deploy process, you as an admin).
2. For each identity, write the *minimum* set of permissions it needs in one line. Notice how narrow they can be.
3. Find one secret in that system (a database password) and describe where it should live — and three places it should *never* live.
4. If you have a cloud account, check whether MFA is enabled on your root/owner login. If it isn't, enabling it is the highest-value five minutes you'll spend all week.

If you can describe a system as a set of narrow identities and well-guarded secrets, you're thinking about security the right way.
