# 02 — Users, Permissions, and Processes

Linux was built from the start to be used by many people on one machine at the same time. That single design decision explains most of how permissions and processes work. Even when you're the only human on a cloud server, the system still thinks in terms of users, ownership, and running programs. Understanding these three ideas will save you from a lot of confusing "permission denied" errors and runaway processes.

## Users and groups

Every account on a Linux system is a **user**, identified by a username and a numeric ID. There's one special user called **root** (user ID 0) who can do anything: read any file, kill any process, change any setting. Regular users are deliberately limited so a mistake or a compromised program can't wreck the whole machine.

Users can belong to **groups**, which let several accounts share access to the same files. You might have a `developers` group whose members can all read a shared codebase.

```bash
whoami          # which user am I logged in as?
id              # show my user ID, groups, and more
groups          # list the groups I belong to
```

## File permissions

Every file and directory has an **owner**, a **group**, and a set of permissions describing who may do what. Run `ls -l` and you'll see something like:

```bash
-rw-r--r--  1 maya developers  1240 Jul  1 09:15 report.txt
```

The first block, `-rw-r--r--`, is the permission string. Read it in four parts:

- The first character is the type: `-` for a regular file, `d` for a directory.
- The next three (`rw-`) are the **owner's** permissions.
- The next three (`r--`) are the **group's** permissions.
- The last three (`r--`) are for **everyone else**.

Each trio uses three letters: `r` (read), `w` (write), `x` (execute). A dash means that permission is absent. So `rw-r--r--` means the owner can read and write, while the group and everyone else can only read.

For directories the meanings shift slightly: `r` lists the contents, `w` creates or deletes files inside, and `x` lets you enter the directory.

## Changing ownership and permissions

Two commands do the work. `chmod` changes permissions; `chown` changes ownership.

```bash
chmod +x deploy.sh          # make a script executable
chmod 644 report.txt        # owner read/write, everyone else read-only
sudo chown maya report.txt  # change the owner to maya
```

The number `644` is shorthand: each digit encodes one permission trio, where read=4, write=2, execute=1, added together. So `6` is read+write, `4` is read-only, and `755` (a common one for programs) is read/write/execute for the owner and read/execute for everyone else.

## Becoming the administrator with sudo

You generally don't log in as root. Instead, when a single command needs administrator rights, you prefix it with **`sudo`** ("superuser do"):

```bash
sudo apt install nginx      # install software system-wide
sudo systemctl restart nginx
```

`sudo` asks for your password, then runs that one command with elevated privileges. This is far safer than staying logged in as root, because you only borrow power for the exact action that needs it. The rule of thumb: reach for `sudo` only when a command touches system-wide files or services, and read the command carefully before you run it as root.

## Processes: programs that are running

A **process** is a running instance of a program. Each has a **PID** (process ID), an owner, and a share of CPU and memory. Tools let you see what's running and how much it's consuming.

```bash
ps aux              # snapshot of all running processes
top                 # live, updating view of CPU and memory use
htop                # a friendlier, colorized version (if installed)
```

In `top`, you'll see processes sorted by CPU usage. This is how you spot the training job that's eating every core or the leaky service slowly consuming all your memory. Press `q` to quit.

To find a specific process:

```bash
ps aux | grep python     # find running python processes
```

## Stopping processes

Sometimes a program hangs or misbehaves and you need to stop it. You do that by sending it a **signal** with `kill`, using its PID.

```bash
kill 4821           # politely ask process 4821 to shut down (SIGTERM)
kill -9 4821        # force it to stop immediately (SIGKILL)
```

Always try a plain `kill` first. It sends a signal that lets the program clean up: finish writing a file, close a connection, save state. Only escalate to `kill -9` if the process ignores the polite request, because a forced kill gives it no chance to tidy up and can leave things in a messy state.

If a program is running in your current terminal and you want to stop it, pressing `Ctrl+C` sends the same polite termination signal.

## Key takeaways

- Every file has an owner, a group, and permissions for owner / group / everyone.
- Read a permission string in trios: `r` read, `w` write, `x` execute.
- `chmod` changes permissions (numbers like `644` and `755` are shorthand); `chown` changes ownership.
- Don't live as root. Use `sudo` to borrow admin rights for one command at a time.
- A process is a running program with a PID; `ps`, `top`, and `htop` show what's active and what it's consuming.
- Stop processes with `kill` (polite) before `kill -9` (forced); `Ctrl+C` stops the program in your terminal.

## Try it

On a Linux shell:

1. Run `whoami` and `id` to see who you are and which groups you belong to.
2. Create a script: `echo 'echo hello' > run.sh`. Try `./run.sh` — it will likely fail because the file isn't executable.
3. Fix it with `chmod +x run.sh`, then run `./run.sh` again. Check the new permissions with `ls -l run.sh`.
4. Open `top`, watch it for a few seconds, and identify the process using the most CPU. Press `q` to exit.
5. In one terminal, start a long-running command like `sleep 300`. In another, find its PID with `ps aux | grep sleep` and stop it with `kill <PID>`.

If you can create a file, adjust its permissions, and start and stop a process on demand, you've got the essentials.
