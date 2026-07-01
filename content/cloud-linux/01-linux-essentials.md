# 01 — Linux Essentials

Most servers you'll ever touch in the cloud run Linux. It powers web apps, databases, and the machines that train and serve machine learning models. The good news: you don't need to be a systems expert to be productive. You need a mental model of how Linux is organized and a handful of commands you'll use every day. This lesson gives you both.

## The filesystem is one big tree

On Linux, everything lives under a single starting point called the **root directory**, written as `/`. There are no drive letters like `C:` or `D:`. Instead, every disk, folder, and file hangs off that one tree.

A few directories you'll see constantly:

- `/home` — where personal files live. Your account gets a folder like `/home/maya`.
- `/etc` — system configuration files.
- `/var` — data that changes over time, like logs (`/var/log`).
- `/tmp` — temporary scratch space, often wiped on reboot.
- `/usr` — installed programs and their supporting files.

Your **home directory** is your personal space, and the shell uses `~` as a shortcut for it. So `~/notes.txt` and `/home/maya/notes.txt` mean the same thing.

Paths come in two flavors. An **absolute path** starts at root (`/var/log/app.log`) and always points to the same place. A **relative path** is interpreted from wherever you currently are (`log/app.log` means "the log folder next to me").

## The shell: talking to the machine

The **shell** is a program that reads text commands and runs them. The most common one is `bash`. When you connect to a cloud server, you usually land in a shell with a **prompt** waiting for input, often ending in `$`.

Commands follow a simple pattern: the command name, then options (flags), then arguments.

```bash
ls -l /home/maya
```

Here `ls` is the command, `-l` is a flag meaning "long format," and `/home/maya` is the argument telling it what to list.

## Moving around and looking at files

These are the commands you'll reach for constantly:

```bash
pwd              # print working directory: where am I right now?
ls               # list files in the current directory
ls -la           # list all files (including hidden) in long format
cd /var/log      # change directory to /var/log
cd ..            # go up one level to the parent directory
cd ~             # go home
```

Files that start with a dot, like `.bashrc`, are **hidden** by default. That's a naming convention, not a security feature. `ls -a` reveals them.

To look at what's inside a file:

```bash
cat notes.txt        # dump the whole file to the screen
less notes.txt       # scroll through a file (press q to quit)
head -n 20 app.log   # first 20 lines
tail -n 20 app.log   # last 20 lines
tail -f app.log      # follow a log live as new lines arrive
```

`tail -f` is a favorite for watching a running service write logs in real time.

## Creating, moving, and deleting

```bash
mkdir projects           # make a new directory
touch todo.txt           # create an empty file (or update its timestamp)
cp todo.txt backup.txt   # copy a file
mv todo.txt tasks.txt    # rename or move a file
rm backup.txt            # delete a file
rm -r old_project        # delete a directory and everything inside it
```

A word of caution: `rm` does not move things to a trash can. There is no undo. Double-check before you delete, and be especially careful with `rm -r`.

## Finding things

Two tools cover most searches. `find` locates files by name or attributes; `grep` searches inside files for text.

```bash
find . -name "*.log"          # find all .log files under the current directory
grep "error" app.log          # show lines in app.log containing "error"
grep -ri "timeout" /etc       # search recursively, case-insensitive
```

You can also chain commands with a **pipe** (`|`), which feeds the output of one command into the next:

```bash
cat app.log | grep "error" | tail -n 5   # last 5 error lines
```

## Installing software with a package manager

You rarely download and install programs by hand on Linux. Instead, a **package manager** fetches software and its dependencies from trusted repositories. Which one you use depends on the Linux distribution:

```bash
# Debian and Ubuntu family
sudo apt update              # refresh the list of available packages
sudo apt install htop        # install a package named htop

# Red Hat, Fedora, Amazon Linux family
sudo dnf install htop
```

`apt update` refreshes the catalog; `apt install` does the actual installing. The `sudo` prefix means "run this with administrator rights," which we'll cover in the next lesson. Installing software affects the whole system, so it needs elevated permission.

## Getting help

Every command has documentation built in. When you forget how something works:

```bash
man ls           # full manual page for ls (press q to quit)
ls --help        # a shorter usage summary
```

These are always available, even on a server with no internet browser.

## Key takeaways

- Linux organizes everything under one root (`/`); your personal space is `~`.
- The shell runs commands in the form: `command -flags arguments`.
- `pwd`, `ls`, `cd`, `cat`, `less`, `head`, and `tail` handle navigation and viewing.
- `mkdir`, `touch`, `cp`, `mv`, and `rm` handle files — and `rm` has no undo.
- `find` locates files; `grep` searches inside them; the pipe `|` chains commands together.
- Package managers like `apt` and `dnf` install software system-wide, which is why they need `sudo`.
- `man` and `--help` are your built-in reference.

## Try it

Open a Linux shell (a cloud VM, a container, or a virtual machine on your own computer all work). Then:

1. Run `pwd` to confirm where you are, then `cd ~` to go home.
2. Create a workspace: `mkdir practice && cd practice`.
3. Make a file with some text: `echo "hello linux" > greeting.txt`, then read it back with `cat greeting.txt`.
4. Copy it (`cp greeting.txt copy.txt`), list the directory with `ls -la`, then delete the copy with `rm copy.txt`.
5. Search for your text across the folder: `grep -r "hello" .`

If every step behaved the way you expected, you have the core Linux workflow down.
