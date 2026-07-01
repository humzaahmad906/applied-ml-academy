# 01 — The Terminal and Shell

Every engineer, no matter how fancy their editor, eventually types commands into a black window with a blinking cursor. That window is the **terminal**, and the program listening to what you type is the **shell**. Learning to move around in it is the single highest-leverage skill you can pick up early. Let's demystify it.

## What is a shell, really?

A **terminal** is just the window (the app). A **shell** is the program running inside it that reads your commands, runs them, and prints the results. The most common shell is **bash**; many modern Macs default to **zsh**. For everything in this course, they behave the same.

When you type a command and press Enter, the shell finds the matching program, runs it, and shows you the output. That's the whole loop:

```bash
echo "hello, world"
```

```
hello, world
```

`echo` simply prints back whatever you give it. Not glamorous, but it proves the loop works: you typed, the shell ran, you got output.

## Opening the terminal

- **macOS**: Press `Cmd + Space`, type "Terminal", press Enter.
- **Linux**: Look for "Terminal" in your applications, or press `Ctrl + Alt + T`.
- **Windows**: Use "Windows Terminal" or install WSL (Windows Subsystem for Linux) for a real Linux shell.

When it opens, you'll see a **prompt** — something like:

```
humza@laptop ~ %
```

This tells you your username, your machine, and where you currently are (`~` means your home folder). The `%` or `$` at the end is where your typing goes. Prompts vary; don't worry about the exact look.

## Where am I? `pwd`

The filesystem is a tree of folders (called **directories**). At any moment, your shell sits *inside* one directory — your **current working directory**. To ask where you are:

```bash
pwd
```

```
/Users/humza
```

`pwd` stands for "print working directory." That slash-separated string is a **path**: a route from the top of the filesystem (`/`, the "root") down to where you stand.

## What's here? `ls`

To list what's in the current directory:

```bash
ls
```

```
Desktop    Documents    Downloads    Pictures    projects
```

Commands take **flags** (options) to change their behavior. A couple of useful ones for `ls`:

```bash
ls -l
```

```
drwxr-xr-x   5 humza  staff   160 Jun 30 09:12 Documents
-rw-r--r--   1 humza  staff  2048 Jun 29 14:03 notes.txt
```

The `-l` flag gives a "long" listing with permissions, sizes, and dates. A line starting with `d` is a directory; one starting with `-` is a file.

```bash
ls -a
```

The `-a` flag reveals **hidden files** — names beginning with a dot, like `.gitignore` or `.bashrc`. They're hidden by default because they're usually configuration you rarely touch. You can combine flags: `ls -la`.

## Moving around: `cd`

`cd` means "change directory." Give it a path and your shell walks there:

```bash
cd Documents
pwd
```

```
/Users/humza/Documents
```

A few special shortcuts make navigation fast:

```bash
cd ..
```

`..` means "the parent directory" — go up one level. To go up two levels, `cd ../..`.

```bash
cd ~
```

`~` always means your home directory, from anywhere.

```bash
cd -
```

`-` jumps back to the directory you were just in — handy for bouncing between two places.

Running `cd` with no argument at all also takes you home.

## Absolute vs. relative paths

This trips up every beginner, so let's be explicit.

- An **absolute path** starts with `/` and describes the full route from root: `/Users/humza/projects`. It works no matter where you currently are.
- A **relative path** is measured from where you stand right now: `projects/website` means "the `website` folder inside the `projects` folder inside my current directory."

```bash
cd /Users/humza/projects    # absolute — always lands the same place
cd projects                 # relative — depends on where you are
```

If a `cd` fails with "No such file or directory," you're almost always confusing the two. Run `pwd` and look before you leap.

## Tab completion: your best friend

You do not type full names. Type the first few letters and press **Tab**; the shell finishes the name for you:

```bash
cd Doc<Tab>
```

...becomes `cd Documents/`. If several names match, press Tab twice to see the options. This saves typos and time — use it constantly.

## Making and removing directories

```bash
mkdir practice
cd practice
```

`mkdir` ("make directory") creates a new folder. To remove an *empty* directory:

```bash
cd ..
rmdir practice
```

Be careful with removal commands — there's no recycle bin at the terminal. We'll treat deleting files with more caution as we go.

## A note on getting unstuck

If a command seems frozen, `Ctrl + C` cancels it and returns your prompt. If your screen fills with clutter, `clear` wipes it clean. And almost every command explains itself:

```bash
ls --help
```

## Key takeaways

- The **terminal** is the window; the **shell** (bash or zsh) is the program running commands inside it.
- `pwd` prints where you are, `ls` lists what's there, `cd` moves you around.
- **Flags** like `-l` and `-a` modify what a command shows.
- **Absolute paths** start from `/`; **relative paths** start from where you are. Run `pwd` when confused.
- Use **Tab** to autocomplete names and `Ctrl + C` to cancel a stuck command.

## Try it

Open your terminal and, without using your mouse:

1. Run `pwd` and note where you start.
2. `cd` into your home directory with the shortcut, then list everything including hidden files.
3. Create a directory called `sandbox`, `cd` into it, and confirm with `pwd`.
4. Go back up one level, then use the shortcut to jump straight back into `sandbox`.
5. Return home and remove the empty `sandbox` directory.

If every step printed what you expected, you can now navigate a filesystem entirely by keyboard. That's the foundation for everything else.
