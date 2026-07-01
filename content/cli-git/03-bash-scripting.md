# 03 — Bash Scripting Basics

Typing commands one at a time is great, but the moment you find yourself repeating the same sequence, you should save it. A **script** is just a file full of shell commands that run top to bottom. Scripting turns "I do this by hand every morning" into "I run one command." Let's build one from scratch.

## Your first script

A script is a plain text file. Create one called `hello.sh`:

```bash
#!/bin/bash
echo "Hello from a script!"
```

The first line — `#!/bin/bash` — is called the **shebang**. It tells the system which program should run this file (here, bash). Always put it at the top.

To run it, you first make the file **executable**, then call it:

```bash
chmod +x hello.sh
./hello.sh
```

```
Hello from a script!
```

`chmod +x` grants "execute" permission. The `./` in front means "the file right here in this directory" — the shell won't run a local script without it. You only need `chmod` once per file.

## Variables

A **variable** stores a value you can reuse. Assign with `=` (no spaces around it!), and read the value back with a `$`:

```bash
#!/bin/bash
name="Ada"
echo "Hello, $name"
```

```
Hello, Ada
```

The no-spaces rule matters: `name = "Ada"` breaks, because bash would think `name` is a command. Keep it tight: `name="Ada"`.

Wrap variables in `${...}` when they sit next to other text so bash knows where the name ends:

```bash
file="report"
echo "${file}_final.txt"
```

```
report_final.txt
```

You can also capture the output of a command into a variable using `$(...)`:

```bash
today=$(date +%Y-%m-%d)
echo "Backup for $today"
```

```
Backup for 2026-07-01
```

## Quoting: single vs. double

This bites everyone once. **Double quotes** let variables expand; **single quotes** treat everything literally:

```bash
name="Ada"
echo "Hi $name"    # Hi Ada
echo 'Hi $name'    # Hi $name
```

Rule of thumb: use double quotes around variables almost always — especially paths, because it protects against filenames with spaces.

## Arguments: passing values in

A script can accept inputs when you run it. Inside the script, `$1` is the first argument, `$2` the second, and so on. `$#` is the count of arguments, and `$0` is the script's own name.

```bash
#!/bin/bash
echo "Greeting $1, argument count: $#"
echo "Hello, $1! You are $2 years old."
```

```bash
./greet.sh Ada 36
```

```
Greeting Ada, argument count: 2
Hello, Ada! You are 36 years old.
```

Arguments make scripts reusable — the same script greets anyone.

## Conditionals: doing things only when needed

An `if` statement runs a block only when a test passes:

```bash
#!/bin/bash
if [ "$1" = "" ]; then
  echo "Please provide a name."
else
  echo "Hello, $1"
fi
```

```bash
./greet.sh
```

```
Please provide a name.
```

The spacing inside `[ ... ]` is required — it's actually a command, so it needs spaces around the brackets and operators. Useful tests:

```bash
[ -f "$file" ]     # true if a file exists
[ -d "$dir" ]      # true if a directory exists
[ "$a" = "$b" ]    # strings are equal
[ "$n" -gt 5 ]     # number greater than: -gt -lt -ge -le -eq
```

## Loops: repeating over a list

A `for` loop runs a block once per item. This is where scripting pays off:

```bash
#!/bin/bash
for name in Ada Alan Grace; do
  echo "Hello, $name"
done
```

```
Hello, Ada
Hello, Alan
Hello, Grace
```

Loops shine over files. Combine them with wildcards to process a whole folder:

```bash
#!/bin/bash
for f in *.txt; do
  echo "Processing $f ..."
  wc -l "$f"
done
```

```
Processing notes.txt ...
      3 notes.txt
Processing log.txt ...
     18 log.txt
```

Notice `"$f"` is quoted — that protects filenames containing spaces.

## A small, real script

Let's tie it all together into something you might actually use: a script that backs up every `.txt` file into a dated folder. Save it as `backup.sh`:

```bash
#!/bin/bash
# Back up all .txt files into a timestamped folder.

target="backup-$(date +%Y-%m-%d)"

if [ -d "$target" ]; then
  echo "Backup folder already exists: $target"
else
  mkdir "$target"
  echo "Created $target"
fi

count=0
for f in *.txt; do
  cp "$f" "$target/"
  echo "  copied $f"
  count=$((count + 1))
done

echo "Done. Backed up $count files into $target."
```

Run it:

```bash
chmod +x backup.sh
./backup.sh
```

```
Created backup-2026-07-01
  copied log.txt
  copied notes.txt
Done. Backed up 2 files into backup-2026-07-01.
```

Look at what's here: a comment (lines starting with `#` are ignored), a variable built from a command, a conditional that avoids re-creating the folder, a loop over files, and simple arithmetic with `$((...))`. Every piece is something you just learned. That's a complete, useful tool in fifteen lines.

## Key takeaways

- A script starts with a **shebang** (`#!/bin/bash`); make it runnable with `chmod +x` and run it with `./name.sh`.
- Assign variables with `name=value` (**no spaces**); read them with `$name` or `${name}`; capture command output with `$(...)`.
- Use **double quotes** around variables to be safe; single quotes are literal.
- `$1`, `$2`, ... are **arguments**; `$#` is how many there are.
- `if [ ... ]; then ... fi` runs code conditionally (mind the spaces); `for x in list; do ... done` repeats over a list.

## Try it

Write a script called `count.sh` that:

1. Starts with the correct shebang.
2. Takes a directory name as its first argument (`$1`).
3. Prints an error and stops if no argument was given.
4. Prints an error if the argument isn't an existing directory (`[ -d ... ]`).
5. Loops over every file in that directory and prints the filename plus its line count.
6. At the end, prints how many files it processed.

Make it executable and run it against a folder with a few files. Then run it with no argument to confirm your error message fires.
