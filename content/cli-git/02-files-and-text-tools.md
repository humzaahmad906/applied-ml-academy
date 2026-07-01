# 02 — Files and Text Tools

Now that you can move around the filesystem, let's do something useful: read files, search inside them, and connect small tools into powerful pipelines. This is where the terminal starts to feel like a superpower. The core idea is simple — each tool does one thing well, and you wire them together.

## Reading files: `cat` and `less`

To dump the entire contents of a file to the screen:

```bash
cat notes.txt
```

```
Buy milk
Call the dentist
Finish the report
```

`cat` ("concatenate") is perfect for short files. But for a 5,000-line log file, it floods your screen. Use `less` instead:

```bash
less server.log
```

`less` opens a scrollable view. Use the arrow keys or `Space` to page down, `b` to page up, `/word` to search forward, and press `q` to quit back to the prompt. When a file is big or you just want to peek, `less` is the friendlier choice.

For a quick glance at just the start or end of a file:

```bash
head -n 5 server.log     # first 5 lines
tail -n 5 server.log     # last 5 lines
```

`tail` is especially handy for logs; `tail -f server.log` even follows a file live as new lines are written.

## Searching inside files: `grep`

`grep` finds lines matching a pattern. It's one of the most-used tools in existence:

```bash
grep "error" server.log
```

```
[10:32] error: connection refused
[10:41] error: timeout after 30s
```

Only the matching lines print. Some flags you'll reach for constantly:

```bash
grep -i "error" server.log    # -i: case-insensitive (matches Error, ERROR)
grep -n "error" server.log    # -n: show line numbers
grep -c "error" server.log    # -c: just count the matches
grep -r "error" logs/         # -r: search recursively through a folder
```

```bash
grep -rn "TODO" src/
```

```
src/app.py:42:    # TODO: handle empty input
src/utils.py:8:   # TODO: add caching
```

That last one — recursively find every `TODO` in your code with line numbers — is a genuine daily-driver command.

## Redirection: sending output to files

By default, command output goes to your screen. **Redirection** sends it somewhere else instead. The `>` operator writes output into a file:

```bash
ls > filelist.txt
```

Nothing prints to the screen; the listing went into `filelist.txt`. Careful: `>` **overwrites** the file completely. To **append** to the end instead, use `>>`:

```bash
echo "new entry" >> log.txt
```

You can also read *from* a file into a command with `<`, though you'll use `>` and `>>` far more often.

## Pipes: connecting tools together

Here's the big idea. The **pipe** `|` takes the output of one command and feeds it as the input of the next. Instead of one giant tool, you chain small ones:

```bash
cat server.log | grep "error"
```

The output of `cat` (the whole file) flows into `grep`, which keeps only the error lines. You can keep chaining:

```bash
cat server.log | grep "error" | head -n 3
```

That reads the file, filters to errors, then shows only the first three. Each `|` is a hand-off. A few classic combinations:

```bash
grep "error" server.log | wc -l
```

`wc -l` counts lines, so this counts how many error lines exist. (`grep -c` does the same thing, but the pipe version shows the pattern.)

```bash
ls -l | sort -k5 -n
```

Sort a long listing by the 5th column (file size), numerically. `sort` orders lines; `uniq` collapses adjacent duplicates. Together they're a favorite for tallying:

```bash
cat access.log | sort | uniq -c | sort -rn
```

That sorts lines, counts each unique one, then sorts by count descending — a quick "what shows up most?" report from any list.

## Wildcards: matching many files at once

You rarely want to type ten filenames. **Wildcards** (also called globbing) let the shell expand a pattern into all the matching names:

```bash
ls *.txt
```

```
notes.txt    log.txt    filelist.txt
```

The `*` matches any run of characters. So `*.txt` means "every name ending in `.txt`." More patterns:

```bash
ls report-*.csv      # report-jan.csv, report-feb.csv, ...
ls img?.png          # ? matches exactly one character: img1.png, img2.png
ls data/*/*.json     # every .json one level deep inside data/
```

Wildcards work with almost any command:

```bash
grep "failed" *.log
```

```
app.log:3:  request failed
sync.log:12: upload failed
```

The shell expands `*.log` into every log file before `grep` even runs, so you search them all at once — and `grep` helpfully prefixes each match with its filename.

## Putting it together

Say you want the three most common error messages in your logs. One line:

```bash
grep -h "error" *.log | sort | uniq -c | sort -rn | head -n 3
```

```
  47 error: timeout after 30s
  19 error: connection refused
   4 error: disk full
```

Read it left to right: pull error lines from every log (`-h` hides filenames so identical messages group), sort them, count duplicates, sort by count, keep the top three. Five small tools, one useful answer. That composability is the whole philosophy of the command line.

## Key takeaways

- `cat` dumps a file; `less` scrolls big files (`q` to quit); `head`/`tail` peek at the ends.
- `grep` finds matching lines — learn `-i`, `-n`, `-c`, and `-r`.
- `>` writes output to a file (overwriting), `>>` appends.
- The **pipe** `|` feeds one command's output into the next; chain small tools into big results.
- **Wildcards** (`*`, `?`) expand a pattern into many filenames before the command runs.

## Try it

Create a file with a few lines, then practice the pipeline pattern:

1. Run `echo "apple"`, then use `>>` three times to build a file `fruits.txt` containing `apple`, `banana`, `apple`.
2. `cat fruits.txt` to confirm the contents.
3. Count how many lines mention `apple` using `grep` and `wc -l` connected by a pipe.
4. Build a frequency report: pipe `fruits.txt` through `sort`, then `uniq -c`, then `sort -rn`. Which fruit appears most?
5. Use a wildcard to `grep` for `apple` across every `.txt` file in the directory at once.
