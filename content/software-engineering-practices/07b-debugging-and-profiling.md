# 07b — Debugging and Profiling

Logging tells you what a program did after the fact. Debugging and profiling are the two things you do while you are still in the room with the problem: debugging answers "why is this wrong?" and profiling answers "why is this slow?" Both have a beginner's version — `print()` again, or a stopwatch and a guess — and both have a professional version that is faster, more precise, and requires you to stop guessing. This lesson teaches the professional version of each.

## Debugging past print

The instinct when code misbehaves is to scatter `print()` calls and re-run. It works, barely, but every run tells you only what you thought to print, and you have to edit and re-run to ask a new question. A debugger inverts that: it pauses the program mid-flight and lets you interrogate the live state — inspect any variable, evaluate any expression, step forward one line at a time — without editing and re-running for each question.

Python's built-in debugger is `pdb`, and since Python 3.7 you drop into it with the `breakpoint()` builtin:

```python
def normalise(values):
    total = sum(values)
    breakpoint()          # execution pauses here, drops you into pdb
    return [v / total for v in values]
```

When execution hits `breakpoint()` you get an interactive `(Pdb)` prompt sitting exactly at that line. The essential commands are a handful of single letters:

- **`n`** (next) — run the current line, stop at the next one in this function.
- **`s`** (step) — run the current line, but step *into* any function it calls.
- **`c`** (continue) — resume until the next breakpoint (or the end).
- **`p expr`** (print) — evaluate an expression and show it: `p total`, `p len(values)`.
- **`l`** (list) — show the source around the current line, so you can see where you are.
- **`w`** (where) — print the call stack: how you got to this line.

The distinction between `n` and `s` is the one beginners trip on. `n` treats a function call as a single step; `s` dives inside it. Use `n` to move through code you trust and `s` when the bug is *inside* a call. `p` is the workhorse — it is `print()` without the edit-and-re-run loop, and it can evaluate anything, including calls like `p sorted(values)`.

## The IDE debugger is the day-to-day tool

`pdb` is always available and worth knowing cold — it is the only debugger you have on a bare server. But for daily work, the IDE debugger (VS Code, PyCharm, and others) is more comfortable and does the same thing with a GUI. You set a **breakpoint** by clicking in the gutter to the left of a line number — a red dot appears — and press F5 to run under the debugger. When execution reaches the dot it pauses, and a panel shows every local variable and its value without you typing `p` for each one.

The stepping controls map onto the `pdb` letters: **Step Over** (F10) is `n`, **Step Into** (F11) is `s`, **Step Out** (Shift+F11) runs until the current function returns, and **Continue** (F5) is `c`. A **Watch** pane lets you pin expressions — say `total` or `len(values) == 0` — and see them update as you step. A **Debug Console** lets you evaluate arbitrary code in the paused program's context, which is `pdb`'s `p` with autocomplete.

Two features earn their keep once the basics are muscle memory. A **conditional breakpoint** pauses only when a condition is true — right-click the red dot and enter `user_id == 8842` — so instead of hitting a breakpoint on all ten thousand loop iterations you stop only on the one that misbehaves. And **post-mortem debugging** drops you into the debugger *at the point an unhandled exception was raised*, with the full stack intact, so you can inspect the state that caused the crash rather than reconstructing it:

```python
import pdb

try:
    run_pipeline()
except Exception:
    pdb.pm()          # post-mortem: inspect the frame where it blew up
```

Running a script with `python -m pdb script.py` gives you post-mortem for free — on any uncaught exception it drops to the prompt instead of exiting.

## Debugging is a method, not a tool

The debugger is only as good as how you use it. Effective debugging is a discipline, and it is the same regardless of language:

1. **Reproduce it reliably.** A bug you cannot trigger on demand you cannot fix — you can only guess. Find the exact input, config, and sequence that produces it every time. If it is intermittent, that unreliability is itself the first thing to investigate.
2. **Shrink to a minimal repro.** Strip away everything that still leaves the bug present. A 500-line failure that reduces to 8 lines is a bug you have almost understood already, because everything irrelevant is gone.
3. **Bisect.** The bug is somewhere between "known good" and "observably broken." Check the middle. Is the state correct halfway through? That halves the search space every time — the same idea as `git bisect` for finding the commit that introduced a regression.
4. **Form a hypothesis, then test it.** "I think `total` is zero here" is a hypothesis; `p total` confirms or kills it. Debugging by hypothesis is fast because each check eliminates possibilities. Poking around hoping to notice something is slow because it eliminates nothing.

The debugger accelerates steps 3 and 4. It does not replace them.

## "It's slow" — measure, don't guess

Performance work has one iron rule: **do not guess where the time goes.** Programmer intuition about hotspots is famously wrong — the slow part is almost never the part that looks slow. You measure first, and the tool for a whole run is `cProfile`, in the standard library. Consider a script with a deliberate hotspot:

```python
# slow.py
def slow_square_sum(n):
    result = 0
    for i in range(n):
        result += i ** 2        # the hotspot lives here
    return result

def fast_lookup(table, key):
    return table.get(key, 0)

def main():
    table = {i: i for i in range(1000)}
    total = 0
    for _ in range(200):
        total += slow_square_sum(50_000)
        fast_lookup(table, 42)
    print(total)

if __name__ == "__main__":
    main()
```

Profile the entire run from the command line, saving results to a file:

```bash
python -m cProfile -o run.prof slow.py
```

Then read the top offenders sorted by **cumulative time** — the total time spent in a function *including* everything it calls:

```python
import pstats
stats = pstats.Stats("run.prof")
stats.sort_stats("cumulative").print_stats(10)
```

Cumulative time answers "where is the wall-clock time going?" and points you at `slow_square_sum`, which dominates the run, while `fast_lookup` barely registers — exactly the kind of result that contradicts a guess. Reading tables gets old fast, so visualise instead. `snakeviz` (install with `uv add snakeviz`) opens the same `.prof` file as an interactive icicle chart in your browser:

```bash
snakeviz run.prof
```

Each box is a function, its width proportional to time spent; click any box to zoom in. The widest boxes are your targets, and the picture makes the hierarchy obvious in a way the table does not.

## Profiling a process you can't stop: py-spy

`cProfile` requires launching the program under the profiler and adds noticeable overhead — fine for a script, wrong for a training run that has been going for six hours or a service in production. `py-spy` (install with `uv add py-spy`) is a **sampling profiler**: it runs as a *separate process*, reads the target's stacks out of memory, and needs no code changes and no restart. You point it at a running process by PID:

```bash
py-spy top --pid 12345           # live top-like view of hot functions
py-spy record -o profile.svg --pid 12345   # record a flame graph
py-spy dump --pid 12345          # one-shot stack trace, "what is it doing right now?"
```

Its overhead is low enough that the project documents it as safe against production workloads. This makes it the right tool for a long-running or unmodifiable process — a stuck training job, a slow API server — where "restart it under cProfile" is not an option. (On Linux, attaching by PID usually needs `sudo`.)

When `cProfile` has named the guilty *function* but you need the guilty *line*, reach for `line_profiler` (install `uv add line_profiler`). You decorate the suspect function with `@profile` and it reports time spent per line, so you can see that `result += i ** 2` is where the loop bleeds.

## Memory, and one tool for both

Slowness is sometimes really a memory problem — allocations, leaks, an array copied when it should have been viewed. The standard library ships `tracemalloc`, which snapshots allocations and tells you which lines allocated the most:

```python
import tracemalloc

tracemalloc.start()
build_big_structure()
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics("lineno")[:5]:
    print(stat)          # top 5 lines by allocated memory
```

If you would rather learn one tool than several, `scalene` (install `uv add scalene`) is a modern profiler that measures **CPU and memory together at line granularity**, separates time spent in Python from time in native library code, and reports GPU time on NVIDIA systems — genuinely useful for ML work. You run it with `scalene slow.py` and get a combined report. It uses sampling, so its overhead stays modest even on expensive workloads.

## Profile before you optimize

The point of all this is a single professional habit: **profile before you optimize, and optimize the top of the flame graph.** The widest box, the highest cumulative-time function, the line `line_profiler` flags — that is the only code worth touching first, because a 2x speedup on code that is 1% of the runtime buys you nothing, while a 2x speedup on code that is 80% of the runtime nearly halves the whole run. Optimizing anything else is effort spent making the program no faster and usually harder to read. Measure, fix the top, then measure again — because once you fix the biggest thing, the second-biggest is now the top, and it may not be what you expected either.

## Key takeaways

- A debugger pauses a live program so you can inspect any variable and step line by line — it replaces the edit-and-re-run loop of `print()` debugging.
- Learn `pdb` via `breakpoint()` and the core commands `n`, `s`, `c`, `p`, `l`, `w`; `n` steps over calls, `s` steps into them.
- The IDE debugger (VS Code, PyCharm) is the day-to-day tool: gutter breakpoints, a variables pane, watches, and F10/F11/Shift+F11/F5 for step over/into/out/continue. Use conditional breakpoints and post-mortem (`pdb.pm()`, `python -m pdb`) to reach the exact failing state.
- Debugging is a method: reproduce reliably, shrink to a minimal repro, bisect the search space, and test one hypothesis at a time.
- Never guess where time goes. Use `cProfile` for a whole run, read the top cumulative-time functions, and visualise with `snakeviz`.
- `py-spy` samples a running process with no code changes or restart — the right tool for production and long training runs; `line_profiler` gives per-line detail on a hot function.
- For memory use `tracemalloc`, or `scalene` as a modern combined CPU+memory (and GPU) line-level profiler.
- Profile before optimizing, and optimize the top of the flame graph — speeding up anything else buys nothing.

## Try it

Save the `slow.py` example above and profile it: `python -m cProfile -o run.prof slow.py`, then `snakeviz run.prof`, and confirm with your own eyes that `slow_square_sum` dominates while `fast_lookup` is invisible — the measurement, not your intuition, names the hotspot. Now open the file in your IDE, set a breakpoint inside `slow_square_sum`, make it conditional on `i == 25000`, run under the debugger, and inspect `result` at that iteration in the variables pane. Finally, install `scalene` (`uv add scalene`) and run `scalene slow.py`; note how it attributes both time and memory to individual lines, and compare what it tells you against the cProfile result. When they agree, you have found the one line worth optimizing — and everything else, you now know to leave alone.
