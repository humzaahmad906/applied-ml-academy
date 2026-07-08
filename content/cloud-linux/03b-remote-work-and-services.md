# 03b — Remote Work, Services, and GPU Basics

Renting a VM is only the first step. Once you have a machine in the cloud, you need to work on it the way a professional does: log in securely, move data and models back and forth, keep a training run alive when your laptop goes to sleep, run jobs unattended, give your datasets somewhere to live, and check that the expensive GPU you're paying for is actually busy. This lesson walks through those week-one skills, one at a time.

## SSH the right way: keys, not passwords

The previous lesson connected with `ssh -i ~/.ssh/my-key.pem maya@203.0.113.42`. That works, but let's understand where that key comes from and how to set one up yourself.

An SSH **key pair** is two files: a **private key** that never leaves your laptop, and a **public key** you're free to hand out. The server keeps a copy of your public key; when you connect, SSH proves you hold the matching private key without ever sending it over the wire. This is safer than a password — there's nothing to guess or intercept.

Generate a modern key pair once:

```bash
ssh-keygen -t ed25519 -C "maya@laptop"
```

`ed25519` is the recommended key type today — short, fast, and strong. This creates `~/.ssh/id_ed25519` (private) and `~/.ssh/id_ed25519.pub` (public). Guard the private key; treat the `.pub` file as public.

To let a server accept your key, its public half must land in `~/.ssh/authorized_keys` on the server. The easy way, if you can already log in:

```bash
ssh-copy-id maya@203.0.113.42
```

That appends your public key to the right file with the right permissions. (Many cloud providers do this for you at launch — you paste a public key into a web form, and it's placed in `authorized_keys` on first boot.)

### SSH config aliases

Typing `ssh -i ~/.ssh/id_ed25519 maya@203.0.113.42` every time gets old. Put the details in `~/.ssh/config`:

```
Host trainbox
    HostName 203.0.113.42
    User maya
    IdentityFile ~/.ssh/id_ed25519
```

Now `ssh trainbox` does the whole thing. The same alias works for file copies too, which is the next piece.

### Moving data and models: scp and rsync

`scp` copies files over the same SSH connection. The syntax mirrors `cp`, with `host:path` for the remote side:

```bash
scp ./data.csv trainbox:~/datasets/          # laptop -> server
scp trainbox:~/runs/model.pt ./              # server -> laptop
```

For anything large or repeated, prefer `rsync`. It only transfers the parts that changed and can resume, which matters when a dataset is tens of gigabytes:

```bash
rsync -avz --progress ./dataset/ trainbox:~/datasets/
```

`-a` preserves file attributes, `-v` is verbose, `-z` compresses in transit, and `--progress` shows a live bar. Run the same command again after adding a few files and rsync skips everything already there.

## Keep work alive with tmux

Here's a trap every new cloud user hits. You SSH in, start a training run, and close your laptop to catch the train. When you reconnect, the run is gone — killed the instant your SSH connection dropped. Any process tied to your login dies with it.

The fix is a **terminal multiplexer**: `tmux` (or the older `screen`). It runs a shell on the server that lives independently of your SSH connection. You attach to it, do work, detach, and the shell — and everything running in it — keeps going.

```bash
tmux new -s train      # start a new session named "train"
# ...launch your training script here...
```

To leave it running, **detach**: press `Ctrl-b` then `d`. Your SSH connection is now free to drop. Later, from anywhere:

```bash
tmux attach -t train   # reattach to the "train" session
tmux ls                # list running sessions
```

Inside tmux, `Ctrl-b` is the "prefix" that precedes every command. A few worth knowing:

- `Ctrl-b c` — create a new **window** (like a browser tab).
- `Ctrl-b n` / `Ctrl-b p` — next / previous window.
- `Ctrl-b %` — split the current pane vertically; `Ctrl-b "` splits horizontally.
- `Ctrl-b` then an arrow key — move between **panes**.

A common setup: one pane running the training script, another running a GPU monitor beside it. `screen` does the same job with different keys (`Ctrl-a` as its prefix) if that's what a machine already has.

## Long-running work as a service

tmux is perfect for interactive runs you're babysitting. But some work should run on its own — an inference API that must come back after a crash, or a nightly job. For that, hand the work to the operating system.

### systemd: a service that restarts itself

`systemd` is Linux's manager for background services. You describe your program in a small **unit file** and systemd keeps it running. Create `/etc/systemd/system/inference.service`:

```
[Unit]
Description=Model inference API
After=network.target

[Service]
User=maya
WorkingDirectory=/home/maya/app
ExecStart=/home/maya/app/venv/bin/python serve.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

The lines that matter most: `ExecStart` is the command to run, and `Restart=on-failure` tells systemd to relaunch the program if it exits with an error or is killed — the broadest safety net for an unattended service. `RestartSec=5` waits five seconds between attempts so a crash-looping service doesn't hammer the machine. Then:

```bash
sudo systemctl daemon-reload        # re-read unit files
sudo systemctl enable --now inference   # start now + on every boot
sudo systemctl status inference     # is it running?
journalctl -u inference -f          # follow its logs live
```

One caution: a service that restarts too eagerly can hide a real problem — it looks healthy while silently failing over and over. `systemd` also has `StartLimitBurst` to give up after too many restarts in a window, which is worth adding once you care about alerting.

### cron: jobs on a schedule

For work that runs at fixed times — retrain every Sunday, sync logs each hour — use `cron`. Run `crontab -e` and add a line. The five fields are minute, hour, day-of-month, month, day-of-week:

```
0 3 * * 0  /home/maya/app/venv/bin/python /home/maya/app/retrain.py >> /home/maya/logs/retrain.log 2>&1
```

That runs at 03:00 every Sunday (day-of-week `0`). Redirecting output to a log file (`>>` for stdout, `2>&1` to fold in errors) is essential — cron jobs fail silently otherwise, and you'll want the log when something breaks.

## Disks: giving datasets a home

Cloud VMs often boot with a small root disk. Datasets and checkpoints go on a separate **data volume** you attach in the provider's console. Once attached, the OS sees a new raw device but can't use it yet. List block devices to find it:

```bash
lsblk
```

You'll see the boot disk (often `sda` or `nvme0n1`) and your new empty volume (say `nvme1n1`) with no mountpoint.

A brand-new volume needs a **filesystem** before it can hold files. This is the one dangerous step in this lesson. The command below, `mkfs`, **erases everything on the target device** — run it only on a new, empty volume, and triple-check the device name, because pointing it at your boot disk or a disk with data destroys that data permanently and unrecoverably. Confirm with `lsblk` that the device has no partitions and no mountpoint before you touch it. On a genuinely new volume, you format it once with something like `sudo mkfs -t ext4 /dev/nvme1n1`.

After formatting, **mount** it — attach it to a folder in the filesystem:

```bash
sudo mkdir -p /mnt/data
sudo mount /dev/nvme1n1 /mnt/data
```

Now `/mnt/data` reads and writes to your volume. But a plain `mount` doesn't survive a reboot. To make it permanent, add a line to `/etc/fstab`, which lists disks to mount at boot. Best practice is to reference the disk by its **UUID** (stable) rather than its device name (which can change). Get it with `sudo blkid /dev/nvme1n1`, then add a line to `/etc/fstab` mapping that UUID to `/mnt/data`. Test the fstab entry with `sudo mount -a` before rebooting — a bad fstab line can stop the machine from booting cleanly.

## GPU health: is the card actually working?

You're paying for a GPU, so confirm it's busy. The essential tool is `nvidia-smi`:

```bash
nvidia-smi
```

The output packs a lot in. Focus on three things:

- **GPU-Util** — the percentage of recent time the GPU was executing work. During training this should be high (often 90%+). If it's stuck near 0% while your script "runs," the GPU is idle and your bottleneck is elsewhere (usually the data loader feeding it too slowly). Note that 100% doesn't guarantee efficiency — it means the GPU was busy, not that it was busy with useful work.
- **Memory-Usage** — how much of the card's VRAM is in use versus total. Out-of-memory crashes are the most common training failure; watch this climb as batch size grows.
- **Processes** — at the bottom, the PIDs using the GPU and how much memory each holds. This is how you spot a zombie process from a previous run still hogging VRAM (`kill` its PID to free it).

A single `nvidia-smi` is one snapshot. To watch it live, refresh once a second:

```bash
watch -n1 nvidia-smi
```

Run that in one tmux pane while your training script runs in another, and you can see utilization and memory move in real time. For scripting or logging, `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` prints just the numbers you ask for, ready to pipe into a file.

## Key takeaways

- Log in with an `ed25519` SSH key pair, not a password; put your public key in the server's `authorized_keys` (via `ssh-copy-id`) and use `~/.ssh/config` aliases to shorten commands.
- Move data with `scp` for one-offs and `rsync -avz` for large or repeated transfers that resume.
- Run training inside `tmux` so it survives an SSH disconnect — `new -s`, detach with `Ctrl-b d`, reattach with `attach -t`.
- Hand unattended work to the OS: a `systemd` unit with `Restart=on-failure` for services, `cron` for scheduled jobs (always log their output).
- Attach a data volume, find it with `lsblk`, format a new one with `mkfs` (destructive — verify the device), `mount` it, and add it to `/etc/fstab` by UUID to persist across reboots.
- Use `nvidia-smi` to read GPU utilization, VRAM, and processes; `watch -n1 nvidia-smi` shows it live. Idle GPUs mean a bottleneck somewhere else.

## Try it

Using a Linux VM (a cheap CPU instance is fine for everything except the GPU step):

1. Generate an `ed25519` key on your laptop, add it to the VM, and set up an `~/.ssh/config` alias so you can connect with a single short word.
2. `rsync` a folder from your laptop to the VM, then add one file locally and rsync again — confirm it only transfers the new file.
3. Start a `tmux` session, run a long command like `sleep 600` inside it, detach, disconnect your SSH session entirely, reconnect, and reattach to find it still running.
4. Write a `cron` line that appends the date to a log file every minute. Watch the log grow, then remove the line.
5. If you have a GPU instance, run `watch -n1 nvidia-smi` in one pane while a small training script runs in another, and watch GPU-Util and memory move.

These are the moves you'll repeat every single day on a remote machine — worth getting into your fingers now.
