# 01 — Why Containers

You just finished a machine learning project. It trains, it predicts, everything works beautifully on your laptop. You hand it to a teammate, and within five minutes you get the message every engineer dreads: "It doesn't work on my machine." This lesson is about why that happens and how containers make it stop happening.

## The reproducibility problem

Software never runs in a vacuum. Your ML script depends on a specific version of Python, a pile of libraries like NumPy and PyTorch, system packages, environment variables, and sometimes the exact operating system underneath. Change any one of those and behavior can shift, sometimes silently.

A few concrete ways this bites you:

- You have Python 3.11; your teammate has 3.9, and a function you rely on was added in 3.10.
- You installed a library six months ago and it quietly upgraded. Now the same code produces different numbers.
- Your model loads a CUDA driver that exists on your GPU box but not in the cloud instance you deployed to.
- A script works in your terminal because of a setting in your shell profile that nobody else has.

"Works on my machine" is not a joke, it is a symptom. The machine you developed on carries hundreds of tiny assumptions, and none of them travel with your code when you copy the files somewhere else. For ML this is especially painful because results are supposed to be *reproducible*: the same data and the same code should give the same answer. If the environment drifts, reproducibility is gone.

## What a container actually is

A container is a way to package your application *together with everything it needs to run*: the code, the libraries, the system tools, the configuration. That bundle runs the same way on your laptop, your teammate's laptop, a server, or a cloud instance, because the environment is no longer borrowed from the host, it is carried inside the package.

The mental model that helps most people: a container is a lightweight, isolated box that has its own filesystem, its own installed software, and its own view of the world, but shares the host machine's operating system kernel. Inside the box, your program sees exactly the environment you defined. Outside the box, the host barely notices it is there.

That isolation is the whole point. Two containers on the same machine can use two different versions of Python without conflict, because neither one can see the other's files.

## Containers versus virtual machines

If you have used a virtual machine (VM), containers might sound familiar. Both give you isolation. The difference is what they include.

A virtual machine emulates an entire computer. It runs a full guest operating system on top of your real one, managed by a hypervisor. That means every VM carries its own kernel, its own system processes, gigabytes of overhead, and takes minutes to boot.

A container skips the guest operating system. It shares the host's kernel and only packages the layers *above* it: your application and its dependencies. The result is dramatically lighter.

| | Virtual Machine | Container |
|---|---|---|
| Includes a full guest OS | Yes | No |
| Startup time | Seconds to minutes | Milliseconds to seconds |
| Typical size | Gigabytes | Megabytes to hundreds of MB |
| Isolation | Very strong (separate kernel) | Strong (shared kernel) |
| Resource overhead | High | Low |

For ML work this matters. You can run a dozen containers on one machine, spin them up in seconds during an experiment, and ship a tidy image that is a fraction of a VM's size. When you need to scale an inference service to handle more traffic, launching ten more lightweight containers is far cheaper than booting ten more virtual machines.

## Where containers fit in ML

Think about the full lifecycle of a model. You develop in a notebook, train on a GPU box, then deploy the trained model behind an API so other services can call it. Each of those environments is different. Containers let you define the environment *once* and carry it through every stage.

- **Development**: everyone on the team runs the identical environment, so onboarding is `run the container` instead of a two-day setup document.
- **Training**: the exact library versions used to train are captured, so the run is reproducible months later.
- **Deployment**: the container that passed your tests is the same artifact that runs in production. Nothing gets rebuilt or reinstalled along the way, so nothing drifts.

This is why containers are often called the unit of deployment for modern ML. The thing you build and test is the thing that ships.

## The toolchain

The most common tool for building and running containers is Docker. Throughout this course you will use its command line, which starts with `docker`. Here is the very first command worth knowing, which confirms the tool is installed and working:

```bash
# Print the installed Docker version
docker --version
```

If that prints a version number, you are ready. If it errors, Docker is not installed or not running yet, and the rest of the course will assume it is available.

## Key takeaways

- Software depends on its environment, and that environment does not travel with your code by default. This is the root of "works on my machine."
- A container packages your application together with its dependencies so it runs identically everywhere.
- Containers share the host's kernel, making them far lighter and faster than virtual machines, which each carry a full guest OS.
- In ML, containers give you reproducibility across development, training, and deployment. The artifact you test is the artifact you ship.

## Try it

You do not need Docker installed to do this reflection, but if you have it, run `docker --version` first.

1. Open a terminal and run `python --version` (or `python3 --version`). Write down the exact version.
2. Ask a friend or colleague to run the same command on their machine, or check a cloud instance you have access to. Compare.
3. Now list three libraries your last project used and note which versions you have installed with `pip list`.
4. Ask yourself: if you copied only your code files to another machine, would it run? Write down every piece of the environment that would need to match. That list is exactly what a container will capture for you in the coming lessons.
