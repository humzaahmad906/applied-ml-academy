# 07 — Cost and Trade-offs

The cloud's greatest strength is also its most dangerous trap: you can summon almost unlimited resources with a few clicks, and the meter runs whether or not you're paying attention. Beginners routinely get surprise bills — a forgotten GPU instance, a runaway data transfer, storage that quietly piled up for months. This final lesson is about spending wisely: understanding how you're charged, choosing services deliberately, and building habits that prevent nasty surprises.

## How cloud billing actually works

The cloud rents you resources by usage, and the units are worth internalizing because they explain most surprise bills:

- **Compute** is billed by time — per second or per hour a machine runs. A VM costs the same whether it's crunching hard or sitting idle; if it's on, you pay.
- **Storage** is billed by capacity over time — dollars per gigabyte per month. It's usually cheap per gigabyte, but it accumulates silently and never stops until you delete the data.
- **Data transfer** is billed by volume moved, and this is the sneaky one. Moving data *into* the cloud is often free; moving it *out* to the internet, or between regions, frequently costs money. Large downloads and chatty cross-region traffic add up fast.
- **Requests and operations** — some services (object storage, databases, serverless functions) also charge tiny amounts per request. Individually negligible, but at scale they matter.

The mental shift from buying hardware is this: with the cloud there's no such thing as "already paid for." Everything left running is an ongoing cost.

## The core trade-off: flexibility versus commitment

Almost every cloud pricing decision is a spectrum between paying more for flexibility and paying less for commitment.

- **On-demand** pricing is the flexible, expensive end: no commitment, pay for what you use, walk away anytime. Right for experiments, unpredictable load, and anything short-lived.
- **Committed / reserved** pricing is the discounted, locked-in end: promise a year or more of usage and pay significantly less. Right for steady, predictable, always-on workloads.
- **Spot / preemptible** pricing is the cheap-but-interruptible option: use spare capacity at a deep discount, accepting that it can be reclaimed with little warning. Right for fault-tolerant batch work that can restart, like many machine learning training jobs that checkpoint their progress.

The discipline is matching the model to the workload. Paying on-demand rates for a server that runs 24/7 all year wastes money; committing to a year for a two-week experiment wastes money too.

## Managed versus self-managed

A second big trade-off is how much you run yourself. Take a database:

- **Self-managed**: run the database software on a plain VM. Cheaper in raw dollars, but *you* handle backups, patches, scaling, and 3 a.m. failures.
- **Managed**: use the provider's database service. It costs more per hour, but the provider handles the operational grind.

The right answer depends on the true cost, which includes your time. Managed services usually win for small teams and beginners, because the hours you'd spend babysitting infrastructure are worth more than the price difference. As you grow or your needs get unusual, self-managing specific pieces can pay off. Name both costs — dollars *and* effort — before deciding.

## Serverless and scaling to zero

Some services are **serverless**: you don't rent a machine at all, you just provide code or store data, and you're billed only for what you actually use — down to zero when idle. For spiky or low-traffic workloads this can be dramatically cheaper than a VM sitting on all day. The trade-off is less control and, at very high sustained volume, sometimes a higher unit cost than a dedicated machine. As a rule: serverless for bursty and small, dedicated machines for steady and heavy.

## Right-sizing: don't buy more than you need

A pervasive waste is **over-provisioning** — renting a machine far bigger than the job requires "to be safe." Because you can resize later, start small and scale up when you have evidence you need to. Watch actual CPU and memory use (with tools like `top`/`htop` on the machine, or the provider's monitoring dashboards) and pick the smallest resource that comfortably handles the load.

## Habits that prevent surprise bills

Cost control is mostly discipline, not cleverness:

- **Turn things off.** Stop or terminate VMs when you're done. This is the biggest single lever, especially for expensive GPU instances.
- **Set budgets and alerts.** Nearly every provider lets you set a spending threshold that emails you when you cross it. Do this on day one — it's the safety net that catches the mistake you didn't foresee.
- **Tag your resources.** Label resources by project or owner so you can see where the money goes and find orphaned resources to delete.
- **Clean up leftovers.** Detached storage volumes, old snapshots, unused IP addresses, and idle load balancers all quietly bill you. Sweep for them periodically.
- **Mind data egress.** Before moving large amounts of data out of the cloud or across regions, check what it costs. Keep computation close to where the data lives.
- **Use the free tier and calculators.** Providers offer free tiers for learning and pricing calculators for estimating a design's cost before you build it.

## Key takeaways

- You're billed for **compute** (by time), **storage** (by capacity over time), **data transfer** (by volume, especially outbound), and sometimes **per request**.
- Pricing trades **flexibility for commitment**: on-demand (flexible, pricey), reserved (discounted, locked-in), spot (cheap, interruptible) — match the model to the workload.
- **Managed** services cost more in dollars but save your time; weigh both costs, not just the price tag.
- **Serverless** shines for bursty, low-traffic work; dedicated machines win for steady, heavy load.
- **Right-size** by starting small and scaling on evidence; over-provisioning is pure waste.
- Prevent surprises with off-switches, **budgets and alerts**, tagging, cleanup, and awareness of data-egress charges.

## Try it

Design the cheapest sensible setup for a small project — say, a personal blog with occasional traffic spikes and a modest database:

1. For each piece (web serving, database, file/image storage), decide between on-demand, reserved, serverless, or managed, and justify it in one sentence.
2. Identify the one resource most likely to cause a surprise bill if you forget about it, and name the specific habit that would catch it.
3. If you have a cloud account, set a monthly budget alert right now — pick a low threshold. It takes a few minutes and is the best insurance against a runaway bill.
4. Open the provider's pricing calculator and estimate your design's monthly cost. Then try halving the biggest line item by changing one decision, and see how much you save.

If you can turn a project description into a cost-aware set of choices — and you've set an alert to catch your own mistakes — you're ready to use the cloud responsibly.
