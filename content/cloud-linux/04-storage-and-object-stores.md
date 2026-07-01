# 04 — Storage and Object Stores

Computers are useless without somewhere to keep data. In the cloud, storage isn't one thing — it comes in a few distinct flavors, each with different trade-offs. Picking the wrong kind is a common and expensive mistake: people store terabytes of backups on premium disks, or try to run a database on storage that was never designed for it. This lesson explains the main types and, more importantly, when to use each.

## Block storage: the virtual hard drive

**Block storage** behaves like a physical hard drive attached to a computer. The cloud calls these **volumes** (AWS EBS, Google Persistent Disk, Azure Managed Disks). You attach a volume to a VM, format it with a filesystem, mount it, and from then on it looks like any ordinary disk — you read and write files with the normal Linux commands.

```bash
lsblk                          # list block devices attached to this machine
sudo mount /dev/sdb1 /data     # mount a volume at /data
df -h                          # show disk space usage in human-readable form
```

Block storage is fast and low-latency, which is exactly what an operating system or a database needs. Its defining trait: it attaches to **one machine at a time**. It's the disk your VM boots from and where a database keeps its files.

Use block storage for:

- The boot disk of a VM.
- Databases and anything that does frequent, small, random reads and writes.
- Any workload that expects a normal filesystem.

## Object storage: the infinite bucket

**Object storage** works completely differently. Instead of a disk with folders, you have a **bucket** — a flat container that holds **objects**. Each object is a blob of data (a file's contents) plus some metadata and a unique **key** (its name). Examples are Amazon S3, Google Cloud Storage, and Azure Blob Storage.

You don't mount a bucket like a disk. You talk to it over the network through an API, usually with a command-line tool or a library in your program:

```bash
# the exact tool depends on the provider; the shape is always similar
cloud storage cp report.pdf s3://my-bucket/reports/report.pdf   # upload
cloud storage ls s3://my-bucket/reports/                        # list objects
cloud storage cp s3://my-bucket/reports/report.pdf ./           # download
```

Notice the key looks like a path (`reports/report.pdf`), but that's just a naming convention — buckets are flat, with no real folders underneath.

Object storage is built for scale and durability. A single bucket can hold effectively unlimited data, providers keep multiple copies so objects almost never get lost, and it's cheap per gigabyte. The trade-off is that it's higher latency and you can't do quick in-place edits — you replace a whole object rather than changing a few bytes in the middle.

Use object storage for:

- Large files: images, video, backups, logs, datasets.
- Machine learning training data and saved model files.
- Static website assets and anything you want to serve or share at scale.
- Data that many machines or services need to read at once.

## Storage classes: paying for how often you read

Object storage often comes in **classes** that trade retrieval speed for price. A "standard" class is cheap to read and slightly pricier to store, meant for data you touch often. "Cold" or "archive" classes cost very little to store but charge more (and take longer) to retrieve — perfect for backups and compliance data you rarely open. Matching the class to your access pattern can cut storage bills dramatically.

## File storage: the shared drive

There's a third option worth knowing: **file storage** (network file systems like NFS). It sits between the other two. Like block storage it presents a normal filesystem with folders, but like object storage many machines can mount it at once. It's handy when a group of servers needs to share the same set of files with familiar filesystem semantics. It's usually pricier than object storage, so reach for it only when you specifically need shared, mountable, read-write access from multiple machines.

## A quick decision guide

- Need a disk for one VM or a database, with fast random access? **Block storage.**
- Have lots of files, backups, datasets, or media to store cheaply and access over the network? **Object storage.**
- Need several machines to mount and share the same filesystem? **File storage.**

## Durability versus backups

One caution: durability (the provider keeping multiple copies so data isn't lost to hardware failure) is **not** the same as a backup. If you accidentally delete an object or overwrite a file with garbage, high durability faithfully preserves the mistake. You still need **snapshots** (point-in-time copies of a volume) and **versioning** (keeping old copies of objects in a bucket) to recover from human error.

## Key takeaways

- **Block storage** is a virtual hard drive attached to one machine — fast, low-latency, ideal for boot disks and databases.
- **Object storage** is a bucket of key-addressed blobs accessed over the network — cheap, virtually unlimited, ideal for files, datasets, and backups.
- **File storage** is a shared, mountable filesystem for when multiple machines need the same files.
- Object storage classes let you trade retrieval speed for lower storage cost based on how often you read the data.
- Durability protects against hardware failure, not against your own mistakes — you still need snapshots and versioning.

## Try it

Reason through the storage for a small photo-sharing app:

1. The app runs on a VM and needs a disk to boot from and to hold its program files. Which storage type, and why?
2. Users upload millions of photos that must be stored cheaply and served to browsers worldwide. Which storage type?
3. Some photos are viewed constantly (recent uploads); others are years old and rarely opened. How would storage classes help, and which would you assign to each?
4. If you have access to a cloud account with a free tier, create a bucket, upload a file with the provider's copy command, list the bucket to confirm it's there, then download it back and delete it.

If you can defend each choice in one sentence, you understand cloud storage.
