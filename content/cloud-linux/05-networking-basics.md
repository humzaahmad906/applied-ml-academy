# 05 — Networking Basics

Every cloud service — a website, a database, an API — is reachable only because of networking. When something "can't connect," the cause is almost always a networking concept you can learn in an afternoon: an address, a port, a name, or a firewall rule. This lesson demystifies the handful of ideas that explain how machines find and talk to each other, without the acronym overload.

## IP addresses: the phone numbers of the internet

Every machine on a network has an **IP address**, a numeric label that identifies it. The familiar form looks like `203.0.113.42` (four numbers, this is IPv4). There's a newer, longer form called IPv6, but the idea is the same: it's the address other machines use to reach this one.

In the cloud you'll meet two kinds:

- A **public IP** is reachable from the open internet. A web server needs one so browsers can find it.
- A **private IP** is only reachable inside your own cloud network. Databases and internal services usually get only a private IP, so they can't be reached directly from the internet at all — a strong, simple security default.

Cloud providers let you group your machines inside a private network of your own (AWS calls it a VPC, others use similar terms). Inside that network, machines talk to each other over private IPs; only the ones you deliberately expose get a public address.

## Ports: the doors on a machine

An IP address gets you to the right machine, but a single machine can run many services at once — a web server, a database, an SSH login. A **port** is a numbered door that identifies which service you want. A port is just a number from 0 to 65535.

Some ports are conventional:

- **22** — SSH (remote login)
- **80** — HTTP (unencrypted web)
- **443** — HTTPS (encrypted web)
- **5432** — a common database port (PostgreSQL)

So "connect to `203.0.113.42` on port 443" means "reach that machine's HTTPS web service." You can inspect what's listening locally:

```bash
ss -tlnp             # show listening TCP ports on this machine
curl http://localhost:80    # make a request to the local web server on port 80
```

## DNS: names instead of numbers

Nobody wants to memorize `203.0.113.42`. **DNS** (the Domain Name System) is the internet's phone book: it translates human-friendly names like `example.com` into IP addresses. When you type a domain, your computer asks a DNS server "what's the IP for this name?" and then connects to the answer.

```bash
dig example.com +short     # look up the IP address for a domain
nslookup example.com       # another lookup tool
```

The key **record types** you'll see:

- An **A record** maps a name to an IPv4 address.
- A **CNAME record** maps one name to another name (an alias).

DNS changes don't take effect instantly. Records carry a **TTL** (time to live) telling other servers how long to cache the answer, so an update can take minutes to hours to spread everywhere. This is why a freshly pointed domain sometimes "isn't working yet" — it's caching, not a mistake.

## Firewalls and security groups: who's allowed in

By default you don't want just anyone connecting to your machines. A **firewall** enforces rules about which traffic is allowed. In the cloud, the most common form is a **security group**: a set of rules attached to a VM (or a group of them) that says what's permitted.

A rule typically specifies:

- **Direction** — inbound (traffic coming to the machine) or outbound (traffic leaving it).
- **Port** — which door the rule applies to.
- **Source or destination** — which IP addresses are allowed, often written as a range.

For example, a web server's security group might allow inbound traffic on ports 80 and 443 from anywhere, allow inbound port 22 (SSH) only from your office's IP address, and deny everything else. That way the world can view your site, but only you can log in.

Security groups usually **default to denying** inbound traffic. That's a feature: you open exactly the doors you need and nothing more. When a service is mysteriously unreachable, an overly strict security group is the first thing to check — often the machine is fine and a rule simply isn't letting you in.

## Putting it together

Imagine visiting a website hosted on a cloud VM:

1. You type `example.com`. **DNS** resolves it to the VM's public **IP**.
2. Your browser connects to that IP on **port** 443 (HTTPS).
3. The VM's **security group** has a rule allowing inbound 443 from anywhere, so the connection is permitted.
4. The web server, listening on port 443, answers, and the page loads.

If any link in that chain is broken — wrong DNS record, service not listening on the port, or a firewall rule blocking it — the site won't load. Knowing the chain tells you exactly where to look.

## Key takeaways

- Every machine has an **IP address**; public IPs are reachable from the internet, private IPs only within your cloud network.
- **Ports** are numbered doors identifying which service on a machine you want (22 SSH, 80 HTTP, 443 HTTPS).
- **DNS** translates names like `example.com` into IP addresses; changes propagate slowly because of caching (TTL).
- **Firewalls / security groups** control which traffic is allowed, by direction, port, and source; they default to denying, so you open only what you need.
- Most "can't connect" problems trace to one link in the chain: DNS, IP, port, or a firewall rule.

## Try it

From any Linux machine with internet access:

1. Look up an address: run `dig example.com +short` (or `nslookup example.com`) and note the IP it returns.
2. See who's listening locally: run `ss -tlnp` and identify at least one service and the port it's on.
3. Make a real request: `curl -I https://example.com` — the `-I` shows just the response headers, confirming a successful connection on port 443.
4. Reason through a scenario: you launch a web server on a cloud VM but visitors get a timeout. List three networking causes to check, in order, and how you'd rule each one out.

If you can trace a connection from name to IP to port to firewall rule, networking will stop feeling like a black box.
