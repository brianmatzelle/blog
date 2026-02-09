---
title: "Is This Secure? Running Mullvad and Tailscale Simultaneously with nftables"
date: 2026-02-09
tags: [linux, networking, tailscale, mullvad, nftables, vpn, security]
---

# Is This Secure? Running Mullvad and Tailscale Simultaneously with nftables

I got Mullvad VPN and Tailscale running at the same time on my Linux server by marking Tailscale's CGNAT range as excluded traffic in Mullvad's nftables firewall. It works, but I'm punching holes in a kill switch that exists for a reason. I'd like to hear from people who know nftables better than me: **did I do this right, or did I open something I shouldn't have?**

## What I'm Trying to Do

I run a home server (Debian, Linux 6.12) that proxies video streams and serves them to my devices. The requirements:

- **Outbound traffic** (fetching streams from the internet) should go through **Mullvad VPN**
- **Inbound traffic** (my phone accessing the app) should come through **Tailscale**
- Both at the same time

## What Happens Out of the Box

Neither service knows the other exists, and they fight over routing and DNS.

| Mullvad | Tailscale | Internet | Tailscale reachable |
|---------|-----------|----------|---------------------|
| Off     | Off       | Yes      | No                  |
| On      | Off       | Yes      | No                  |
| Off     | On        | No       | Yes                 |
| On      | On        | No       | No                  |

Row 4 is the goal state, and it's the worst one.

## Why It Breaks

Mullvad creates a `wg0-mullvad` interface and enforces its kill switch via nftables. Dumping the ruleset (`nft list table inet mullvad`) reveals `input` and `output` chains with a **policy of drop**. Traffic is only allowed if it:

- Passes through `wg0-mullvad`
- Carries `ct mark 0x00000f41` (Mullvad's exclusion mark)
- Falls within RFC1918 ranges (10.x, 172.16.x, 192.168.x) when LAN sharing is on

Tailscale uses **100.64.0.0/10** (CGNAT). Not RFC1918. Mullvad's LAN sharing doesn't cover it. Every Tailscale packet hits the default drop policy.

The DNS problem is the other half. Both services want to own DNS resolution. When Tailscale takes over DNS but Mullvad is the only allowed path to the internet, DNS dies.

## My Solution

Three steps, applied in order. I did these interactively rather than scripting them, because bricking networking on a remote box is not fun.

### 1. Enable Mullvad LAN sharing

```bash
mullvad lan set allow
```

Doesn't fix Tailscale directly (wrong IP range), but keeps DHCP and local discovery working.

### 2. Start Tailscale without DNS management

```bash
sudo tailscale up --accept-dns=false --accept-routes=false
```

This is critical. Tailscale will complain with a health warning about not reaching DNS servers. Ignore it -- Mullvad handles DNS via its resolver at 10.64.0.1 through the tunnel.

### 3. Mark Tailscale traffic with Mullvad's exclusion marks

Here's the part I want scrutinized:

```bash
sudo nft -f - <<'EOF'
table inet mullvad_tailscale {
  chain outgoing {
    type route hook output priority -100; policy accept;
    ip daddr 100.64.0.0/10 ct mark set 0x00000f41 meta mark set 0x6d6f6c65;
  }
  chain incoming {
    type filter hook input priority -100; policy accept;
    ip saddr 100.64.0.0/10 ct mark set 0x00000f41 meta mark set 0x6d6f6c65;
  }
}
EOF
```

This creates a separate nftables table that runs at priority `-100` (before Mullvad's chains at `0`). It stamps two marks onto any packet in the Tailscale CGNAT range:

- `ct mark 0x00000f41` -- connection tracking mark. Mullvad's input and output chains both have `ct mark 0x00000f41 accept` near the top, so marked packets sail through the kill switch.
- `meta mark 0x6d6f6c65` -- packet mark (hex for "mole"). Mullvad's routing policy uses this to bypass the VPN tunnel for excluded traffic.

By the time Mullvad's `chain input` evaluates a packet from the Tailscale CGNAT range, it already carries the exclusion mark and gets accepted.

## What I'm Not Sure About

This works. But I have questions:

**1. Am I leaking traffic outside the Mullvad tunnel?** The marks tell Mullvad "this traffic is excluded, let it bypass the tunnel." That's intentional for 100.64.0.0/10 -- Tailscale has its own WireGuard encryption. But could a crafted packet with a spoofed source in the CGNAT range bypass the kill switch and reach the internet unencrypted?

**2. Is the CGNAT range too broad?** 100.64.0.0/10 is the entire CGNAT block. My Tailscale network only uses a few IPs. Would it be more secure to mark only my specific Tailscale subnet or even individual peer IPs?

**3. Does `policy accept` on these chains matter?** Both chains have `policy accept`, but they only set marks -- they don't make accept/drop decisions themselves. Mullvad's chains (at priority 0) still do the actual filtering. I think this is fine, but I'm not 100% sure a `policy accept` at priority -100 can't interfere.

**4. Should the `incoming` chain be `type filter` or `type route`?** I used `type filter` for input and `type route` for output, matching what I found in community examples. Is there a reason to prefer one over the other here?

**5. Is there a better way entirely?** Tailscale has an official Mullvad integration where you use Mullvad exit nodes through Tailscale's control plane. That would avoid all this nftables surgery. But it means giving up the standalone Mullvad client and routing Mullvad through Tailscale's infrastructure. For my use case (server that needs both inbound Tailscale and outbound Mullvad), is the nftables approach reasonable, or am I overcomplicating this?

## The Result

With all three steps applied:

| What | Route |
|------|-------|
| Outbound HTTP | server -> wg0-mullvad -> Mullvad exit -> internet |
| Inbound from phone | phone -> Tailscale WireGuard -> tailscale0 -> server:3000 |
| DNS | Mullvad resolver (10.64.0.1 via wg0-mullvad) |

```
$ curl -s https://am.i.mullvad.net/ip
<mullvad exit ip>

$ tailscale status
100.x.x.x   server      user@  linux
100.x.x.x   phone       user@  iOS
100.x.x.x   desktop     user@  linux
```

Both work. Phone loads the app over Tailscale. Outbound requests exit through Mullvad.

## Persistence

These nftables rules don't survive a reboot. To make them permanent, save to a file and hook into the tailscaled service:

```bash
sudo mkdir -p /etc/nftables
sudo tee /etc/nftables/mullvad-tailscale.conf <<'EOF'
table inet mullvad_tailscale {
  chain outgoing {
    type route hook output priority -100; policy accept;
    ip daddr 100.64.0.0/10 ct mark set 0x00000f41 meta mark set 0x6d6f6c65;
  }
  chain incoming {
    type filter hook input priority -100; policy accept;
    ip saddr 100.64.0.0/10 ct mark set 0x00000f41 meta mark set 0x6d6f6c65;
  }
}
EOF
```

Then add an `ExecStartPre` to tailscaled so the rules load before Tailscale starts:

```bash
printf '[Service]\nExecStartPre=/usr/sbin/nft -f /etc/nftables/mullvad-tailscale.conf\n' \
  | sudo SYSTEMD_EDITOR=tee systemctl edit tailscaled.service
```

The `--accept-dns=false` flag doesn't need its own persistence -- Tailscale stores it in `/var/lib/tailscale/tailscaled.state` after the first `tailscale up` call, so it carries over across restarts.

On boot, the sequence is: Mullvad connects via its own service, then `tailscaled` starts, runs the `ExecStartPre` to load the nftables marks, and comes up with DNS disabled. No manual intervention needed.

## Teardown

```bash
sudo nft delete table inet mullvad_tailscale
sudo tailscale down
mullvad status  # still connected, kill switch intact
```

## Setup

- **OS**: Debian (Linux 6.12)
- **Mullvad relay**: Sweden
- **Services**: FastAPI backend on :8000, Next.js frontend on :3000

If you see a problem with this approach, I'd genuinely like to know. Thanks for reading.
