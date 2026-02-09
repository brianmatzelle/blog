---
title: "Hosting a Fullstack App Exclusively on Tailscale (Part 2)"
date: 2026-02-09
tags: [linux, tailscale, selfhosting, systemd, nextjs, fastapi, nftables]
---

# Hosting a Fullstack App Exclusively on Tailscale (Part 2)

In [Part 1](/2026-02-09-mullvad-tailscale-coexistence), I got Mullvad VPN and Tailscale running simultaneously on the same Linux server using nftables traffic marking. This post covers the next step: hosting a fullstack application that is *only* reachable from my Tailscale network -- invisible to the public internet, but accessible from my phone, laptop, or any device on my tailnet.

## The Architecture

The app is a two-service stack:

- **Backend**: A FastAPI server with an MCP endpoint that scrapes streaming sources and proxies HLS video. Runs on port 8000.
- **Frontend**: A Next.js app that talks to Claude AI, calls MCP tools on the backend, and renders video streams. Runs on port 3000.

Both services bind to `0.0.0.0`, meaning they listen on all network interfaces. But the only interface that matters for access is `tailscale0`. The server has no ports forwarded on the router, no public DNS, no reverse proxy. If you're not on my tailnet, these ports don't exist to you.

```
Phone (Tailscale) ──> tailscale0 ──> Next.js (:3000)
                                        │
                                        ├──> Claude API (outbound, via Mullvad)
                                        │
                                        └──> FastAPI/MCP (:8000, localhost)
                                                │
                                                └──> Stream sources (outbound, via Mullvad)
```

The browser on my phone talks to Next.js over Tailscale. Next.js server-side API routes talk to the MCP backend over localhost. All outbound traffic (Claude API calls, stream scraping) exits through Mullvad. The only thing traversing Tailscale is the client-server traffic between my devices and the app.

## Making Services Tailscale-Aware

### The Backend Problem

The backend has an MCP tool called `SHOW_CONTENT` that returns video stream URLs. These URLs point back to the backend's HLS proxy -- the browser fetches them directly. If the tool returns `http://localhost:8000/mcp/proxy/playlist.m3u8`, that works when the browser is on the same machine. But my phone isn't on the same machine. It needs the Tailscale IP.

The fix: the tool reads a `PUBLIC_SERVER_URL` environment variable at startup, falling back to localhost if unset.

```python
server_url: str = os.environ.get("PUBLIC_SERVER_URL", "http://localhost:8000")
```

The systemd service populates this dynamically:

```ini
ExecStartPre=/bin/bash -c 'echo PUBLIC_SERVER_URL=http://$(tailscale ip -4):8000 > /tmp/crackgpt-env'
EnvironmentFile=-/tmp/crackgpt-env
```

On startup, it asks Tailscale "what's my IP?", writes it to a temp env file, and the service reads it. If Tailscale is down, it falls back to localhost gracefully.

### The Frontend Problem

Next.js dev server defaults to `localhost:3000` -- only accessible from the machine itself. Adding `--hostname 0.0.0.0` to the dev script fixes this:

```json
"dev": "next dev --turbopack --hostname 0.0.0.0"
```

The Next.js server-side code connects to the MCP backend using the URL from `config.json`, which is `http://localhost:8000`. This is fine -- both services are on the same machine, so server-to-server communication stays on loopback. Only the browser-facing URLs need to use the Tailscale IP.

### CORS

The backend's CORS allowlist needs to include the Tailscale origins, otherwise the browser blocks cross-origin requests to port 8000:

```python
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://<tailscale-ip>:3000",
    "http://<tailscale-dns>:3000",
]
```

## Systemd Services

Both services run as systemd units, ordered to start after Tailscale.

### Backend

```ini
[Unit]
Description=crackGPT Backend (FastAPI/MCP)
After=network.target tailscaled.service mullvad-daemon.service
Wants=tailscaled.service

[Service]
Type=simple
User=cowboy
WorkingDirectory=/home/cowboy/projects/crackGPT
Environment=PATH=/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin
ExecStartPre=/bin/bash -c 'echo PUBLIC_SERVER_URL=http://$(tailscale ip -4 2>/dev/null || echo localhost):8000 > /tmp/crackgpt-env'
EnvironmentFile=-/tmp/crackgpt-env
ExecStart=/home/linuxbrew/.linuxbrew/bin/uv run uvicorn server:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Frontend

```ini
[Unit]
Description=crackGPT Frontend (Next.js)
After=network.target crackgpt-backend.service
Wants=crackgpt-backend.service

[Service]
Type=simple
User=cowboy
WorkingDirectory=/home/cowboy/projects/crackGPT/web-client
Environment=PATH=/home/cowboy/.nvm/versions/node/v24.13.0/bin:/usr/local/bin:/usr/bin:/bin
Environment=NODE_ENV=development
ExecStart=/home/cowboy/.nvm/versions/node/v24.13.0/bin/npm run dev
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

The dependency chain on boot:

```
mullvad-daemon
    └──> tailscaled (loads nftables marks via ExecStartPre)
             └──> crackgpt-backend (resolves Tailscale IP via ExecStartPre)
                      └──> crackgpt-frontend
```

## What's Exposed, What's Not

This is what I like about this setup. There's no nginx, no Cloudflare tunnel, no port forwarding. The attack surface is minimal:

| Interface | Ports open | Who can reach it |
|-----------|-----------|-----------------|
| `enp0s31f6` (LAN) | None forwarded | LAN devices only, but app isn't advertised |
| `wg0-mullvad` | None listening | Mullvad tunnel, outbound only |
| `tailscale0` | 3000, 8000 | Tailscale devices authenticated to my tailnet |
| `lo` | 3000, 8000 | Local only |

The services bind to `0.0.0.0`, so technically a LAN device could hit `192.168.x.x:3000`. If you want to lock it down further, you could bind only to the Tailscale IP. But for a home network behind a router with no port forwarding, this is fine for my purposes.

Tailscale handles authentication -- every device on the tailnet is identified by a WireGuard key tied to my account. There's no username/password for the app itself. If you're on my tailnet, you're me.

## The Day-to-Day

I open my phone, type `http://<tailscale-ip>:3000` in the browser, and the app is there. Works from home on Wi-Fi (direct LAN connection via Tailscale), works from a coffee shop on cellular (Tailscale DERP relay), works from another machine on my tailnet. The server sits in a closet, Mullvad keeps its outbound traffic private, and Tailscale keeps the app private to my devices.

If the server reboots, everything comes back up automatically. If Mullvad reconnects to a different relay, the nftables marks still apply. If Tailscale assigns a different IP (unlikely but possible), the backend's `ExecStartPre` picks up the new one.

No DNS to manage. No certificates to renew. No cloud hosting bill. Just two WireGuard tunnels, some nftables marks, and four systemd units.

## Setup Summary

For anyone replicating this:

1. Install Tailscale and Mullvad on the server
2. Get them coexisting with nftables marks ([Part 1](/2026-02-09-mullvad-tailscale-coexistence))
3. Bind your services to `0.0.0.0`
4. Make any service-generated URLs use the Tailscale IP instead of localhost
5. Add Tailscale origins to CORS if your frontend and backend are on different ports
6. Create systemd services ordered after `tailscaled.service`
7. That's it. Your tailnet is your private cloud.
