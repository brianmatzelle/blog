---
title: "concurrently: What If Claude Code Ran Agents in Parallel?"
date: 2026-03-16
tags: [ai, agents, claude-code, concurrency, rust, tui]
---

# concurrently: What If Claude Code Ran Agents in Parallel?

Claude Code runs one conversation at a time. You ask it something, it works, you wait, you respond, it works some more. If you want three things done, you do them sequentially. The total wall clock time is the sum of all three.

I built [concurrently](https://github.com/brianmatzelle/concurrently) to run multiple Claude Code agents at once and chat with all of them from a single terminal.

## The Problem

Say you need to add auth to an API, write tests for it, and update the docs. Three independent tasks. In Claude Code, you'd do them one at a time:

```
auth code     ──────────────────>  done
                                   tests         ──────────────────>  done
                                                                      docs  ──────────────────>  done
                                                                                                ↑
                                                                                         still waiting
```

Each task takes 30-60 seconds. You're looking at 2-3 minutes. These tasks are independent -- the test writer doesn't need to wait for the auth code. The docs writer doesn't need either of them.

## What concurrently Does

```
  Spawned agent-1: add auth to the REST API
  Spawned agent-2: write tests for the auth endpoints
  Spawned agent-3: update the docs with the new auth flow
[agent-1] I'll start by examining the current route structure...
[agent-1] Read: src/routes/api.rs
[agent-2] Let me look at the existing test setup...
[agent-2] Read: tests/api_test.rs
[agent-3] I'll check the current docs...
[agent-1] Edit: src/routes/api.rs
[agent-3] Read: docs/api.md
[agent-1] Adding JWT middleware to all /api routes...
[agent-2] Bash: cargo test
[agent-3] Edit: docs/api.md
  -- agent-2 done ($0.0312) --
> @agent-2 also add a test for expired tokens
[agent-2] Good catch. Adding an expiry test...
```

You spawn agents with `/s <task>`. Each one is a real Claude Code process -- an actual `claude -p` instance with full tool access. File editing, bash, search, everything. They all run simultaneously, and their output interleaves into a single chat stream. Each agent gets a distinct color and name.

When an agent finishes a turn, you can reply. Bare text goes to the most recently idle agent. `@agent-2` targets a specific one. Tab cycles focus. The interaction model is less "submit a batch job" and more "manage a team in one group chat."

## What You Can Do With It

The interesting stuff isn't "the same tasks but faster." Parallelism changes what's practical.

### Multi-file refactors

"Rename this API from v1 to v2 across all services." One `/s` per service. They each search their service for v1 references, update routes, update clients, update tests. They don't step on each other because they're in different directories. What would be a 10-minute sequential crawl finishes in the time it takes to refactor one service.

### Research + implement simultaneously

Spawn one agent to read the middleware and explain how it works. Spawn another to start implementing rate limiting on it. They both read the same files -- that's fine, reads don't conflict. By the time you're reading agent 1's explanation, agent 2 already has a working implementation. If agent 2 has a question, you answer it without interrupting agent 1.

### Parallel code review

Throw a diff at three agents: one for security issues, one for performance, one for API design. Three focused reviews in the time it takes to do one. If the security reviewer flags something, you can tell it to dig deeper while the other two keep working.

### Competitive approaches

```
/s implement a cache layer using Redis
/s implement a cache layer using an in-memory LRU
/s implement a cache layer using SQLite
```

Three agents, three implementations, same interface. Compare them side by side. This is impractical sequentially -- you wouldn't wait 3 minutes for three competing implementations. At 60 seconds total, it's a useful way to explore the design space.

### Audit at scale

"Check every API endpoint for proper input validation." One agent per module or route file. A codebase with 15 route files gets 15 agents scanning simultaneously. The alternative is one agent reading files sequentially, which means either it takes forever or it cuts corners.

## What Doesn't Parallelize

Not everything is independent. If task B depends on the output of task A, don't spawn them at the same time. That's your call -- you decide what to spawn and when.

The other constraint is write conflicts. Two agents editing the same file at the same time will produce chaos. Scope your agents to different files or directories. If a task requires coordinated edits to a single file, that's one agent's job, not two.

## The Real Win

The O(n) to O(1) framing is the clean way to think about it, but the practical win is more subtle. It's not just about wall clock time. It's about what you're willing to ask for.

When each task takes a minute and they run sequentially, you naturally scope your requests down. "Just fix this one thing." You avoid ambitious asks because the feedback loop is too slow.

When agents run in parallel and you can chat with each one independently, you start asking for more. "Refactor all three services." "Review the entire PR from five angles." "Implement it three different ways and let me compare." The cost per request goes up (more agents = more API calls), but the time doesn't. And time is the thing you're actually bottlenecked on.

## Install

```bash
# Arch Linux (AUR)
yay -S concurrently-bin

# macOS (Homebrew)
brew tap brianmatzelle/tap
brew install concurrently
```

Requires the `claude` CLI installed and on PATH.
