---
title: "concurrently: What If Claude Code Ran Agents in Parallel?"
date: 2026-03-16
tags: [ai, agents, claude-code, concurrency, rust, tui]
---

# concurrently: What If Claude Code Ran Agents in Parallel?

Claude Code runs agents sequentially. You ask it to do something complex, it spawns a sub-agent, waits for the result, spawns the next one, waits again. Each agent is fully capable -- it can read files, edit code, run commands, search your codebase -- but they take turns. The total wall clock time is the sum of all agent runtimes.

I built [concurrently](https://github.com/brianmatzelle/concurrently-releases) to see what happens when you run them all at once.

## The Problem

Say you ask Claude Code: "Add authentication to this REST API, write tests for it, and update the docs." Internally, that might decompose into three sub-agents:

```
agent1 (auth code)    ──────────────────>  done
                                           agent2 (tests)    ──────────────────>  done
                                                                                  agent3 (docs)  ──────────────────>  done
                                                                                                 ↑
                                                                                          you're still waiting
```

Each agent takes maybe 30-60 seconds. You're looking at 2-3 minutes. And these tasks are independent -- the test writer doesn't need to wait for the auth code to be committed before it can start scaffolding test files. The docs writer doesn't need either of them.

This is O(n) in the number of subtasks. The total time scales linearly with how many things need to happen. For a queue, that's fine. For a developer staring at a terminal, it's slow.

## What concurrently Does

```
agent1 (auth code)    ──────────────────>  done
agent2 (tests)        ──────────────────>  done
agent3 (docs)         ──────────────────>  done
                                           ↑
                                     all done at once
```

O(1). Bounded by the slowest agent, not the sum. Three 45-second agents finish in 45 seconds, not 135.

You type a task. An orchestrator decomposes it into independent subtasks. Each subtask spawns a real Claude Code process -- not a chat completion, an actual `claude -p` instance with full tool access. They all run simultaneously. A TUI shows every agent's live output, what files they're reading, what commands they're running, what edits they're making. When they're done, you can synthesize the results into a single coherent output.

Every agent sees the same shared context -- a conversation "kernel" that accumulates across rounds. Round 2's agents know what round 1 discussed and produced.

## What You Can Do With It

The interesting stuff isn't "the same tasks but faster." Parallelism changes what's practical.

### Multi-file refactors

"Rename this API from v1 to v2 across all services." One agent per service. They each search their service for v1 references, update routes, update clients, update tests. They don't step on each other because they're in different directories. What would be a 10-minute sequential crawl finishes in the time it takes to refactor one service.

### Research + implement simultaneously

"How does the auth middleware work? Also, add rate limiting to it." Agent 1 reads the middleware, traces the call chain, writes an explanation. Agent 2 starts implementing rate limiting immediately. They both read the same files -- that's fine, reads don't conflict. By the time you're reading agent 1's explanation, agent 2 already has a working implementation.

### Parallel code review

Throw a diff at it with: "Review this PR for security issues, performance problems, and API design." Three agents, three lenses, all reading the same diff simultaneously. You get three focused reviews in the time it takes to do one.

### Competitive approaches

"Implement a cache layer for this API. One agent should use Redis, one should use an in-memory LRU, one should use SQLite." Three agents, three implementations, same interface. Compare them side by side. This is impractical sequentially -- you wouldn't wait 3 minutes for three competing implementations. At 60 seconds total, it's a useful way to explore the design space.

### Audit at scale

"Check every API endpoint in this repo for proper input validation." One agent per module or route file. A codebase with 15 route files gets 15 agents scanning simultaneously. The alternative is one agent reading files sequentially, which means either it takes forever or it cuts corners and skims.

## What Doesn't Parallelize

Not everything is independent. If task B genuinely depends on the output of task A, you can't run them at the same time. The orchestrator handles this -- it only decomposes into subtasks that can run without dependencies. If the task is inherently sequential ("first create the database schema, then write the migration, then update the ORM models"), it'll either run it as one agent or decompose it into a single subtask.

The other constraint is write conflicts. Two agents editing the same file at the same time will produce chaos. The decomposition step accounts for this by scoping agents to different files or directories. But if your task genuinely requires coordinated edits to a single file, that's a sequential task.

## The Real Win

The O(n) → O(1) framing is the clean way to think about it, but the practical win is more subtle. It's not just about wall clock time. It's about what you're willing to ask for.

When each sub-agent takes a minute and they run sequentially, you naturally scope your requests down. "Just fix this one thing." You avoid ambitious asks because the feedback loop is too slow.

When agents run in parallel, you start asking for more. "Refactor all three services." "Review the entire PR from five angles." "Implement it three different ways and let me compare." The cost per request goes up (more agents = more API calls), but the time doesn't. And time is the thing you're actually bottlenecked on.

## Install

```bash
# Arch Linux (AUR)
yay -S concurrently-bin

# macOS (Homebrew)
brew tap brianmatzelle/tap
brew install concurrently
```

Requires the `claude` CLI installed and `ANTHROPIC_API_KEY` set.
