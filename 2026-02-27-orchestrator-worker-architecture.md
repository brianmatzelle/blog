---
title: "The Orchestrator Pattern: How Claude Code's Sub-Agent Architecture Actually Works"
date: 2026-02-27
tags: [ai, agents, architecture, claude-code, mcp, sub-agents]
---

# The Orchestrator Pattern: How Claude Code's Sub-Agent Architecture Actually Works

I've been using Claude Code daily to build [manim-mcp](https://github.com/brianmatzelle/manim-mcp), a math tutoring platform that renders Manim animations via MCP. At some point I wanted to move the project toward a multi-agent architecture, and the best reference implementation I had was sitting right in front of me -- Claude Code itself.

So I asked it to introspect. "How does `/init` work? Is there a pre-built agent DAG? Do sub-agents coordinate with each other?" The answers were clarifying.

## What I Expected

I assumed `/init` triggered some pre-configured pipeline. Something like:

```
/init → Agent A (scan structure) → Agent B (find commands) → Agent C (write CLAUDE.md)
```

A fixed graph, agents with predefined roles, maybe a coordinator passing state between them.

## What Actually Happens

`/init` is a **prompt expansion**. The slash command gets expanded into a detailed set of instructions ("analyze this codebase and create a CLAUDE.md file...") that gets injected into the orchestrator's context. From there, it's a single agent -- the orchestrator -- making all decisions at runtime.

There is no DAG. There is no pipeline configuration. The orchestrator reads the expanded prompt and decides:

- Whether to use sub-agents at all (for trivial tasks, it doesn't)
- What each sub-agent should do (tasks are defined at runtime, not pre-configured)
- Whether to parallelize or go sequential
- When to spawn follow-up agents based on initial results
- When it has enough information to act directly

```
User Request
    |
    v
Prompt Expansion (optional -- transforms shorthand into detailed instructions)
    |
    v
Orchestrator (full context, all decision-making power)
    |--- Sub-agent A (scoped task, returns result, dies)
    |--- Sub-agent B (parallel)
    |--- Sub-agent C (parallel)
    |
    v
Orchestrator synthesizes, decides next step
    |--- Maybe spawns more agents
    |--- Or acts directly
    |
    v
Output
```

The intelligence is in the routing, not the workers.

## Five Design Decisions That Make It Work

### 1. Sub-agents are stateless and disposable

A sub-agent gets a prompt, does work, returns a single message, and is gone. It doesn't know about other sub-agents. It can't talk to them. All coordination flows through the orchestrator.

This is the opposite of how most people think about multi-agent systems. The instinct is to build agents that maintain state and communicate laterally. But lateral communication creates coordination overhead, race conditions, and debug nightmares. The hub-and-spoke model is simpler and the orchestrator already has all the context it needs to coordinate.

### 2. Sub-agents have scoped tool access

Different agent types get different tools. An "Explore" agent can read files and search code but can't edit anything. A "Plan" agent can research and design but can't write files. A general-purpose agent gets everything.

This is blast-radius control. A research task can't accidentally edit your codebase. A planning task can't jump ahead and start implementing. The constraints aren't just safety -- they help the sub-agent focus. When you can only read, you read more carefully.

### 3. Granularity is decided at runtime

The orchestrator doesn't always spawn sub-agents. For a simple task ("fix this typo"), it acts directly. For a complex task ("analyze this codebase"), it might spawn four Explore agents in parallel to scan different aspects, read their results, then spawn two more to dig deeper into what they found.

This is the key difference from pipeline architectures. A pipeline commits to a fixed decomposition upfront. The orchestrator decomposes adaptively based on what it learns. First-round agents might reveal that the codebase is a monorepo with three services -- now the orchestrator knows to spawn service-specific agents for the second round. That decomposition couldn't have been predicted from the initial prompt.

### 4. Sub-agents can be resumed

If the orchestrator needs follow-up work from a previous sub-agent, it can resume it with its full prior context preserved. This avoids re-doing discovery work. The sub-agent picks up where it left off.

This is cheaper and more accurate than spawning a new agent with a summary of what the previous one found. Context loss is the enemy of multi-agent systems -- every time you summarize, you lose signal. Resumption preserves the full chain of reasoning.

### 5. Foreground vs background is a scheduling decision

Sub-agents can run in the foreground (orchestrator blocks until they return) or background (orchestrator continues with other work). Foreground when you need results before proceeding. Background when the work is independent.

This is the difference between `await` and "fire and forget with notification." The orchestrator doesn't poll background agents -- it gets notified when they complete. Simple, but it means the orchestrator is never blocked waiting for work it doesn't immediately need.

## Why Not a Framework?

The pattern is minimal enough to implement without a framework. The pieces are:

1. **Prompt expansion layer** -- transforms user intent into structured instructions. This can be as simple as a lookup table of slash commands to prompt templates.

2. **Orchestrator** -- holds conversation state, reads expanded prompts, decides what to delegate. This is your main agent loop.

3. **Typed workers** -- agent instances with constrained tool access. Each type is just a configuration: which tools it can use, what system prompt it gets. The worker itself is the same LLM.

4. **Spawn/collect interface** -- send a prompt to a worker, get a result back. Not a framework. A function.

```python
result = await spawn_agent(
    type="explore",
    prompt="Find all API route handlers and list their HTTP methods and paths",
)
# result is a string. That's it.
```

The orchestrator's system prompt, the worker type definitions, and the spawn function. That's the entire architecture. Everything else -- what to delegate, when to parallelize, how many rounds of agents to run -- emerges from the orchestrator's reasoning about the specific task.

## The Insight

The thing that surprised me: the sub-agents aren't the interesting part. They're just scoped LLM calls. The interesting part is that the **orchestrator decides everything at runtime**. There's no workflow engine, no state machine, no pre-defined graph. The orchestrator reads the situation and improvises.

This works because LLMs are good at exactly this kind of judgment: "given this task and what I know so far, what's the most useful thing to do next?" You don't need to encode that logic in a DAG. You describe the goal, give the orchestrator tools (including the tool to spawn other agents), and let it figure out the decomposition.

The best multi-agent architecture is barely an architecture at all. It's an agent with the ability to delegate.
