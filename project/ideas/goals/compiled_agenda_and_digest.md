# Compiled Agenda And Digest

## Core idea

Prompt injection should happen through a compiled control surface, not by dumping raw goal records into the model context.

## Reframed for agent-internal goals

The reusable pattern is:

- keep canonical state elsewhere
- compile active concerns into an agenda
- render a bounded digest for the current execution context

For internal goals, the agenda might include:

- hard constraints
- open epistemic gaps
- active follow-through
- current focus items
- recently resolved items worth brief continuity

## Why reusable

This keeps the prompt small, debuggable, and situational. It also creates a clean seam between storage, control logic, and model-facing context.

## Research roots

- [Hayes-Roth, B. (1985). *A blackboard architecture for control*.](https://doi.org/10.1016/0004-3702(85)90063-3) Retained here for the agenda or blackboard framing: multiple signals are compiled into a shared control surface before final action selection.

The digest aspect itself is an architectural pattern rather than a direct lift from a cited paper.
