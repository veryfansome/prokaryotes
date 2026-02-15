# Agent-Internal Goals Ideas

This directory holds ideation documents for designing an **agent-internal goals system** that can be used with the harnesses in this project.

The focus here is not primarily "how the agent should regulate itself around a user's durable goals." The focus is instead:

- what goals the agent can meaningfully have for itself
- how those goals should be represented and assessed
- how they might shape behavior inside the web, script, or future worker harnesses
- what reusable control concepts are worth preserving as we work toward a real persistent agentic loop

## What belongs here

Documents in this directory are for:

- architectural ideas
- reusable concepts
- design fragments
- research-grounded notes
- early framing for future internal-goals implementations

These files do **not** imply that the architecture is settled. They are intended to help us think clearly before committing to a concrete substrate.

## Current emphasis

At the moment, this directory mostly contains small concept notes about reusable control and representation ideas for internal-agent use. Examples include:

- standing drives
- durable goals and assessments
- epistemic vs pragmatic pressure
- personality and control-style modulation
- triggers and epistemic subgoals
- compiled agenda and digest
- maintenance vs terminal goals
- temporal goal modes

Where applicable, the concept notes retain citations to papers that originally shaped the ideas.

## Relationship to the harnesses

The intended target is a goals system that can operate within the runtime shape of this repo, including:

- the current web harnesses
- the script harnesses
- future background workers or persistent agent loops

This means the documents here should stay grounded in the practical constraints of the existing project, not drift into a generic agent architecture detached from the codebase.

## Working stance

Treat this directory as a design thinking space for a future subsystem:

- exploratory, but not vague
- grounded in the current repo
- willing to separate reusable ideas from accidental assumptions
- biased toward an eventual agent that can maintain internal state and act over time, not just react inside a single user turn
