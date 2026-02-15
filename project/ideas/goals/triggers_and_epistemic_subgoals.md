# Triggers And Epistemic Subgoals

## Core idea

Durable goals should not directly encode every execution step. Triggers bridge stable commitments and concrete follow-through.

## Reframed for agent-internal goals

A trigger is a cue-response object:

- if a condition appears, surface a reminder
- if uncertainty becomes blocking, gather evidence
- if a failure repeats, switch into repair mode

This is also a clean home for temporary epistemic subgoals. Instead of merely marking a goal as stale, the system can generate short-lived follow-through such as:

- inspect the latest tool error
- re-open the relevant file
- verify whether an assumption is still true
- ask the user for a missing datum

## Why reusable

This keeps the durable layer calm while still allowing the agent to respond to failure, drift, and fresh opportunities with concrete behavior.

## Research roots

- [Gollwitzer, P. M. (1999). *Implementation Intentions: Strong Effects of Simple Plans*.](https://doi.org/10.1037/0003-066X.54.7.493) Retained here as the direct inspiration for cue-response triggers.
- [Parr, T., & Friston, K. J. (2017). *Uncertainty, epistemics and active inference*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5721148/) Retained here for the epistemic-subgoal side: some triggers should restore observability before pushing pragmatic action.
- [Orkin, J. (2005). *Agent Architecture Considerations for Real-Time Planning in Games*.](https://ocs.aaai.org/Library/AIIDE/2005/aiide05-018.php) Retained here for the separation between durable objectives and the lighter follow-through machinery used to pursue them.
