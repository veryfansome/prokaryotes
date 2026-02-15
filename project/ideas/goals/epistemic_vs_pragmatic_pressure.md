# Epistemic Vs Pragmatic Pressure

## Core idea

A goal may require action to change the world, or action to understand the world better. Those are different pressures.

## Reframed for agent-internal goals

For agent-self-goals, this distinction is especially important.

- pragmatic pressure: do the thing
- epistemic pressure: inspect, measure, verify, or disambiguate first

Examples:

- If the agent is unsure which file to edit, epistemic pressure should dominate before code changes begin.
- If a tool failed for unclear reasons, the next step may be diagnosis rather than another blind retry.
- If context is stale, the right move may be to recover state before giving advice.

## Why reusable

This gives the agent a clean way to prefer "know better" before "act harder" when uncertainty is the real blocker.

## Research roots

- [Parr, T., & Friston, K. J. (2017). *Uncertainty, epistemics and active inference*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5721148/) Retained here as the clearest source for separating uncertainty-reduction work from world-changing work.
- [Carver, C. S., & Scheier, M. F. (1982). *Control theory: A useful conceptual framework for personality-social, clinical, and health psychology*.](https://doi.org/10.1037/0033-2909.92.1.111) Retained here for the pragmatic side of the split: pressure arises from discrepancy between current and desired state.
