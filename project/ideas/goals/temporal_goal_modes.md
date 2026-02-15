# Temporal Goal Modes

## Core idea

Goals should carry explicit time semantics rather than all being treated as the same kind of object.

## Reframed for agent-internal goals

An internal-goals architecture may need multiple modes, for example:

- `achieve_once`: resolve a concrete blocking problem
- `maintain_range`: keep context usage or error rate within bounds
- `recur_cadence`: re-check a risky assumption every so often
- `budget_cumulative`: stay within a tool-call or time budget
- `trend_direction`: reduce uncertainty or failure frequency over time

The exact mode set can change, but the reusable idea is that temporal shape matters and should be modeled explicitly.

## Why reusable

This lets the agent reason differently about completion, maintenance, cadence, accumulation, and directional improvement instead of forcing them all into one generic lifecycle.

## Research roots

- [Morgan, R., Pulawski, S., Selway, M., Grossmann, G., Mayer, W., & Stumptner, M. (2023). *Modelling temporal goals in runtime goal models*.](https://www.sciencedirect.com/science/article/pii/S0169023X23000654) Retained here as the direct source for treating temporal goal types as first-class instead of reducing everything to binary completion.
- [Yang, H., Stamatogiannakis, A., & Chattopadhyay, A. (2015). *Pursuing Attainment versus Maintenance Goals: The Interplay of Self-Construal and Goal Type on Consumer Motivation*.](https://doi.org/10.1093/jcr/ucv008) Retained here as the adjacent distinction between one-shot attainment and ongoing maintenance.
