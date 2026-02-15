# Observations And Events

## Core idea

Do not treat all goal-related data as one undifferentiated log.

## Reframed for agent-internal goals

The agent should distinguish:

- observations: typed evidence about state
- events: append-only history about what happened

For internal goals, observations might come from:

- tool results
- context inspection
- planner outputs
- execution traces
- user replies that confirm or invalidate an assumption

Events are different:

- goal adopted
- plan failed
- retry scheduled
- trigger fired
- assumption invalidated
- commitment archived

## Why reusable

This makes time-series reasoning possible without forcing every query through a heterogeneous event stream. It also keeps causal history without confusing it for evidence.

## Research roots

- [Carver, C. S., & Scheier, M. F. (1982). *Control theory: A useful conceptual framework for personality-social, clinical, and health psychology*.](https://doi.org/10.1037/0033-2909.92.1.111) Retained here because control depends on preserving observed state as something distinct from the actions and transitions around it.
- [Epstein, D. A., Ping, A., Fogarty, J., & Munson, S. A. (2015). *A Lived Informatics Model of Personal Informatics*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12435389/) Retained here for the distinction between world-state health and tracking-process health, which depends on explicit observational structure.
