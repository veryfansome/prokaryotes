# Intrusiveness And Dormancy

## Core idea

"How present should this goal be right now?" should be computed, not stored as a manual status flag.

## Reframed for agent-internal goals

For an agent with many internal concerns, dormancy is prompt hygiene and control hygiene. A goal should go quiet when:

- pressure is low
- resonance is low
- evidence is fresh
- confidence is high
- no trigger is pulling it forward

Intrusiveness is the opposite side of the same control surface: how much prompt space, agenda space, or controller attention a goal deserves right now.

## Why reusable

This keeps the agent from carrying every good idea into every turn. Internal goals can remain durable without becoming constantly visible.

## Research roots

This note is largely a design synthesis of control ideas. The closest research roots are:

- [Carver, C. S., & Scheier, M. F. (1982). *Control theory: A useful conceptual framework for personality-social, clinical, and health psychology*.](https://doi.org/10.1037/0033-2909.92.1.111) Retained here because dormancy depends in part on low discrepancy and low corrective pressure.
- [Hayes-Roth, B. (1985). *A blackboard architecture for control*.](https://doi.org/10.1016/0004-3702(85)90063-3) Retained here because intrusiveness is fundamentally about what earns surface area on a shared agenda at a given moment.
