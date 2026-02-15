# Standing Drives

## Core idea

Some goals should not be ordinary rows in a goals table. They are always-on behavioral drives that shape the agent globally.

## Reframed for agent-internal goals

In an agent-self-goals design, this concept should move from the margins to the center. Examples:

- preserve user trust
- maintain safety and truthfulness boundaries
- preserve context coherence
- reduce critical uncertainty before overcommitting
- maintain active commitments once made
- respect time, tool, and autonomy budgets

These drives are not "owned" by the user in the same way a user goal is. They are part of the agent's operating constitution.

## Why reusable

This gives the agent a stable backbone. Situational goals can come and go, but standing drives remain the persistent source of control pressure and conflict resolution.

## Research roots

This note is mostly an architectural synthesis, but the closest research roots are:

- [Rao, A. S., & Georgeff, M. P. (1995). *BDI Agents: From Theory to Practice*.](https://aaai.org/papers/icmas95-042-bdi-agents-from-theory-to-practice/) Retained here for the distinction between durable motivational structure and transient execution commitments.
- [Hayes-Roth, B. (1985). *A blackboard architecture for control*.](https://doi.org/10.1016/0004-3702(85)90063-3) Retained here for the idea that persistent control influences can shape a shared agenda without being the same thing as immediate actions.
