# Autonomy Budget

## Core idea

A goal should not imply unlimited initiative. The system needs an explicit cap on how far it may go before rechecking with the user or another controller.

## Reframed for agent-internal goals

For agent-self-goals, autonomy budget is a control-depth limiter. It answers questions like:

- may the agent only notice the issue?
- may it propose a next step?
- may it take a reversible action?
- may it execute a full plan through tools?

This budget can be attached to a standing drive, a situational goal, or an execution context.

## Why reusable

This gives the architecture a clean way to separate "the agent cares about this" from "the agent is allowed to act on this now."

## Research roots

This note is a design extension, but it inherits two important distinctions from the cited sources below:

- [Rao, A. S., & Georgeff, M. P. (1995). *BDI Agents: From Theory to Practice*.](https://aaai.org/papers/icmas95-042-bdi-agents-from-theory-to-practice/) Retained here for the separation between durable motivational state and the narrower set of commitments currently adopted for action.
- [Orkin, J. (2005). *Agent Architecture Considerations for Real-Time Planning in Games*.](https://ocs.aaai.org/Library/AIIDE/2005/aiide05-018.php) Retained here for the architectural separation between durable objectives and the concrete action machinery used to satisfy them.
