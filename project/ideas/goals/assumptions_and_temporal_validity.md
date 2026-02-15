# Assumptions And Temporal Validity

## Core idea

Assessments should stay connected to the assumptions that supported them, and those assumptions should have time sensitivity.

## Reframed for agent-internal goals

Many internal agent judgments depend on facts that quietly expire:

- the repo state has not changed
- the last tool result is still representative
- the current plan is still valid
- a previously observed constraint still holds

Instead of storing only a conclusion, the agent should retain assumptions such as:

- what it believed
- why it believed it
- when that belief should be reconsidered

## Why reusable

This makes staleness legible. It gives the agent a principled way to downgrade confidence and create new epistemic work when old assumptions stop being safe to carry forward.

## Research roots

- [Doyle, J. (1979). *A Truth Maintenance System*.](https://doi.org/10.1016/0004-3702(79)90008-0) Retained here for the idea that judgments should remain connected to their supporting assumptions and reasons.
- [Parr, T., & Friston, K. J. (2017). *Uncertainty, epistemics and active inference*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5721148/) Retained here for the idea that stale or weak assumptions should generate new uncertainty-reduction work.
