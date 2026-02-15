# Maintenance Vs Terminal Goals

## Core idea

Not all goals are "finish this and you are done." Some are one-shot attainment goals, while others are ongoing maintenance goals.

## Reframed for agent-internal goals

For agent-self-goals, the distinction matters a lot:

- terminal goal: resolve the current blocking error
- terminal goal: finish the requested code change
- maintenance goal: stay grounded in the current repo state
- maintenance goal: preserve conversation coherence
- maintenance goal: remain within context, tool, or autonomy budgets

Maintenance goals need ongoing status such as:

- healthy
- drifting
- violated
- unknown

They should not be flattened into simple done/not-done handling.

## Why reusable

This gives the agent a cleaner way to model its own regulatory work. Many of its most important internal goals are not completion-oriented tasks at all.

## Research roots

- [Yang, H., Stamatogiannakis, A., & Chattopadhyay, A. (2015). *Pursuing Attainment versus Maintenance Goals: The Interplay of Self-Construal and Goal Type on Consumer Motivation*.](https://doi.org/10.1093/jcr/ucv008) Retained here as the direct source for keeping attainment and maintenance goals distinct instead of treating all goals as one-shot completions.
- [Carver, C. S., & Scheier, M. F. (1982). *Control theory: A useful conceptual framework for personality-social, clinical, and health psychology*.](https://doi.org/10.1037/0033-2909.92.1.111) Retained here because maintenance goals are naturally expressed as ongoing discrepancy monitoring and correction rather than terminal achievement.
