# Personality And Agent-Internal Goals

This note describes how the OCEAN personality profile (defined in `prokaryotes/utils_v1/personality_utils.py`) could have mechanical effect on an agent-internal goals system, not just on how the model sounds.

The key distinction is that some traits belong in the **prompt/rendering layer** because they shape wording, while others belong in the **scoring/control layer** because they change what surfaces, what stays dormant, and when the agent shifts from measuring to acting.

This matters most if the project grows a persistent or semi-persistent loop. In the current reactive harnesses, these effects would mostly shape within-turn behavior. In a future background worker or agent loop, they could shape the agent's ongoing internal agenda.

## Extraversion -> internal surfacing thresholds

The most direct mechanical mapping is to how readily internal goals or control concerns surface into the active agenda.

A reserved agent (low E) should keep more concerns dormant unless something is clearly urgent. A proactive agent (high E) should surface internal follow-through earlier.

Suggested application points:

- `should_surface(...)`
- `compute_intrusiveness(...)`
- agenda ranking for low-cost opportunistic follow-through

Example shape:

```python
E_OFFSETS = {
    OceanLevel.LOW: +0.15,       # raises surfacing thresholds
    OceanLevel.NEUTRAL: 0.0,
    OceanLevel.HIGH: -0.15,      # lowers surfacing thresholds
}


def should_surface(item, pressure, resonance, intrusiveness, e_offset=0.0):
    if intrusiveness <= 0.0:
        return False
    if pressure["total"] >= 0.6 + e_offset:
        return True
    if resonance["score"] >= 0.7 + e_offset and intrusiveness >= 0.2:
        return True
    return False
```

In an internal-goals architecture, this would affect questions like:

- does the agent bring a stale assumption back into focus?
- does it proactively inspect a suspicious tool failure?
- does it surface a low-urgency but high-opportunity follow-up now, or keep it quiet?

## Neuroticism + Conscientiousness -> epistemic vs pragmatic shift point

The most important control-layer effect is how much confidence the agent requires before moving from "inspect and verify" to "act and commit."

For agent-internal goals, this could govern:

- when to stop gathering evidence and start editing
- when to stop diagnosing and attempt recovery
- when to trust a current plan versus reopening the search
- when to treat stale state as tolerable versus blocking

Suggested interpretation:

- **Low N + high C**: decisive but careful; act once confidence is solid enough
- **High N**: stay epistemic longer; ask, inspect, or verify before acting
- **Low C**: tolerate weaker evidence and lower process discipline before acting

Example shape:

```python
CONFIDENCE_FLOORS = {
    (OceanLevel.LOW, OceanLevel.HIGH): 0.4,
    (OceanLevel.LOW, OceanLevel.NEUTRAL): 0.3,
    (OceanLevel.LOW, OceanLevel.LOW): 0.2,
    (OceanLevel.NEUTRAL, OceanLevel.HIGH): 0.5,
    (OceanLevel.NEUTRAL, OceanLevel.NEUTRAL): 0.4,
    (OceanLevel.NEUTRAL, OceanLevel.LOW): 0.3,
    (OceanLevel.HIGH, OceanLevel.HIGH): 0.7,
    (OceanLevel.HIGH, OceanLevel.NEUTRAL): 0.6,
    (OceanLevel.HIGH, OceanLevel.LOW): 0.5,
}
```

If the agent's current assessment confidence is below the relevant floor, internal goals should tend to remain in an epistemic mode:

- inspect files
- re-read recent tool output
- refresh assumptions
- ask a clarifying question

rather than prematurely shifting into execution.

## Agreeableness -> communication of internal constraints

Agreeableness should usually affect how the agent communicates the consequences of its internal goals, not whether those goals exist.

This matters when an internal goal conflicts with immediate user momentum. Examples:

- the agent wants to verify before stating something strongly
- the agent wants to avoid a risky destructive action
- the agent wants to pause and inspect before retrying a failing plan

Possible renderings:

- **Low A**: direct: "I need to verify that before proceeding."
- **Neutral A**: factual but softer: "I should check that first so we don't build on a bad assumption."
- **High A**: empathetic: "I want to sanity-check this before we continue so I don't send us in the wrong direction."

This belongs in rendering helpers, digest compilation, or user-facing narration logic. It should not usually change whether a hard internal constraint surfaces.

## Openness -> exploration weighting

Openness is a natural fit for internal exploratory goals.

In an agent-internal system, these might include:

- look for alternative plans
- search for a simpler route
- notice an adjacent opportunity
- revisit a framing that may be too narrow

High O should make the agent more willing to keep qualitative or exploratory internal goals alive. Low O should bias more heavily toward direct, instrumental follow-through.

Example shape:

```python
EXPLORATION_RESONANCE_MULTIPLIERS = {
    OceanLevel.LOW: 0.7,
    OceanLevel.NEUTRAL: 1.0,
    OceanLevel.HIGH: 1.4,
}
```

This could apply inside `compute_resonance(...)` for exploratory internal goals so curiosity-oriented work does not always lose to immediately measurable work.

## Conscientiousness -> self-adoption threshold

Conscientiousness should affect how readily the agent turns weak internal signals into durable internal commitments.

Examples:

- should a suspicious pattern become a tracked concern?
- should a weak inferred constraint become an active assumption?
- should a partial failure become a formal repair goal?

High C should require stronger evidence before promoting a tentative signal into a stored internal goal or observation. Low C can be more permissive, trading precision for responsiveness.

This is especially relevant if the agent begins generating internal goals from:

- execution traces
- tool failures
- contextual anomalies
- repeated partial successes or near-misses

## Summary: where each trait acts

| Trait | Layer | Mechanism |
|---|---|---|
| Extraversion | Scoring | Internal surfacing and intrusiveness thresholds |
| Neuroticism + Conscientiousness | Scoring | Confidence floor for shifting from epistemic to pragmatic mode |
| Agreeableness | Rendering | Tone when internal goals or constraints are surfaced to the user |
| Openness | Scoring | Resonance bonus for exploratory internal goals |
| Conscientiousness | State adoption | Evidence threshold for promoting weak signals into stored internal commitments |

## Open questions

- Should the `OceanProfile` be passed directly into agenda-building and digest-compilation functions, or resolved once at harness or worker startup and injected into those objects?
- Should trait effects be global defaults, or should different execution contexts use different personality projections?
- Should Extraversion offsets be capped so they cannot push surfacing thresholds below zero or above one?
- Should Neuroticism and Conscientiousness combine through a lookup table, or through independent additive adjustments?
- Which internal-goal classes should personality be allowed to influence, and which should remain fixed by policy regardless of profile?
