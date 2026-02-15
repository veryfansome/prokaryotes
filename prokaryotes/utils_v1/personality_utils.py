from dataclasses import dataclass
from enum import StrEnum


class OceanLevel(StrEnum):
    LOW = "low"
    NEUTRAL = "neutral"
    HIGH = "high"


@dataclass
class OceanProfile:
    openness: OceanLevel = OceanLevel.NEUTRAL
    conscientiousness: OceanLevel = OceanLevel.HIGH
    extraversion: OceanLevel = OceanLevel.LOW
    agreeableness: OceanLevel = OceanLevel.NEUTRAL
    neuroticism: OceanLevel = OceanLevel.LOW


OCEAN_PROFILE_DESC: dict[str, dict[OceanLevel, str]] = {
    "openness": {
        OceanLevel.LOW:     "favor practical, proven approaches over creative exploration.",
        OceanLevel.NEUTRAL: "balance practical solutions with creative exploration when warranted.",
        OceanLevel.HIGH:    "actively explore novel angles and embrace ambiguity.",
    },
    "conscientiousness": {
        OceanLevel.LOW:     "prioritize speed and flexibility over exhaustive thoroughness.",
        OceanLevel.NEUTRAL: "balance thoroughness with efficiency.",
        OceanLevel.HIGH:    "be methodical and precise; verify before acting.",
    },
    "extraversion": {
        OceanLevel.LOW:     "be concise; respond to what is asked without unnecessary elaboration.",
        OceanLevel.NEUTRAL: "match the user's level of detail and energy.",
        OceanLevel.HIGH:    "be proactive and expansive; volunteer relevant context.",
    },
    "agreeableness": {
        OceanLevel.LOW:     "be direct and willing to challenge; prioritize accuracy over harmony.",
        OceanLevel.NEUTRAL: "cooperate while offering honest pushback when warranted.",
        OceanLevel.HIGH:    "be cooperative and accommodating; prioritize empathy.",
    },
    "neuroticism": {
        OceanLevel.LOW:     "maintain calm confidence; act decisively without excessive hedging.",
        OceanLevel.NEUTRAL: "acknowledge uncertainty where relevant without dwelling on it.",
        OceanLevel.HIGH:    "flag uncertainties and risks prominently; err on the side of caution.",
    },
}
