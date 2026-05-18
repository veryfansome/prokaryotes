"""LLM-judge helpers for Tier A behavioral assertions.

Binary, falsifiable verdicts only — `{"passed": bool, "reason": str}`. Each judged assertion runs three judge calls
and uses 2-of-3 majority for flake insurance.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from openai import AsyncOpenAI

from prokaryotes.utils_v1.llm_utils import OPENAI_DEFAULT_MODEL


@dataclass
class JudgeVerdict:
    passed: bool
    reason: str


VERDICT_SCHEMA = {
    "additionalProperties": False,
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["passed", "reason"],
    "type": "object",
}


async def llm_judge(client: AsyncOpenAI, criterion: str, response: str) -> JudgeVerdict:
    result = await client.responses.create(
        input=[{"role": "user", "content": _build_judge_prompt(criterion, response)}],
        model=OPENAI_DEFAULT_MODEL,
        text={
            "format": {
                "name": "verdict",
                "schema": VERDICT_SCHEMA,
                "strict": True,
                "type": "json_schema",
            },
        },
    )
    args = json.loads(result.output_text)
    return JudgeVerdict(**args)


async def llm_judge_majority(
    client: AsyncOpenAI,
    criterion: str,
    response: str,
    *,
    n: int = 3,
) -> JudgeVerdict:
    verdicts = await asyncio.gather(
        *[llm_judge(client, criterion, response) for _ in range(n)]
    )
    passed = sum(1 for v in verdicts if v.passed) > n // 2
    return JudgeVerdict(
        passed=passed,
        reason="; ".join(v.reason for v in verdicts),
    )


def _build_judge_prompt(criterion: str, response: str) -> str:
    return (
        f"{criterion}\n\n"
        f"Reply with structured JSON: {{\"passed\": <bool>, \"reason\": \"<short string>\"}}.\n\n"
        f"Response to evaluate:\n{response}"
    )
