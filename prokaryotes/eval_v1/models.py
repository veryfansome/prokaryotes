import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    check_output: str = ""
    duration_seconds: float
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    passed: bool
    task_id: str
    think_call_count: int = 0
    tier: int
    tool_call_count: int = 0
    turn_count: int = 0


class EvalRun(BaseModel):
    impl: str
    max_tool_call_rounds: int | None = None
    model: str
    reasoning_effort: str | None = None
    results: list[EvalResult] = []
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    def summary(self) -> str:
        lines = [f"Run {self.run_id} | {self.impl} {self.model}"]
        for tier in sorted({r.tier for r in self.results}):
            tier_results = [r for r in self.results if r.tier == tier]
            passed = sum(r.passed for r in tier_results)
            total = len(tier_results)
            lines.append(f"  Tier {tier}: {passed}/{total} ({passed / total:.0%})")
        passed = sum(r.passed for r in self.results)
        total = len(self.results)
        lines.append(f"  Total:  {passed}/{total} ({self.pass_rate:.0%})")
        return "\n".join(lines)


class EvalTask(BaseModel):
    check_command: str
    description: str
    id: str
    prompt: str
    setup_command: str | None = None
    setup_files: dict[str, str] = {}
    tier: int
    timeout_seconds: int = 180
