import asyncio
import logging
import time
import traceback
from pathlib import Path

from prokaryotes.eval_v1.models import EvalResult, EvalRun, EvalTask
from prokaryotes.utils_v1.llm_utils import ANTHROPIC_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path("/tmp/prokaryotes_eval")


class EvalHarness:
    def __init__(
        self,
        impl: str = "anthropic",
        max_tool_call_rounds: int | None = 20,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.impl = impl
        self.max_tool_call_rounds = max_tool_call_rounds
        self.model = model or (ANTHROPIC_DEFAULT_MODEL if impl == "anthropic" else OPENAI_DEFAULT_MODEL)
        self.reasoning_effort = reasoning_effort

    @classmethod
    def count_turns(cls, items) -> int:
        """Count LLM API calls (turns) from a partition's item list.

        A turn starts on the first function_call after user/function_call_output items, or on any
        assistant message. An assistant message and the tool calls it triggers
        belong to the same turn.
        """
        turns = 0
        in_turn = False
        for item in items:
            if item.role in ("system", "developer", "user"):
                in_turn = False
            elif item.type == "function_call":
                if not in_turn:
                    turns += 1
                    in_turn = True
            elif item.type == "function_call_output":
                in_turn = False
            elif item.role == "assistant" and item.type == "message":
                turns += 1
                in_turn = True
        return turns

    def make_script_harness(self):
        if self.impl == "anthropic":
            from prokaryotes.anthropic_v1.script_harness import ScriptHarness
        else:
            from prokaryotes.openai_v1.script_harness import ScriptHarness
        return ScriptHarness(model=self.model, reasoning_effort=self.reasoning_effort)

    async def run_task(self, task: EvalTask, workspace: Path) -> EvalResult:
        workspace.mkdir(parents=True, exist_ok=True)
        check_output = ""
        error = None
        input_tokens = 0
        output_tokens = 0
        passed = False
        think_call_count = 0
        tool_call_count = 0
        turn_count = 0

        start = time.monotonic()
        try:
            for rel_path, content in task.setup_files.items():
                dest = workspace / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)

            if task.setup_command:
                proc = await asyncio.create_subprocess_shell(
                    task.setup_command,
                    cwd=str(workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

            def on_usage(in_toks: int, out_toks: int) -> None:
                nonlocal input_tokens, output_tokens
                input_tokens += in_toks
                output_tokens += out_toks

            harness = self.make_script_harness()
            try:
                partition = await asyncio.wait_for(
                    harness.run(
                        task=task.prompt,
                        cwd=str(workspace),
                        max_tool_call_rounds=self.max_tool_call_rounds,
                        on_usage=on_usage,
                        verbose=False,
                    ),
                    timeout=task.timeout_seconds,
                )
            finally:
                await harness.close()
            if partition is not None:
                tool_call_count = sum(
                    1 for item in partition.items if item.type == "function_call"
                )
                think_call_count = sum(
                    1 for item in partition.items if item.type == "function_call" and item.name == "think"
                )
                turn_count = self.count_turns(partition.items)
                (workspace / "context_partition.json").write_text(partition.model_dump_json(indent=2))

            proc = await asyncio.create_subprocess_shell(
                task.check_command,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            check_output = (stdout + stderr).decode(errors="replace").strip()
            passed = proc.returncode == 0

        except TimeoutError:
            error = f"Task timed out after {task.timeout_seconds}s"
        except Exception:
            error = traceback.format_exc()

        duration = time.monotonic() - start
        logger.info("Task %s: %s (%.1fs)", task.id, "PASS" if passed else "FAIL", duration)
        return EvalResult(
            check_output=check_output,
            duration_seconds=duration,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            passed=passed,
            task_id=task.id,
            think_call_count=think_call_count,
            tier=task.tier,
            tool_call_count=tool_call_count,
            turn_count=turn_count,
        )

    async def run(
        self,
        tasks: list[EvalTask],
        output_path: Path | None = None,
    ) -> EvalRun:
        run = EvalRun(
            model=self.model,
            impl=self.impl,
            reasoning_effort=self.reasoning_effort,
            max_tool_call_rounds=self.max_tool_call_rounds,
        )

        run_dir = WORKSPACE_ROOT / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Tasks run sequentially: ScriptHarness.run() calls os.chdir() which is process-global.
        for task in tasks:
            result = await self.run_task(task, run_dir / task.id)
            run.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"[{status}] {task.id} ("
                f"{result.duration_seconds:.1f}s"
                f", {result.turn_count} turns"
                f", {result.tool_call_count} tool calls"
                f", {result.think_call_count} think"
                f", {result.input_tokens} in / {result.output_tokens} out tokens"
                f") — {run_dir / task.id}"
            )
            if result.error:
                print(f"       error: {result.error.splitlines()[0]}")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(run.model_dump_json(indent=2))
            logger.info("Results written to %s", output_path)

        return run
