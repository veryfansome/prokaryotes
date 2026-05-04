import asyncio
import json
import logging
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field

from prokaryotes.api_v1.models import (
    ContextPartitionItem,
    FunctionToolCallback,
    ToolParameters,
    ToolSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamBuffer:
    max_lines: int
    lines: deque[str] = field(init=False)
    total_lines: int = 0
    truncated: bool = False

    def __post_init__(self):
        self.lines = deque(maxlen=self.max_lines)

    def append_text(self, text: str):
        if not text:
            return
        for line in text.splitlines():
            self.total_lines += 1
            if len(self.lines) == self.max_lines:
                self.truncated = True
            self.lines.append(line)

    def render_lines(self) -> list[str]:
        rendered = list(self.lines)
        if self.truncated:
            omitted = max(0, self.total_lines - len(rendered))
            rendered.append(
                f"--- Truncated to last {len(rendered)} lines; omitted {omitted} earlier lines ---"
            )
        return rendered


@dataclass
class BackgroundJob:
    job_id: str
    command: str
    process: asyncio.subprocess.Process
    started_at: float
    stdout: StreamBuffer
    stderr: StreamBuffer
    stdout_task: asyncio.Task
    stderr_task: asyncio.Task
    wait_task: asyncio.Task
    completed_at: float | None = None
    exit_code: int | None = None

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def status(self) -> str:
        return "completed" if self.exit_code is not None else "running"

    @property
    def runtime_seconds(self) -> float:
        end_time = self.completed_at if self.completed_at is not None else time.time()
        return max(0.0, end_time - self.started_at)


class ShellCommandTool(FunctionToolCallback):
    """Tool to let the model run shell commands"""

    max_output_lines = 400
    foreground_timeout_seconds = 5.0
    completed_job_retention_seconds = 3600.0

    def __init__(
        self,
        foreground_timeout_seconds: float | None = None,
        max_output_lines: int | None = None,
        completed_job_retention_seconds: float | None = None,
    ):
        if foreground_timeout_seconds is not None:
            self.foreground_timeout_seconds = foreground_timeout_seconds
        if max_output_lines is not None:
            self.max_output_lines = max_output_lines
        if completed_job_retention_seconds is not None:
            self.completed_job_retention_seconds = completed_job_retention_seconds
        self._jobs: dict[str, BackgroundJob] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def call(self, arguments: str, call_id: str) -> ContextPartitionItem | None:
        error = ""
        output = ""
        try:
            payload = json.loads(arguments)
            output = await self._dispatch(payload)
        except Exception:
            error = traceback.format_exc()
        if error:
            if output:
                output += "\n\n"
            output += f"An error occurred:\n{error}"
        logger.info(f"{self.__class__.__name__}[{call_id}]:\n{output}")
        return ContextPartitionItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
        )

    async def _dispatch(self, payload: dict) -> str:
        await self._cleanup_jobs()
        if "job_id" in payload:
            return await self._handle_job_action(payload)
        command = payload["command"]
        return await self._run_command(command)

    async def _run_command(self, command: str) -> str:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        started_at = time.time()
        stdout_buffer = StreamBuffer(self.max_output_lines)
        stderr_buffer = StreamBuffer(self.max_output_lines)
        stdout_task = asyncio.create_task(self._pump_stream(process.stdout, stdout_buffer))
        stderr_task = asyncio.create_task(self._pump_stream(process.stderr, stderr_buffer))
        wait_task = asyncio.create_task(process.wait())
        job = BackgroundJob(
            job_id=str(uuid.uuid4()),
            command=command,
            process=process,
            started_at=started_at,
            stdout=stdout_buffer,
            stderr=stderr_buffer,
            stdout_task=stdout_task,
            stderr_task=stderr_task,
            wait_task=wait_task,
        )
        self._jobs[job.job_id] = job
        self._ensure_cleanup_task()

        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=self.foreground_timeout_seconds)
            await self._finalize_job(job)
            return self._render_completed_job(job, header="Completed within foreground timeout.")
        except asyncio.TimeoutError:
            return self._render_running_job(job)

    async def _handle_job_action(self, payload: dict) -> str:
        job_id = payload["job_id"]
        action = payload.get("action", "status")
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job_id: {job_id}")
        if action == "status":
            if job.wait_task.done() and job.exit_code is None:
                await self._finalize_job(job)
            if job.exit_code is None:
                return self._render_running_job(job)
            return self._render_completed_job(job)
        if action == "terminate":
            if job.exit_code is None:
                job.process.terminate()
                await self._finalize_job(job)
            return self._render_completed_job(job, header="Termination requested.")
        raise ValueError(f"Unsupported action: {action}")

    async def _pump_stream(self, stream: asyncio.StreamReader | None, buffer: StreamBuffer):
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            buffer.append_text(chunk.decode(errors="replace"))

    async def _finalize_job(self, job: BackgroundJob):
        if job.exit_code is not None:
            return
        await asyncio.shield(job.wait_task)
        await asyncio.gather(job.stdout_task, job.stderr_task)
        job.exit_code = job.process.returncode
        job.completed_at = time.time()

    def _render_running_job(self, job: BackgroundJob) -> str:
        return "\n".join([
            f"Status: running (foreground timeout after {self.foreground_timeout_seconds:.1f}s)",
            f"Job ID: {job.job_id}",
            f"PID: {job.pid}",
            f"Command: {job.command}",
            f"Runtime seconds: {job.runtime_seconds:.2f}",
            "Use `job_id` with optional `action`=`status` or `terminate` in a later call.",
            "# STDOUT",
            *job.stdout.render_lines(),
            "# STDERR",
            *job.stderr.render_lines(),
        ])

    def _render_completed_job(self, job: BackgroundJob, header: str | None = None) -> str:
        parts = []
        if header:
            parts.append(header)
        parts.extend([
            "Status: completed",
            f"Job ID: {job.job_id}",
            f"PID: {job.pid}",
            f"Command: {job.command}",
            f"Runtime seconds: {job.runtime_seconds:.2f}",
            f"Exit code: {job.exit_code}",
            "# STDOUT",
            *job.stdout.render_lines(),
            "# STDERR",
            *job.stderr.render_lines(),
        ])
        return "\n".join(parts)

    def _ensure_cleanup_task(self):
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        try:
            while self._jobs:
                await asyncio.sleep(60)
                await self._cleanup_jobs()
        except asyncio.CancelledError:
            raise

    async def _cleanup_jobs(self):
        now = time.time()
        removable_job_ids = []
        for job_id, job in list(self._jobs.items()):
            if job.wait_task.done() and job.exit_code is None:
                await self._finalize_job(job)
            if job.completed_at is not None and (
                now - job.completed_at >= self.completed_job_retention_seconds
            ):
                removable_job_ids.append(job_id)
        for job_id in removable_job_ids:
            self._jobs.pop(job_id, None)

    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def system_message_parts(self) -> list[str]:
        lines = [
            f"## Using the `{self.name}` tool",
            (
                "- Don't chain multiple commands with '&&' or ';' unless the intended task can't be accomplished"
                " without doing so. Whenever possible, use a single, focused `command`, with a distinct `reason`"
                " per tool call."
            ),
            "- When reading files, default to previewing the first 200 lines, e.g. `sed -n '1,200p' <path>`.",
            (
                f"- New commands are waited on for up to {self.foreground_timeout_seconds:.1f} seconds. If a command"
                " is still running after that, it continues in the background and the tool returns a `job_id`."
            ),
            "- To check a background command later, call the tool with `job_id` and optional `action` of `status` or `terminate`.",
            f"- Captured stdout and stderr are each truncated to the last {self.max_output_lines} lines.",
        ]
        return lines

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Use the tool to run arbitrary shell commands.",
            parameters=ToolParameters(
                properties={
                    "command": {
                        "type": "string",
                        "description": "A command string to pass to asyncio.create_subprocess_shell().",
                    },
                    "reason": {
                        "type": "string",
                        "description": "A concise reason for the command.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Optional background job identifier returned by an earlier timed-out command.",
                    },
                    "action": {
                        "type": "string",
                        "description": "Optional action for `job_id`: `status` (default) or `terminate`.",
                        "enum": ["status", "terminate"],
                    },
                },
                required=["reason"],
            ),
        )
