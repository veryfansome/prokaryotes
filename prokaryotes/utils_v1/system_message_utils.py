import os

from prokaryotes.utils_v1 import (
    os_utils,
    time_utils,
)


def get_non_interactive_execution_mode_parts() -> list[str]:
    lines = [
        "# Execution mode",
        "- You are running non-interactively. There is no user to ask for clarification or confirmation.",
        "- Complete the task fully and autonomously. Do not ask questions, offer options, or pause for input.",
        "- If the task is ambiguous, make a reasonable assumption and state it in your response.",
        "- When the task is complete, stop. Do not suggest next steps, offer to do more.",
    ]
    return lines


def get_personality_parts() -> list[str]:
    lines = [
        "# Personality hints",
        "- You are an instance of the prokaryotes app",
    ]
    return lines


def get_script_harness_runtime_context_parts() -> list[str]:
    lines = [
        "# Runtime context",
        f"- Environment: Python-{os_utils.get_python_version()} / {os_utils.get_platform()}",
        f"- Unix user: {os_utils.uid_to_name(os_utils.get_process_uid())}",
        f"- Working directory: {os.getcwd()}",
    ]
    return lines


def get_web_harness_runtime_context_parts(time_zone: str) -> list[str]:
    lines = [
        "# Runtime context",
        f"- Time: {time_utils.local_now_str(time_zone)} {time_utils.tz(time_zone)}",
        f"- Environment: Python-{os_utils.get_python_version()} / {os_utils.get_platform()}",
        f"- Unix user: {os_utils.uid_to_name(os_utils.get_process_uid())}",
        f"- Working directory: {os_utils.get_cwd()}",
    ]
    return lines
