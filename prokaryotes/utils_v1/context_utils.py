from prokaryotes.models_v1 import (
    PersonContext,
    PromptContext,
)

def developer_message_parts(prompt_context: PromptContext, user_context: PersonContext) -> list[str]:
    now = prompt_context.received_at.astimezone(tz=prompt_context.time_zone).strftime('%Y-%m-%d %H:%M')
    message_parts = [
        "## Execution context",
        f"- Time: {now} {prompt_context.time_zone}",
        f"- Environment: Python-{prompt_context.python_version} / {prompt_context.platform_short}",
        f"- Unix user: {prompt_context.unix_usr}",
        f"- Working directory: {prompt_context.cwd}",
        "---",
        "## User info",
    ]
    if prompt_context.latitude and prompt_context.longitude:
        message_parts.append(
            "| Recorded at | Fact |\n"
            "|---|---|"
        )
        message_parts.append(
            f"| - | The user's name is {user_context.name} |"
        )
        message_parts.append(
            f"| - | The user is at"
            f" *lat: {prompt_context.latitude:.4f}, long: {prompt_context.longitude:.4f}* |"
        )
    if user_context.facts:
        for fact_doc in user_context.facts:
            message_parts.append(
                f"| {fact_doc.created_at.astimezone(prompt_context.time_zone).strftime('%Y-%m-%d %H:%M')} |"
                f" {fact_doc.text} |"
            )
    message_parts.append("## Other info")
    message_parts.append(
        "| Recorded at | Fact |\n"
        "|---|---|"
    )
    message_parts.append("| - | The assistant is a Python app |")
    return message_parts
