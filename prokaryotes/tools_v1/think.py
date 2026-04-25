from prokaryotes.api_v1.models import (
    ContextPartitionItem,
    FunctionToolCallback,
    ToolParameters,
    ToolSpec,
)


class ThinkTool(FunctionToolCallback):
    """Tool to give the model a scratchpad for structured reasoning between tool calls."""

    async def call(self, arguments: str, call_id: str) -> ContextPartitionItem | None:
        return ContextPartitionItem(
            call_id=call_id,
            output="ok",
            type="function_call_output",
        )

    @property
    def name(self) -> str:
        return "think"

    @property
    def system_message_parts(self) -> list[str]:
        lines = [
            f"## Using the `{self.name}` tool",
            (
                "- Use the think tool as a scratchpad when the task genuinely requires it — for example:"
            ),
            "  - Analysing tool outputs in a long chain of dependent tool calls",
            "  - Navigating policy-heavy or constraint-laden requirements",
            "  - Making sequential decisions where each step builds on prior results and mistakes are costly",
            "- Do not use it for straightforward instruction following or simple single-step tasks.",
        ]
        return lines

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Use the tool to think about something. It will not obtain new information or "
                "persist any data, but just append the thought to a log. Use it when complex "
                "reasoning or some cache memory is needed."
            ),
            parameters=ToolParameters(
                properties={
                    "thought": {
                        "type": "string",
                        "description": "A thought to think about."
                    }
                },
                required=["thought"],
            ),
        )
