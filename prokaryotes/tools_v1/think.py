from prokaryotes.api_v1.models import (
    ContextPartitionItem,
    FunctionToolCallback,
    LLMClient,
    ToolParameters,
    ToolSpec,
)


class ThinkTool(FunctionToolCallback):
    """Tool to give the model a scratchpad for structured reasoning between tool calls."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

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
            "- Use the think tool to perform focused analysis before acting. It makes targeted LLM calls and returns"
            " structured insights you can act on. Use it when:",
            "  - Choosing between implementation options with meaningful trade-offs.",
            "  - Planning multi-step tasks where sequencing mistakes are costly.",
            "  - Navigating complex constraints or policy-heavy requirements.",
            "  - Synthesizing outputs from multiple prior tool calls.",
            "  - Updating a plan or pivoting from a course of action that has been invalidated by new evidence.",
            "- Populate the tool's parameters carefully.",
            "  - `goal`: state exactly what you need to move forward — do not restate the overall task.",
            "  - `context`: include information needed for a thorough analysis, but be concise and focused.",
            "  - `perspectives`: choose the angles of analysis most likely to provide useful insights.",
            "- Do not use the think tool for straightforward tasks where the next action is clear.",
        ]
        return lines

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Analyze a local decision using the supplied context and perspectives."
                " Make one or more targeted LLM calls to analyze the provided `context` from requested `perspectives`"
                " and perform complex reasoning towards a concrete `goal`."
            ),
            parameters=ToolParameters(
                properties={
                    "goal": {
                        "type": "string",
                        "description": (
                            "The objective of this call - the exact question, decision, or uncertainty that must be"
                            " resolved before acting. This parameter guides the output of this call. Use it"
                            " to specify exactly what you need before you can move forward."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Critical information necessary for the analysis: facts, prior tool outputs, links, text"
                            " or code snippets, or anything else that renders the analysis incomplete if left out"
                            " should be included here. Use markdown bullets for facts, tables for tabular data, and"
                            " code blocks for tool outputs and snippets."
                        ),
                    },
                    "perspectives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Lenses through which to analyze the `context` relative to the `goal`, e.g."
                            " 'assumptions', 'constraints', 'dependencies', 'edge cases', 'implementation options',"
                            " 'order of operations', 'risks', 'trade-offs', etc. Use an empty list when multiple"
                            " perspectives are not needed."
                        ),
                    },
                },
                required=["context", "goal", "perspectives"],
            ),
        )
