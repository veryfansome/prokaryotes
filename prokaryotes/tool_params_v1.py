from datetime import datetime, timedelta
from openai.types.responses import FunctionToolParam, WebSearchToolParam

# TODO: how do we do prompt evals?

save_user_context_tool_param = FunctionToolParam(
    type="function",
    name="save_user_context",
    description=(
        "Save new information about the user. This includes anything about the user or their personal life"
        ", including: family, friends, colleagues, past events, opinions and preferences, hobbies, goals"
        ", projects, and more."
        " Call this function when the user volunteers new information that cannot be found elsewhere."
    ),
    parameters={
        "type": "object",
        "properties": {
            "context_summary": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "A flat list of atomic, independent facts. Use simple language"
                    " that clearly articulates what information to save."
                ),
            },
        },
        "additionalProperties": False,
        "required": ["context_summary"],
    },
    strict=True,
)

search_email_tool_param = FunctionToolParam(
    type="function",
    name="search_email",
    description="Search the user's email using a criteria.",
    parameters={
        "type": "object",
        "properties": {
            "search_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "A flat list of IMAP search tokens based on RFC 3501 (IMAP4rev1) and RFC 4731 (ESEARCH)"
                    " for the Python imapclient library."
                    f' Example: ["FROM", "John Smith", "SINCE", "{(datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")}"]'
                ),
            },
        },
        "additionalProperties": False,
        "required": ["search_criteria"],
    },
    strict=True,
)

web_search_tool_param = WebSearchToolParam(
    type="web_search",
    filters={
        "allowed_domains": [
            "en.wikipedia.org"
        ]
    }
)
