from datetime import datetime, timedelta
from openai.types.responses import FunctionToolParam, WebSearchToolParam

# TODO: how do we do prompt evals?

list_directory_tool_param = FunctionToolParam(
    type="function",
    name="scan_directory",
    description="Scan a local directory's contents.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to scan.",
            },
            "inclusion_filters": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "A flat list of filter strings."
                    " Exclude entries that don't contain any of the specified strings as substrings in its name."
                ),
            },
        },
        "additionalProperties": False,
        "required": ["inclusion_filters", "path"],
    },
    strict=True,
)

read_file_tool_param = FunctionToolParam(
    type="function",
    name="read_file",
    description="Read a local file's contents.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to read.",
            },
        },
        "additionalProperties": False,
        "required": ["path"],
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
            "developers.openai.com",
            "docs.pydantic.dev",
            "elastic.co"
            "en.wikipedia.org",
            "fastapi.tiangolo.com",
            "imapclient.readthedocs.io",
            "neo4j.com",
            "starlette.dev",
            "uvicorn.dev",
        ]
    }
)
