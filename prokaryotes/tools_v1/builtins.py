from openai.types.responses import WebSearchToolParam
from openai.types.responses.web_search_tool_param import Filters

web_search_tool_param = WebSearchToolParam(
    type="web_search",
    filters=Filters(
        allowed_domains=[
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
    )
)
