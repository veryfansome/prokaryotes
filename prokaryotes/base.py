import os
from abc import ABC
from fastapi.responses import HTMLResponse

class ProkaryotesBase(ABC):

    @classmethod
    def ui_filename(cls) -> str:
        return "ui.html"

    @classmethod
    async def root(cls):
        """Serve the chat UI."""
        ui_html_path = os.path.join("prokaryotes", cls.ui_filename())
        with open(ui_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

    @classmethod
    async def health(cls):
        """Health check."""
        return {"status": "ok"}
