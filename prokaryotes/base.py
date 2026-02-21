from abc import ABC, abstractmethod
from fastapi.responses import HTMLResponse
from pathlib import Path

class ProkaryotesBase(ABC):

    @abstractmethod
    def ui_filename(self):
        raise NotImplementedError(f"Method 'ui_filename' not implemented for {self.__class__.__name__}")

    async def root(self):
        """Serve the chat UI."""
        ui_html_path = Path(self.ui_filename())
        with open(ui_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

    @classmethod
    async def health(cls):
        """Health check."""
        return {"status": "ok"}
