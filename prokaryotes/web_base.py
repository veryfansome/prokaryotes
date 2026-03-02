from abc import ABC, abstractmethod
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path

class WebBase(ABC):
    @classmethod
    async def health(cls):
        """Health check."""
        return {"status": "ok"}

class ProkaryotesBase(WebBase):
    @abstractmethod
    def get_static_dir(self):
        raise NotImplementedError(f"Method 'static_dir' not implemented for {self.__class__.__name__}")

    async def logo(self):
        logo_png_path = Path(self.get_static_dir()) / "logo.png"
        if logo_png_path.exists():
            return FileResponse(
                media_type="image/png",
                path=logo_png_path,
            )
        raise HTTPException(status_code=404, detail="Not found")

    async def root(self):
        """Serve the chat UI."""
        ui_html_path = Path(self.get_static_dir()) / "ui.html"
        with open(ui_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
