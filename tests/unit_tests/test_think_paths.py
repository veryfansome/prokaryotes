"""ThinkTool `paths` parameter — injects matching active working-file windows into the subprompt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prokaryotes.conversation_v1.models import ProjectedItem, WorkingFileWindow
from prokaryotes.tools_v1.think import ThinkTool


class _FakeLLM:
    def __init__(self):
        self.last_items: list[ProjectedItem] = []
        self.last_instruction: str | None = None

    async def complete(self, *, items, instruction, model, reasoning_effort):
        self.last_items = items
        self.last_instruction = instruction
        return "stub-output"


def _window(window_id: str, path: str, status: str = "live") -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path=path,
        status=status,
        revision="r1",
        rendered_output=f"FILE path={path} revision=r1 status={status}",
        view_start_line=1,
        view_end_line=40,
        requested_end_line=40,
        source_kind="read_lines",
    )


def _args(*, paths: list[str] = ()) -> str:
    return json.dumps(
        {
            "goal": "test",
            "context": "ctx",
            "perspectives": [],
            "paths": list(paths),
        }
    )


@pytest.mark.asyncio
async def test_paths_injects_matching_live_windows(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("...", encoding="utf-8")
    windows = [_window("c-1", str(target))]
    fake = _FakeLLM()
    tool = ThinkTool(
        fake,
        model="m",
        working_file_provider=lambda: windows,
        workspace_root=tmp_path,
    )
    await tool.call(_args(paths=[str(target)]), call_id="t-1")
    body = fake.last_items[0].content or ""
    assert "<active-working-files-" in body
    assert f"FILE path={target}" in body


@pytest.mark.asyncio
async def test_paths_skips_stale_windows(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("...", encoding="utf-8")
    windows = [_window("c-1", str(target), status="stale")]
    fake = _FakeLLM()
    tool = ThinkTool(
        fake,
        model="m",
        working_file_provider=lambda: windows,
        workspace_root=tmp_path,
    )
    await tool.call(_args(paths=[str(target)]), call_id="t-1")
    body = fake.last_items[0].content or ""
    assert "<active-working-files-" not in body


@pytest.mark.asyncio
async def test_empty_paths_omits_block(tmp_path: Path):
    windows = [_window("c-1", "/abs/a.py")]
    fake = _FakeLLM()
    tool = ThinkTool(
        fake,
        model="m",
        working_file_provider=lambda: windows,
        workspace_root=tmp_path,
    )
    await tool.call(_args(paths=[]), call_id="t-1")
    body = fake.last_items[0].content or ""
    assert "<active-working-files-" not in body


@pytest.mark.asyncio
async def test_no_provider_omits_block(tmp_path: Path):
    fake = _FakeLLM()
    tool = ThinkTool(fake, model="m", working_file_provider=None, workspace_root=tmp_path)
    await tool.call(_args(paths=["/abs/a.py"]), call_id="t-1")
    body = fake.last_items[0].content or ""
    assert "<active-working-files-" not in body
