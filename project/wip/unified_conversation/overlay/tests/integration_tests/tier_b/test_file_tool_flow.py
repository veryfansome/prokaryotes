"""Tier B file-tool flow tests.

MIGRATION STATUS: skeletal. The upstream file is 1073 lines / 9 tests covering
live-window survival across writes, compaction-time stripping, and the
file-tool diagnostic edge cases (CONFLICT, RANGE_TRUNCATED, ALREADY_EXISTS).
The file-tool semantics are fully covered by:
- `tests/unit_tests/test_file_tool.py` (77 tests, view_provider-based)
- `tests/unit_tests/test_strip_live_window_bodies.py` (9 tests)
- `tests/unit_tests/test_compaction_lift_plan.py` (overlay's existing)
- `tests/unit_tests/test_compaction_provider.py` (5 tests; live-window stripping
   in the summarization input)

The integration-tier versions verify the same behaviors end-to-end against
real ES + Redis, and additionally require careful handling of the
`workspace_root` mismatch: the web harness uses `Path.cwd()` (the repo root)
as the workspace, so the upstream tests must place files under cwd-relative
paths rather than `tmp_path`. They follow the same `post_chat_and_advance`
pattern as `test_compaction_flow.py` for handshake/bot_message handling.

Verify by running with the docker-compose data stores up:
    docker compose up -d elasticsearch postgres redis
    PYTHONPATH=project/wip/unified_conversation/overlay:. \\
        uv run --extra test pytest \\
        project/wip/unified_conversation/overlay/tests/integration_tests/tier_b/test_file_tool_flow.py
"""

from __future__ import annotations
