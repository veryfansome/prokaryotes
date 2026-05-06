# File Tool Overlay

Intended end state for the `file_tool` feature. See [the design doc](../README.md) for context.

## Layout

```
overlay/
  prokaryotes/
    api_v1/models.py                # adds prokaryotes_annotations; removes text_preamble
    anthropic_v1/__init__.py        # drops text_preamble assignment in stream_turn
    anthropic_v1/web_harness.py     # registers FileTool, calls reconcile_tracked_files
    openai_v1/__init__.py           # drops tool_preamble accumulator + text_preamble assignment
    openai_v1/web_harness.py        # registers FileTool, calls reconcile_tracked_files
    tools_v1/file_tool.py           # FileTool + reconcile_tracked_files + helpers
    tools_v1/README.md              # documents FileTool
    web_v1/__init__.py              # _lift_active_live_windows; uses it in _compact_partition
  scripts/
    static/ui.js                    # adds formatFileToolCallMarkdown + dispatch
  tests/
    test_api_v1_models_annotations.py
    test_compaction_file_tool_lift.py
    test_file_tool.py
    file_tool_ui.test.js
```

## Verification commands

Lint the proposed Python files:

    uv run ruff check \
      project/wip/file-tool-tracked-context/overlay/prokaryotes \
      project/wip/file-tool-tracked-context/overlay/tests

Run the overlay Python tests (overlay package takes precedence; unchanged modules fall back to the real repo package). The overlay `tests/` directory has no `__init__.py` so each test module is self-contained and does not collide with the real `tests` package:

    PYTHONPATH=project/wip/file-tool-tracked-context/overlay:. \
      uv run --extra test pytest project/wip/file-tool-tracked-context/overlay/tests -q

Run the overlay JS tests:

    npx vitest run --root project/wip/file-tool-tracked-context/overlay

## Diffing against the real tree

```
diff -ruN prokaryotes/ project/wip/file-tool-tracked-context/overlay/prokaryotes/
diff -ruN scripts/ project/wip/file-tool-tracked-context/overlay/scripts/
```
