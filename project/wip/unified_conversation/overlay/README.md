# Unified Conversation — Overlay

Intended end state for the migration described in [../README.md](../README.md). Single-commit (the app is not in production). Each file in this tree replaces or adds to the same path under the repo root.

## Fall-through mechanism

Overlay packages that partially override the real package use a `__path__` extension idiom in their `__init__.py`:

```python
import pathlib
import prokaryotes

_HERE = pathlib.Path(__file__).resolve().parent
for _parent_path in prokaryotes.__path__:
    _candidate = pathlib.Path(_parent_path).resolve() / "<package>"
    if _candidate != _HERE and _candidate.is_dir() and str(_candidate) not in __path__:
        __path__.append(str(_candidate))
```

This is needed wherever the overlay overrides `__init__.py` but does *not* override every sibling submodule — `search_v1` (keeps upstream `topics.py`), `tools_v1`, `tools_v1/file_tool` (keep upstream `paths.py`, `reads.py`, `rendering.py`, `validation.py`), `harness_v1`, and `web_v1`. The top-level `tests/conftest.py` prepends `overlay/prokaryotes` to `prokaryotes.__path__`; subpackages opt in to fall-through individually.

## Verification Commands

Run Ruff against the proposed Python files:

    uv run ruff check \
      project/wip/unified_conversation/overlay/prokaryotes \
      project/wip/unified_conversation/overlay/tests

Run the overlay Python tests:

    PYTHONPATH=project/wip/unified_conversation/overlay:. \
      uv run --extra test pytest project/wip/unified_conversation/overlay/tests -q

Run the overlay JS tests:

    npx vitest run --root project/wip/unified_conversation/overlay

## Diff anchor

Diff the overlay against the live tree with:

    diff -ruN prokaryotes/ project/wip/unified_conversation/overlay/prokaryotes/

## Web client integration note

`scripts/static/conversation_client.js` is a new pure module (no DOM, no fetch) carrying the new client-side primitives: `applyHandshake`, `applyBotMessage`, `relabelSnapshotUuid`, `applyResyncHandshake`, `buildRequestMessages`. These are unit-tested under `tests/ui_tests/conversation_client.test.js`.

Wiring these into the existing `scripts/static/ui.js` is straightforward:

- **`sendMessage` → POST `/chat`**: build the request body's `messages` array via `buildRequestMessages(messageTree, activePath)`. Track the client-side `sentClientIds` in submission order.
- **Stream's first event (`handshake`)**: call `applyHandshake(messageTree, sentClientIds, event)` to stamp `source_id` + `snapshot_uuid` on the submitted user nodes. If the event carries `unacknowledged_bot_messages`, route to `applyResyncHandshake` with the compose mode (`"send-from-leaf"` if `editingParentId === null` at send time; `"edit"` or `"regenerate"` otherwise), then close the stream — auto-retry on send-from-leaf, restore draft on edit/regenerate.
- **Stream's last event (`bot_message`)**: call `applyBotMessage(messageTree, {parentNodeId, fullResponse, snapshotUuid: lastHandshakeSnapshotUuid, sourceId: event.bot_message.source_id, createNodeFn})`. The handshake's `snapshot_uuid` is the authoritative branch for both user and assistant nodes — keep it from the same response.
- **Compaction-status poller**: when `{done: true, snapshot_uuid: new}` arrives, call `relabelSnapshotUuid(messageTree, pendingSnapshotUuid, new)`. Clear the indicator. The polling loop is **the only** clearing path; the legacy side-channel clear on subsequent stream handshake is removed (unsafe across branches).
- **Rename throughout**: `partitionUuid` → `snapshotUuid`, `relabelPartitionUuid` → `relabelSnapshotUuid`, `pending_partition_uuid` → `pending_snapshot_uuid`. The compaction-pending indicator becomes branch-scoped (keyed on the `pending_snapshot_uuid` that scheduled the compaction, tracked alongside the active branch).
