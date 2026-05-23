# web_v1

`WebBase` is the FastAPI base class web harnesses extend. The only current subclass is `prokaryotes/harness_v1/web.py`.

## Module layout

- `__init__.py` — `WebBase`, composed from `HarnessBase`, `AuthHandler`, and `CompactionStatusHandler`. Owns the FastAPI app, session middleware, auth routes, static mount, and Postgres lifecycle. Re-exports the shared context-lifecycle surface from `prokaryotes/context_v1/` and `get_postgres_pool` from `prokaryotes/utils_v1/db_utils.py`.
- `auth.py` — `AuthHandler` ABC: login/register/logout/root routes plus the session-gated `/conversation` handler.
- `compaction.py` — `CompactionStatusHandler`: the `/compaction-status` polling endpoint used by the browser UI.

## Shared context lifecycle

The transport-agnostic conversation reconciliation and compaction machinery lives in `prokaryotes/context_v1/`:

- `context_v1/conversation_sync.py` — `ConversationSyncer` ABC: `sync_conversation()` reconciles the active `Conversation` snapshot via Redis fast path → exact ES load → ancestor-chain rebuild. Also assigns `source_id`s to new client messages and detects unacknowledged bot messages for resync handshakes.
- `context_v1/compaction.py` — `ConversationCompactor(ConversationSyncer)`: `_compact_conversation()` (CAS swap into a child snapshot) plus the search-write retry helper.

Connection factories live in `prokaryotes/utils_v1/db_utils.py`: `get_postgres_pool()` and `get_redis_client()`.

## Subclass contract

- Call `super().init()` before adding your own routes or initializing your LLM client — `self.app` does not exist until then.
- Call `await super().on_stop()` before closing provider-specific clients — the base teardown drains background tasks and closes the search and Redis clients.
- Pass a mutable `pending_compaction` list into `stream_and_finalize()` to participate in compaction; the compaction closure itself is built internally via `HarnessBase._build_compact_fn`. Never call `finalize_turn()` directly: `stream_and_finalize()` wraps it to coordinate with the Redis compaction lock, and bypassing the wrapper causes a double-finalize race.

## Related

- [project/features/conversation/README.md](../../project/features/conversation/README.md) — the conversation model, reconciliation, and the web wire protocol.
- [project/features/compaction/README.md](../../project/features/compaction/README.md) — the snapshot-chain compaction lifecycle.
- [prokaryotes/harness_v1/web.py](../harness_v1/web.py) — the only `WebBase` subclass; the chat-route logic, instruction assembly, and tool dispatch live there.
