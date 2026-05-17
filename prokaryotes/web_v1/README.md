# web_v1

`WebBase` is the FastAPI base class web harnesses extend. The only current subclass is `prokaryotes/harness_v1/web.py`.

## Module layout

- `__init__.py` — `WebBase`, composed from `HarnessBase`, `AuthHandler`, and `CompactionStatusHandler`. Owns the FastAPI app, session middleware, auth routes, static mount, and Postgres lifecycle. Re-exports the shared context-lifecycle surface from `prokaryotes/context_v1/` and `get_postgres_pool` from `prokaryotes/utils_v1/db_utils.py`.
- `auth.py` — `AuthHandler` ABC: login/register/logout/root routes plus the session-gated `/conversation` handler.
- `compaction.py` — `CompactionStatusHandler`: the `/compaction-status` polling endpoint used by the browser UI.

## Shared context lifecycle

The transport-agnostic partition reconciliation and compaction machinery now lives in `prokaryotes/context_v1/`:

- `context_v1/partition_sync.py` — `PartitionSyncer` ABC: `sync_context_partition()` reconciles the active partition via Redis fast path → exact ES load → ancestor-chain rebuild.
- `context_v1/compaction.py` — `PartitionCompactor(PartitionSyncer)`: `_compact_partition()` (CAS swap into a child partition) plus the search-write retry helper.

Connection factories live in `prokaryotes/utils_v1/db_utils.py`: `get_postgres_pool()` and `get_redis_client()`.

## Subclass contract

- Call `super().init()` before adding your own routes or initializing your LLM client — `self.app` does not exist until then.
- Call `await super().on_stop()` before closing provider-specific clients — the base teardown drains background tasks and closes the search and Redis clients.
- Pass `compact_fn` and a mutable `pending_compaction` list into `stream_and_finalize()` when you want compaction. Never call `finalize()` directly: `stream_and_finalize()` wraps it to coordinate with the Redis compaction lock, and bypassing the wrapper causes a double-finalize race.

## Related

- [project/features/compaction/README.md](../../project/features/compaction/README.md) — data model, reconciliation flow, and ancestor-summary semantics.
- [prokaryotes/harness_v1/web.py](../harness_v1/web.py) — the only `WebBase` subclass; the chat-route logic, instruction assembly, and tool dispatch live there.
