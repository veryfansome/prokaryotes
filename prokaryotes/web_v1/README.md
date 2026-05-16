# web_v1

`WebBase` is the FastAPI base class web harnesses extend. The only current subclass is `prokaryotes/harness_v1/web.py`.

## Module layout

- `__init__.py` — `WebBase`, composed from `AuthHandler` and `PartitionCompactor`. Owns the FastAPI app and the Redis, Postgres, Search, and Graph client lifecycle. Provides `stream_and_finalize()` and `finalize()`.
- `auth.py` — `AuthHandler` ABC: login/register/logout/root routes plus the session-gated `/conversation` handler.
- `partition_sync.py` — `PartitionSyncer` ABC: `sync_context_partition()` reconciles the active partition via Redis fast path → exact ES load → ancestor-chain rebuild.
- `compaction.py` — `PartitionCompactor(PartitionSyncer)`: `_compact_partition()` (CAS swap into a child partition) and the `/compaction-status` endpoint.
- `stores.py` — `get_postgres_pool()`, `get_redis_client()`.

## Subclass contract

- Call `super().init()` before adding your own routes or initializing your LLM client — `self.app` does not exist until then.
- Call `await super().on_stop()` before closing provider-specific clients — the base teardown closes the search and graph clients.
- Pass `compact_fn` and a mutable `pending_compaction` list into `stream_and_finalize()` when you want compaction. Never call `finalize()` directly: `stream_and_finalize()` wraps it to coordinate with the Redis compaction lock, and bypassing the wrapper causes a double-finalize race.

## Related

- [project/features/compaction/README.md](../../project/features/compaction/README.md) — data model, reconciliation flow, and ancestor-summary semantics.
- [prokaryotes/harness_v1/web.py](../harness_v1/web.py) — the only `WebBase` subclass; the chat-route logic, instruction assembly, and tool dispatch live there.
