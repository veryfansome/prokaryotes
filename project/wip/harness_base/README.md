# HarnessBase Extraction

## Goals

Pull the transport-agnostic conversation lifecycle (Redis + Elasticsearch + compaction + background tasks) out of `WebBase` into a new base class so non-FastAPI harnesses can reuse it without inheriting an HTTP server they don't need.

This is a precondition for the Slack harness ([slack_harness/](../slack_harness/README.md)) and any future worker harness (queue consumers, MCP servers, etc.).

## Observed Repository Context

- `prokaryotes/web_v1/__init__.py` defines `WebBase(AuthHandler, PartitionCompactor)`. It couples four distinct concerns:
  - **Partition lifecycle**: Redis client, Search client, Graph client, `background_tasks`, `finalize`, `stream_and_finalize`, `on_start`, `on_stop`
  - **FastAPI app**: `self.app`, route registration in `init()`, the `lifespan` context manager
  - **Postgres**: `_postgres_pool` field and `postgres_pool` property, used by `AuthHandler` for `chat_user` lookups
  - **Auth + UI**: `AuthHandler` mixin (login/register/logout/conversation/root); `static_dir`, `html_dir`, session middleware, static mount
- `prokaryotes/web_v1/partition_sync.py` and `prokaryotes/web_v1/compaction.py` define `PartitionSyncer` and `PartitionCompactor`. Both are FastAPI-agnostic mixins that depend on Redis + Search but not on Postgres or HTTP. They live under `web_v1/` for historical reasons (added alongside `WebBase`).
- `prokaryotes/harness_v1/__init__.py` currently contains only `build_llm_client`. `harness_v1/script.py` and `harness_v1/eval.py` are standalone harnesses that don't use Redis or ES, so they don't fit this extraction and aren't affected.

## Design

### Layout

```
prokaryotes/
  harness_v1/
    base.py              # new: HarnessBase
  web_v1/
    __init__.py          # WebBase(HarnessBase, AuthHandler) — refactored
    compaction.py        # PartitionCompactor (unchanged; see Open Questions)
    partition_sync.py    # PartitionSyncer (unchanged; see Open Questions)
```

### `HarnessBase`

```python
# prokaryotes/harness_v1/base.py (new)
class HarnessBase(PartitionCompactor):
    """Redis + Search + compaction lifecycle. No transport."""

    def __init__(self):
        self.background_tasks: set[asyncio.Task] = set()
        self._conversation_cache_ex = int(os.getenv("CONVERSATION_CACHE_EXPIRY_SECONDS", 60 * 60 * 24 * 7))
        self.graph_client = GraphClient()
        self._redis_client: Redis | None = None
        self._search_client = SearchClient()

    async def on_start(self):
        self._redis_client = get_redis_client()
        self._search_client.init_client()

    async def on_stop(self):
        await self._search_client.close()
        if self._redis_client is not None:
            await self._redis_client.aclose()

    # background_and_forget, finalize, stream_and_finalize,
    # redis_client / search_client / conversation_cache_ex properties —
    # moved verbatim from WebBase.
```

`HarnessBase` inherits `PartitionCompactor`, which itself inherits `PartitionSyncer`. Subclasses get the full partition-reconciliation surface for free.

### `WebBase` refactor

```python
# prokaryotes/web_v1/__init__.py (refactored)
class WebBase(HarnessBase, AuthHandler):
    """HarnessBase + FastAPI app + Postgres pool + session middleware + auth + static UI."""

    def __init__(self, static_dir: str):
        super().__init__()
        self.app: FastAPI | None = None
        self._postgres_pool: Pool | None = None
        self.static_dir = Path(static_dir)
        self._html_dir = self.static_dir.parent / "html"

    def init(self):
        # build FastAPI app with SessionMiddleware
        # register /health, /compaction-status, /, /login, /register, /logout, /conversation, /static
        # construct lifespan (see below)
        ...

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        self._postgres_pool = await get_postgres_pool()
        await self.on_start()                # HarnessBase
        try:
            yield
        finally:
            # drain background tasks (existing behavior, kept here)
            await self.on_stop()             # HarnessBase
            await self._postgres_pool.close()
```

The existing `lifespan` already does roughly this; the only change is that `on_start` / `on_stop` now live one inheritance layer down.

### What moves where

| Member | Now on |
|---|---|
| `self.app`, `init()` route registration, `lifespan` | `WebBase` (unchanged) |
| `_postgres_pool`, `postgres_pool` property | `WebBase` (unchanged) |
| `static_dir`, `_html_dir`, `html_dir` property | `WebBase` (unchanged) |
| `AuthHandler` mixin, `SessionMiddleware`, static mount | `WebBase` (unchanged) |
| `_redis_client`, `redis_client` property | `HarnessBase` |
| `_search_client`, `search_client` property | `HarnessBase` |
| `graph_client` | `HarnessBase` |
| `background_tasks`, `background_and_forget` | `HarnessBase` |
| `_conversation_cache_ex`, `conversation_cache_ex` property | `HarnessBase` |
| `finalize`, `stream_and_finalize` | `HarnessBase` |
| `on_start`, `on_stop` | `HarnessBase` (`WebBase.lifespan` wraps them with Postgres) |
| `get_health` | `WebBase` (it's a route handler) |
| `get_compaction_status` | Already on `PartitionCompactor`; route registration stays in `WebBase.init()` |

### `WebHarness` impact

None. `WebHarness(WebBase)` continues to call `super().init()`, register `/chat`, and behave identically. The split is invisible to existing callers and to the chat UI.

## Validation

The refactor is mechanical and behavior-preserving:

1. Run the existing unit suite — `tests/unit_tests/test_web_v1*.py`, `tests/unit_tests/test_compaction_*.py`, `tests/unit_tests/test_api_v1_models.py`, `tests/unit_tests/test_anthropic_v1.py`, `tests/unit_tests/test_openai_v1.py` must pass unchanged.
2. Run the Tier B web flow — `tests/integration_tests/tier_b/test_chat_flow.py` and `tests/integration_tests/tier_b/test_file_tool_flow.py` must pass unchanged.
3. Confirm `WebBase` and `WebHarness` public interfaces are unchanged from external callers.

No new tests are needed for `HarnessBase` itself — it's a relocation, and the existing tests exercising these behaviors through `WebBase` continue to do so.

## Open Questions

1. **Should `PartitionSyncer` and `PartitionCompactor` move out of `web_v1/`?** After this refactor, `HarnessBase` (in `harness_v1/`) imports `PartitionCompactor` from `prokaryotes.web_v1.compaction`. The dependency direction is fine (no cycle) but the names mislead — `web_v1/` would house things that aren't web-specific. Candidate destinations: `prokaryotes/harness_v1/partition_sync.py` + `harness_v1/compaction.py`, or a neutral `prokaryotes/partition_v1/`. Not blocking the refactor; can land afterwards if at all.

## Relevant Code Files

| File | Role |
|---|---|
| `prokaryotes/harness_v1/base.py` (new) | `HarnessBase` |
| `prokaryotes/web_v1/__init__.py` (refactored) | `WebBase(HarnessBase, AuthHandler)` — interface unchanged |
