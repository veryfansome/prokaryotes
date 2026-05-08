# Integration Tests for Web Harnesses

## Goals

Add a tier of tests that exercise `prokaryotes/anthropic_v1/web_harness.py` and `prokaryotes/openai_v1/web_harness.py` from the client's perspective — POST `/chat`, parse the NDJSON stream, echo `partition_uuid` back, poll `/compaction-status` — with particular focus on the [compaction](../../features/compaction/README.md) flow. Existing unit tests already cover internal contracts of `_compact_partition`, `_rebuild_from_chain`, and provider summary injection with fakes; they do not cross the HTTP boundary, do not exercise the real Redis `WATCH/MULTI/EXEC` swap, and do not exercise the real ES chain walk. This tier closes that gap.

The new tests must be:

- **Runnable on demand** via a single command per tier.
- **Out of CI for now**, gated by pytest markers so the default `uv run pytest` run is unaffected.
- **Layered**, so one tier is fast and deterministic and the other is a small, expensive smoke check.

Two tiers are introduced:

- **Tier B (primary)** — live Redis / Postgres / Elasticsearch from `docker-compose.yml`, stubbed LLM client, in-process FastAPI app via `httpx.AsyncClient(transport=ASGITransport(app=...))`. Deterministic. No API spend. Exercises the real persistence and reconciliation paths.
- **Tier A (smoke)** — live DBs and a real LLM provider. Small number of tests, structural assertions plus two LLM-judge-backed tests for compaction-specific behavioral contracts. Run by hand before a release.

Out of scope:

- JavaScript stream-parser tests (already covered by `tests/ui_tests/ui.test.js`).

---

## Tier B — Live DBs, stubbed LLM

### Architecture

```
pytest test process
  └── httpx.AsyncClient(transport=ASGITransport(app=harness.app))
        └── FastAPI app (in-process)
              ├── Redis client      ──► docker-compose redis    (real)
              ├── asyncpg pool      ──► docker-compose postgres (real)
              ├── SearchClient (ES) ──► docker-compose elasticsearch (real)
              └── llm_client        ──► FakeLLMClient            (in-process)
```

The harness is constructed in-process. `harness.init()` runs synchronously. Lifespan startup/shutdown is driven via `asgi-lifespan` (new test-only dependency) so `on_start` / `on_stop` execute exactly as they would under uvicorn. The `llm_client` attribute is replaced with a fake before the first request.

### Stubbed LLM client

A `FakeAnthropicClient` and `FakeOpenAIClient`, each implementing the harness-contract surface: `stream_turn`, `complete`, `init_client`, `close`, plus the one non-streaming SDK call each provider's `_summarize_and_compact` reaches for (`async_anthropic.messages.create` and `async_openai.responses.create` — see the return-shape requirements below). The fakes do *not* mimic the streaming SDK surfaces (`messages.stream` / OpenAI Responses-API event types); `stream_turn` emits NDJSON directly. The fakes are scriptable: each test provides a `LLMScript` describing the rounds and events the fake should emit.

Because the harness fixture is session-scoped (one harness shared across all tests in a run) but scripts are per-test, the fake must support both per-test injection and per-test reset. The shape:

```python
class FakeAnthropicClient:
    def __init__(self) -> None:
        self._script: LLMScript | None = None
        self._round_cursor = 0

    def set_script(self, script: LLMScript) -> None:
        """Install a fresh script and rewind the round cursor."""
        self._script = script
        self._round_cursor = 0

    def reset(self) -> None:
        """Drop the current script and any partial state. Called between tests."""
        self._script = None
        self._round_cursor = 0
```

`stream_turn` reads `self._script.rounds[self._round_cursor]`, advances the cursor on each round, and raises `AssertionError("FakeAnthropicClient: no script installed")` if `_script` is `None` — that way a test that forgot to call `set_script` fails loudly rather than silently no-op'ing the harness.

A function-scoped autouse fixture lives in `tests/integration_tests/tier_b/conftest.py` — Tier B only, because Tier A's real `AsyncAnthropic` / `AsyncOpenAI` clients have no `reset()` method. The autouse must defend against being run for a test that doesn't request `web_harness`: depending on `web_harness` directly would force its parametrized `request.param` to resolve, which fails for any test that hasn't been decorated with `@pytest.mark.parametrize("web_harness", ..., indirect=True)`. The defensive form (full version inlined in "Test client fixture" below):

```python
@pytest.fixture(autouse=True)
def _reset_fake_llm(request):
    if "web_harness" not in request.fixturenames:
        yield
        return
    harness = request.getfixturevalue("web_harness")
    harness.llm_client.reset()
    yield
```

Every Tier B test currently parametrizes `web_harness` (state assertions need `web_harness.redis_client` / `web_harness.search_client`), so the no-op branch never trips today; the guard exists so a future Tier B test that only uses `authed_client` doesn't surface as a confusing collection-time error.

Tests that need a script call `web_harness.llm_client.set_script(LLMScript(rounds=[...]))` at the start of the body. Tests that don't drive the LLM at all (e.g. the auth-boundary scenario) leave the fake unset and rely on `set_script`'s missing-script assertion as a tripwire.

```python
@dataclass
class LLMRound:
    text_deltas: list[str] = field(default_factory=list)   # streamed chunks
    tool_calls: list[ToolCallSpec] = field(default_factory=list)  # name + arguments
    input_tokens: int = 1000     # used to drive on_usage / context_pct
    output_tokens: int = 200
    stop_reason: str = "end_turn"  # or "tool_use"

@dataclass
class LLMScript:
    rounds: list[LLMRound]
    summary_text: str = "STUB SUMMARY"  # returned by _summarize_and_compact path
```

The fake emits the same NDJSON sequence the real client emits (`progress_message`, `tool_call`, `text_delta`, `context_pct`), driven from the script. For compaction tests, a round can be scripted with `input_tokens` set high enough that `on_usage` trips the `COMPACTION_TOKEN_THRESHOLD_PCT` check.

The fake's `messages.create` / `responses.create` (the non-streaming path used by `_summarize_and_compact`) returns `summary_text`, but the *return shape* must match what the harness reads, not just contain the string:

- **Anthropic `_summarize_and_compact`** (`anthropic_v1/web_harness.py:59-60`) calls `self.llm_client.async_anthropic.messages.create(...)` and reads `response.content[0].text`. The fake's `messages.create` must return an object exposing `.content` as a list whose first element has a `.text` attribute equal to `summary_text`. Use a small dataclass / `SimpleNamespace`, not a bare string.
- **OpenAI `_summarize_and_compact`** (`openai_v1/web_harness.py:61-67`) calls `self.llm_client.async_openai.responses.create(..., stream=False)` and reads `response.output_text`. The fake's `responses.create(stream=False)` must return an object with an `.output_text` attribute equal to `summary_text`.

These shapes apply only to the *summarization* SDK path. The fake's own `complete()` (used by `ThinkTool`) implements the `LLMClient` protocol directly and just returns a `str` — no SDK wrapping needed because nothing unwraps it.

`summary_text` defaults to `"STUB SUMMARY"`. Tests that span multiple compaction generations set distinct strings explicitly so the assertion can discriminate between generations:

```python
script_gen1 = LLMScript(rounds=[...], summary_text="GEN-1")
# run turn → compaction fires
script_gen2 = LLMScript(rounds=[...], summary_text="GEN-2")
# run turn → second compaction fires
assert partition.ancestor_summaries == ["GEN-1", "GEN-2"]
```

Single-generation tests keep the default. There is no fixture-driven auto-uniqueness — per-test isolation already comes from the random `conversation_uuid`.

The fakes substitute at the harness-contract level, not the SDK-event level. Their `stream_turn` produces NDJSON directly rather than simulating Anthropic's text-stream events or OpenAI Responses-API event types (`response.created`, `response.output_item.added`, etc.). The harness's external contract is the NDJSON stream and the partition state; that's the boundary the tests assert against.

To stay aligned with the production clients, the fake's `stream_turn` must replicate every side effect the harness depends on:

- **Call `on_usage(input_tokens, output_tokens)` after each round.** This is what drives `pending_compaction[0] = True` in the harness's closure; without it, no compaction test can trigger.
- **Mutate the passed `context_partition`.** Append `ContextPartitionItem(type="function_call", ...)` for tool calls (with `text_preamble` on the first call of a round when a preamble was scripted), append the `function_call_output` items returned by callbacks, and append a final `ContextPartitionItem(role="assistant", content=...)` at the end of any non-tool round. Tests inspect the persisted partition; this is where that state comes from.
- **Invoke `tool_callbacks[name].call(args, call_id)` for each scripted tool call.** Production clients dispatch via `asyncio.gather`; the fake should do the same so the resulting `function_call_output` items appear in the partition before the next round.
- **Implement `complete()`** on `LLMClient` with the full protocol signature: `async def complete(self, context_partition: ContextPartition, model: str, reasoning_effort: str | None = None) -> str` (`api_v1/models.py:308-314`). `ThinkTool.call` always passes `reasoning_effort` as a keyword (`tools_v1/think.py:100`), so the fake will `TypeError` if it's omitted. Any test that exercises a `think` tool round must produce a deterministic `complete()` response — not just `stream_turn`. The fake can return a hardcoded string (e.g. `"STUB THINK ANALYSIS"`) since the test boundary is the `function_call_output` item appended to the partition, not the analysis content itself.
- **Implement `init_client()` and `close()` as no-ops.** These are called by the harness's `init()` and `on_stop()`.

Concretely, the fake's `stream_turn` is a small driver that walks the script's rounds, yields the appropriate NDJSON events (in the same order the real client emits them — see "Stream order" below), invokes callbacks, and accumulates partition items. Tests that drift from this contract will pass at the HTTP layer while the partition state quietly diverges from production behavior.

### Triggering compaction

The integration tier sets `COMPACTION_TOKEN_THRESHOLD_PCT=1` globally (see "Triggering compaction in Tier A" for the math), so even small scripted token counts can trip the check. We do not also support a "production-default 80%" path: at 80% on a 128k context, a round needs ~102k input tokens to trip the check, which is the wrong cost/throughput point for a test fake whose `input_tokens` are arbitrary anyway. The default `LLMRound.input_tokens=1000` stays safely below 1% of a 128k context (~1,280 tokens), so non-compaction tests do not accidentally trip; compaction tests script a round with `input_tokens` above ~1,280.

The shared-global has an import-binding pitfall: `prokaryotes.utils_v1.llm_utils` resolves the threshold constant once at module import, and both web harnesses pull it into their own module namespace via `from prokaryotes.utils_v1.llm_utils import COMPACTION_TOKEN_THRESHOLD_PCT` at lines 23/24 of their `web_harness.py`. Reloading `llm_utils` after the harness has been imported does *not* update the harness's binding — `from … import …` is a one-time copy. The same property means **per-tier env split is unsafe** when both tiers run in one pytest invocation: the first tier's harness import freezes the constant for the rest of the process, and the second tier silently inherits the wrong value. That's why threshold and recency-tail are set once in the *root* `tests/integration_tests/conftest.py` and shared by both tiers. The integration tier therefore sets these env vars **before any harness module is imported**:

- Set the env var at the top of `tests/integration_tests/conftest.py` (module level, above any `from prokaryotes…` import).
- Keep harness imports inside fixtures, not at test-module top level. Fixtures resolve at test-collection completion, by which time conftest has already adjusted the env.

A second prerequisite for compaction tests that assert `raw_message_start_index` advancement: `_recency_tail_items` retains the last `COMPACTION_RECENCY_TAIL` user/assistant message items (default 6). When the conversation has fewer items than the recency tail, `tail_offset` is 0 and `raw_message_start_index` does not advance even though compaction succeeds. To exercise the advancement, either:

- **Lower the recency tail** via `COMPACTION_RECENCY_TAIL=2` (set in conftest with the same import-ordering caveat — `web_v1/__init__.py:59` imports it via `from`), or
- **Lengthen the conversation** to at least `COMPACTION_RECENCY_TAIL + 1` user/assistant items before triggering compaction.

The forced-compaction scenario (#4 below) takes the lowered-recency-tail path so a single high-token round can demonstrate advancement without filler turns.

### Test client fixture

Three conftests cooperate: a shared root that does env setup and owns the per-provider `live_keys_*` gates plus `judge_client`, a Tier B-specific one that owns the fake-backed harness fixtures plus the autouse fake-reset, and a Tier A-specific one that owns the real-client harness fixtures gated on the matching `live_keys_*` fixture (shown in the "Key gating" subsection of Tier A below).

```python
# tests/integration_tests/conftest.py — root conftest, shared by Tier A and Tier B.
# Module top, before ANY prokaryotes import (see "Triggering compaction" below for why).
import os
from dotenv import load_dotenv
load_dotenv()                                          # POSTGRES_*, REDIS_*, ELASTIC_URI from .env

# `.env` and `.env.example` use Docker-network service names (`prokaryotes-postgres`,
# `prokaryotes-redis`, `http://prokaryotes-elasticsearch:9200`) because the prod app runs
# inside the compose network. Host-side `uv run pytest` cannot resolve those names, so we
# rewrite them to localhost-mapped ports here. Direct assignment overrides any value already
# loaded from `.env`.
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["REDIS_HOST"] = "localhost"
os.environ["ELASTIC_URI"] = "http://localhost:9200"

# Single global value satisfies both tiers — see "Triggering compaction in Tier A" for the
# rationale (Tier B's fake controls scripted input_tokens; Tier A needs `1` to trip in 2–3
# normal turns; per-tier split would silently break a combined run because of import-time
# binding in the harness modules).
os.environ["COMPACTION_TOKEN_THRESHOLD_PCT"] = "1"
os.environ["COMPACTION_RECENCY_TAIL"] = "2"

# Harness imports are deferred into tier-specific conftests' fixtures so the env above is in place first.
```

```python
# tests/integration_tests/tier_b/conftest.py
import secrets
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _web_harness_anthropic():
    from prokaryotes.anthropic_v1.web_harness import WebHarness
    from integration_tests.fakes import FakeAnthropicClient

    harness = WebHarness(static_dir="scripts/static")
    harness.llm_client = FakeAnthropicClient()         # replace BEFORE init()
    harness.init()                                     # calls llm_client.init_client() — now a no-op on the fake
    async with LifespanManager(harness.app):
        yield harness

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _web_harness_openai():
    from prokaryotes.openai_v1.web_harness import WebHarness
    from integration_tests.fakes import FakeOpenAIClient

    harness = WebHarness(static_dir="scripts/static")
    harness.llm_client = FakeOpenAIClient()
    harness.init()
    async with LifespanManager(harness.app):
        yield harness

@pytest.fixture
def web_harness(request):
    """Indirect fixture. Tests parametrize as `("anthropic", "openai")` and request `web_harness`.
    Every Tier B test must parametrize this fixture — the autouse `_reset_fake_llm` below
    depends on it, and tests inspect `web_harness.redis_client` / `web_harness.search_client`
    for state assertions anyway. Auth-boundary tests (scenario 8) parametrize it too even
    though they hit `/chat` unauthenticated; they still need the harness app running."""
    return request.getfixturevalue(f"_web_harness_{request.param}")

@asynccontextmanager
async def _authed_client_ctx(harness):
    """Plain async context manager — *not* a pytest fixture. Pytest fixtures cannot
    legally be invoked as helpers from another fixture body; the per-provider
    fixtures below `async with` this directly so they can yield the wrapped client."""
    transport = ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        password = secrets.token_urlsafe(16)
        await client.post("/register", data={
            "email": f"peter-{uuid4()}@prokaryotes.test",
            "full_name": "Peter Prokaryote",
            "password": password,
            "confirm_password": password,
        })
        yield client

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def authed_client_anthropic(_web_harness_anthropic):
    async with _authed_client_ctx(_web_harness_anthropic) as client:
        yield client

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def authed_client_openai(_web_harness_openai):
    async with _authed_client_ctx(_web_harness_openai) as client:
        yield client

@pytest.fixture
def authed_client(request):
    return request.getfixturevalue(f"authed_client_{request.param}")

@pytest.fixture(autouse=True)
def _reset_fake_llm(request):
    """Reset fake LLM state before each Tier B test. Defensive: if a test happens to omit
    the `web_harness` fixture (none currently do, but the safety check keeps the autouse
    from blowing up on a future test that only uses `authed_client`), the reset becomes
    a no-op rather than a hard collection failure on `request.param`."""
    if "web_harness" not in request.fixturenames:
        yield
        return
    harness = request.getfixturevalue("web_harness")
    harness.llm_client.reset()
    yield
```

Tests parametrize over both providers via `indirect=True`:

```python
@pytest.mark.parametrize("web_harness, authed_client", [
    ("anthropic", "anthropic"),
    ("openai", "openai"),
], indirect=True)
async def test_single_turn_happy_path(web_harness, authed_client):
    ...
```

Provider-specific tests (e.g. tool-call ordering scenarios where the order itself differs) parametrize with a single value and assert against that provider only.

Notes:

- **Replace `llm_client` before `init()`.** `WebHarness.init()` calls `self.llm_client.init_client()` (anthropic_v1/web_harness.py:65, openai_v1/web_harness.py:72), which constructs a real `AsyncAnthropic` / `AsyncOpenAI` and would fail without `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Tier B must not require those keys, so we swap in the fake first.
- **`load_dotenv()` at the top of conftest, then host-side overrides.** Unlike `scripts/web.py`, the test process doesn't load `.env` automatically. Without `load_dotenv()`, `WebBase` cannot read Postgres / Redis / Elasticsearch credentials. The dev `.env` and `.env.example` then resolve those services by Docker-network name (`prokaryotes-postgres`, `prokaryotes-redis`, `http://prokaryotes-elasticsearch:9200`); host-side `uv run pytest` can't reach those, so the conftest immediately rewrites `POSTGRES_HOST`, `REDIS_HOST`, and `ELASTIC_URI` to localhost-mapped ports. Tests that run *inside* the compose network (e.g. via `docker compose run`) wouldn't need the rewrite, but we don't currently support that path.
- **No `monkeypatch_session`.** `monkeypatch` is function-scoped only. We use direct `os.environ[...] = ...` assignment at module level instead — simpler than instantiating `pytest.MonkeyPatch()`, and unlike `setdefault` it overrides any value the caller's shell may have set, keeping the integration tier deterministic.
- **`loop_scope="session"`** on async session fixtures. Required by pytest-asyncio ≥1.0 to keep the same event loop alive across the session so `LifespanManager`, the Redis client, and other async resources don't get torn down between tests.

A small helper drains the NDJSON stream into a structured list of events, and a second helper waits for backgrounded `finalize()` to complete before any Redis/ES inspection. Because the harness is session-scoped, `harness.background_tasks` is shared across tests — waiting on the whole set would block on stale work from earlier tests *and* any work the current request hasn't yet spawned. So we snapshot the set *before* the request, then wait only on tasks that appeared after.

```python
async def collect_stream(response: httpx.Response) -> list[dict]:
    events = []
    async for line in response.aiter_lines():
        if line.strip():
            events.append(json.loads(line))
    return events


@asynccontextmanager
async def request_scope(harness):
    """Snapshot harness.background_tasks before a request; on exit, await only the new ones."""
    before = set(harness.background_tasks)
    yield
    new_tasks = list(harness.background_tasks - before)
    if new_tasks:
        await asyncio.gather(*new_tasks, return_exceptions=True)
```

Usage pattern in a test:

```python
async with request_scope(web_harness):
    async with authed_client.stream("POST", "/chat", json=payload) as response:
        events = await collect_stream(response)
# After the `async with request_scope` block exits, every background task spawned by this
# request (finalize on the normal path, _compact_partition on the compaction path) has
# completed, and Redis/ES inspections below are safe.
```

This keeps each test's wait scoped to its own request and avoids cross-test interference. Tests that issue multiple requests wrap each one in its own `request_scope`.

### Stream order

Single-turn (no tool call) order is symmetric across providers. Both clients buffer `text_delta` events when `stream_ndjson=True` and emit `context_pct` first, then the buffered text deltas:

```
partition_uuid → context_pct → text_delta(s) → [no compaction_pending]
```

(anthropic_v1/__init__.py:108–115, openai_v1/__init__.py:115 + 192–194.)

**Tool-call rounds are NOT symmetric across providers.** This is a real production-code asymmetry that test fakes and assertions must respect:

- **Anthropic** (anthropic_v1/__init__.py:97–133): the inner stream completes first, *then* the round emits `context_pct` (line 109), `progress_message` (line 120), `tool_call`s (lines 128–133). Order:
  ```
  context_pct (round 1) → progress_message (round 1) → tool_call(s) (round 1) → ...
  ```
- **OpenAI** (openai_v1/__init__.py:81–115, 173–176): events are emitted *during* the inner stream loop. `tool_call` fires on `response.output_item.done` (line 102–108), `context_pct` fires on `response.completed` (line 115), and `progress_message` is yielded only *after* the inner loop exits (line 175–176). Order:
  ```
  tool_call(s) (round 1) → context_pct (round 1) → progress_message (round 1) → ...
  ```

The fakes mirror their respective provider's order. Cross-provider parametrized tests assert the relevant set of events appear, but order assertions for tool-call rounds are provider-specific. Final-round (non-tool) assertions remain symmetric.

**Tests assert event types and order, not chunking.** Real Anthropic streams text approximately character-by-character; real OpenAI streams in larger semantic chunks; the fakes' delta sizes are arbitrary. Tests should check (a) the sequence of NDJSON event *types* (`partition_uuid` → `context_pct` → `text_delta`+ → ...), and (b) that concatenating all `text_delta` payloads in a round yields the round's expected text. Tests should *not* assert exact `text_delta` count, exact partitioning of text into deltas, or tokenizer-aligned splits — those are provider implementation details and would make the suite needlessly brittle.

### Scenarios (Tier B)

Each scenario is parametrized over `[anthropic, openai]` where the harness behavior is symmetric. Every scenario wraps the request in `async with request_scope(web_harness):` so backgrounded `finalize()` (and `_compact_partition` where applicable) drains before any Redis or ES inspection.

1. **Single-turn happy path** — POST `/chat` with one user message, script one round with `stop_reason="end_turn"`. Assert: stream begins with `partition_uuid`, then `context_pct`, then the buffered `text_delta` events, no `compaction_pending`. After the `request_scope` exits: Redis holds a partition for the conversation; ES holds the same partition with `is_compacted=False`.

2. **Multi-turn continuation** — Run scenario 1, send a second message echoing the received `partition_uuid` (each request in its own `request_scope`). Assert the second response's `partition_uuid` matches the first (Redis fast path); after the second `request_scope` exits, the partition's `items` list grows to include both turns.

3. **Tool-call round-trip** — Script round 1 with `stop_reason="tool_use"` and a `shell_command` tool call; round 2 with `stop_reason="end_turn"`. Assert provider-specific event order per the "Stream order" section: Anthropic emits `context_pct → progress_message → tool_call` within round 1; OpenAI emits `tool_call → context_pct → progress_message` within round 1. The final round in both cases emits `context_pct → text_delta(s)`. After `request_scope` exits: partition contains `function_call` and `function_call_output` items in the right order; the first `function_call` carries `text_preamble` if a preamble was scripted.

4. **Forced compaction** — Requires `COMPACTION_RECENCY_TAIL` lowered (e.g., 2) so a short conversation can demonstrate `raw_message_start_index` advancement. Establish 3 user/assistant turns (so the tail doesn't swallow the whole conversation), then on the next turn script a round with high `input_tokens` to trip the threshold. Assert: response ends with `compaction_pending`. Capture the `partition_uuid` of the partition that emitted `compaction_pending` — call it `pending_partition_uuid`. The `request_scope` exiting tells us the background `_compact_partition` task *finished*, but it does not tell us whether the Redis swap *succeeded* (the swap has its own preconditions: lock acquisition, current partition match, prefix-unchanged check at `web_v1/__init__.py:204-213`). The authoritative signal is `/compaction-status`. Within a bounded poll loop (e.g. 30 attempts × 100ms), issue:
   ```
   GET /compaction-status?conversation_uuid={conversation_uuid}&pending_partition_uuid={pending_partition_uuid}
   ```
   and assert it eventually returns `{"done": true}`. The handler returns `done` once the compaction lock is released *and* the active Redis partition's `partition_uuid` differs from `pending_partition_uuid` (`web_v1/__init__.py:408-424`), which is the only state that guarantees the swap actually landed. **Only after `done == true`** do we send the follow-up `/chat` turn — not after the request_scope exits, not after a sleep, not after the stream finishes. Issuing the follow-up earlier risks racing the swap and getting back the still-active *old* partition. The follow-up echoes the *old* `partition_uuid` (= `pending_partition_uuid`); assert the new response carries a *different* `partition_uuid`, that the new partition has `ancestor_summaries == ["STUB SUMMARY"]`, and that `raw_message_start_index` is greater than zero (advancement past the recency-tail boundary).

5. **Retry within recency tail** — Establish a compacted ancestor (run scenario 4), then send a follow-up that re-sends a message inside the recency tail with edited content. Assert: same active `partition_uuid`, partition `items` truncated and re-extended at the divergence point, `ancestor_summaries` retained.

6. **Retry before recency tail** — Establish a compacted ancestor, then send a follow-up whose message count falls below `raw_message_start_index`. Assert: response stream produces a *new* `partition_uuid` with `ancestor_summaries == []` and `raw_message_start_index == 0` — the chain reconstruction path correctly refused to attach a stale summary.

7. **Branch switch** — Establish two branches (two distinct `partition_uuid`s under one `conversation_uuid`). Send a request echoing the second branch's UUID. Assert: chain reconstruction or exact ES load succeeds, and Redis is repopulated for the second branch.

8. **Auth boundary** — POST `/chat` with no session cookie; assert HTTP 400 "Session expired". POST with empty `messages` after authenticating; assert HTTP 400 "At least one message is required".

9. **Stream protocol shape** — Parametrize across all scenarios that produce a stream and assert the event-type sequence matches the order documented in "Stream order" above (`partition_uuid` first; per non-tool round, `context_pct` before any buffered `text_delta`; tool-call rounds use the provider-specific order; `compaction_pending` last when present).

### State assertions for Tier B

Each scenario, after completion, can directly inspect Redis and ES through the harness:

```python
cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
partition = ContextPartition.model_validate_json(cached)
assert partition.ancestor_summaries == ["STUB SUMMARY"]

doc = await web_harness.search_client.get_partition(partition.partition_uuid)
assert doc["is_compacted"] is False
```

For compaction scenarios, also assert the parent ES doc — the compaction swap is two-sided. `_compact_partition` writes `is_compacted=True` plus boundary metadata to the *old* partition's ES doc *before* the Redis swap (`web_v1/__init__.py:176-184`), so any regression that breaks the parent-side write would otherwise pass the active-partition assertion alone:

```python
parent_doc = await web_harness.search_client.get_partition(pending_partition_uuid)
assert parent_doc["is_compacted"] is True
assert parent_doc["summary"]                          # non-empty stub summary
assert parent_doc["boundary_hash"]                    # non-empty
assert parent_doc["tail_hash"]                        # non-empty
assert parent_doc["boundary_message_count"] > 0
assert parent_doc["boundary_user_count"] > 0
```

This is the main reason Tier B uses real DBs: the persisted partition shape after a compaction swap is the contract that matters.

---

## Tier A — Live stack, real LLM (smoke)

### Scope

A small number of end-to-end tests run by hand against a real provider. The intent is not coverage but a final pass before a release that the harness still talks to live providers correctly. Each test costs real tokens. Budget: roughly 5–10 tests total across both providers.

Identical fixture machinery to Tier B except:

- `llm_client` is *not* replaced with a fake.
- Per-provider, session-scoped key gates skip Tier A tests at fixture setup if the provider's API key is missing (see below).
- `COMPACTION_TOKEN_THRESHOLD_PCT` is lowered to `1` (set in the root `tests/integration_tests/conftest.py`; same value Tier B uses — see "Triggering compaction in Tier A" below for the math) so a few-turn conversation can trigger compaction without burning excessive tokens.

### Key gating

A single `live_keys` fixture is wrong here: an Anthropic-only smoke test must not skip when `OPENAI_API_KEY` is the missing one (and vice versa). The skip also has to fire *before* `WebHarness.init()` runs, because `init()` calls `llm_client.init_client()` which constructs `AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))` / `AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))` (`anthropic_v1/__init__.py:54`, `openai_v1/__init__.py:121`); a missing key wouldn't blow up at construction but would surface as an authentication error at first request — a noisy failure rather than a clean skip.

The fix is two session-scoped, provider-specific gates that the per-provider harness fixtures depend on. The gates live in the **root** `tests/integration_tests/conftest.py` (not `tier_a/conftest.py`) so `judge_client` — also in the root conftest — can depend on `live_keys_openai` directly. Skipping in a session-scoped fixture propagates the skip to every test that resolves the dependent harness fixture:

```python
# tests/integration_tests/conftest.py — alongside the env setup shown earlier.
@pytest.fixture(scope="session")
def live_keys_anthropic():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

@pytest.fixture(scope="session")
def live_keys_openai():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def judge_client(live_keys_openai):       # judge tests skip cleanly when key is absent
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        yield client
    finally:
        await client.close()
```

```python
# tests/integration_tests/tier_a/conftest.py
import pytest_asyncio
from asgi_lifespan import LifespanManager

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _web_harness_anthropic(live_keys_anthropic):
    from prokaryotes.anthropic_v1.web_harness import WebHarness
    harness = WebHarness(static_dir="scripts/static")
    harness.init()                              # uses real AsyncAnthropic — needs key
    async with LifespanManager(harness.app):
        yield harness

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _web_harness_openai(live_keys_openai):
    from prokaryotes.openai_v1.web_harness import WebHarness
    harness = WebHarness(static_dir="scripts/static")
    harness.init()
    async with LifespanManager(harness.app):
        yield harness
```


The `web_harness` indirect fixture and the per-provider `authed_client_*` fixtures mirror Tier B's shape (see "Test client fixture" above) — only the underlying `_web_harness_*` differs. There is no autouse fake-reset in Tier A because there is no fake.

The judged scenarios (J1, J2) also depend on `judge_client` (`tests/integration_tests/conftest.py`) which itself requires `OPENAI_API_KEY` — Tier A judge tests therefore double-gate via both the subject key (`live_keys_anthropic` or `live_keys_openai`, depending on the parametrized provider) and the judge key (`live_keys_openai`, indirectly via `judge_client`). When subject == judge == OpenAI, that's still just one missing-key check.

### Scenarios (Tier A — structural)

1. **Live single-turn happy path** — assert structural invariants only: `partition_uuid` arrives first, exactly one `context_pct` event arrives per round (preceding the buffered `text_delta`s in the final round per the Tier B "Stream order" section), at least one `text_delta` arrives, no exception during stream finalization, persisted partition shape is well-formed. No assertion on text content.
2. **Live forced compaction** — same conversation pattern as Tier B scenario 4. Assert `compaction_pending` arrives, `/compaction-status` reaches `done`, the next turn's partition carries non-empty `ancestor_summaries`. Do not assert on the summary's text.
3. **Live tool-call round-trip (best-effort)** — neither client wires `tool_choice`, so the model is free to answer in text without invoking a tool. Prompt with a request that strongly invites `shell_command` use (e.g., "list files in /tmp using a shell command"). Assert that the request finishes without error and the stream ends cleanly. *If* a `tool_call` event arrives, assert it has the expected NDJSON shape (`{"tool_call": {"name": str, "arguments": str}}`). Do not assert which tool was chosen, and do not fail the test on absence — that would make the suite flaky against model behavior we can't currently force. If forcing a tool call later becomes important for this tier, plumb a `tool_choice` parameter into both `stream_turn` implementations first; that's a separate change from this test plan.

### Scenarios (Tier A — LLM-judged)

Two judge-backed tests, both targeting compaction-specific behavioral contracts that structural assertions can't reach.

#### J1. Memory continuity across the compaction boundary

Plant a non-trivial fact in turn 1, fill turns 2..N with neutral filler until the threshold trips, force compaction, ask a question in turn N+1 that requires recalling the planted fact.

```
Turn 1 (user): "Quick fact for later: I drive a 2019 Tesla Model 3, color red, named Mochi."
Turn 1 (asst): <real LLM response>
Turns 2..N: filler ("tell me about X" — keeps tokens climbing)
[compaction fires]
Turn N+1 (user): "What color is my car?"
Turn N+1 (asst): <real LLM response, judged>
```

The judge is asked a binary, falsifiable question:

```
Given the assistant's response below, did it correctly identify the user's car
color as red? Reply with structured output:
{ "passed": <bool>, "reason": "<short string>" }

Response: <captured assistant text from turn N+1>
```

Pass = compaction summary preserved the fact and the model used it.

#### J2. Branch isolation after retry-before-tail

Plant a branch-only fact on the original branch ("my private codename is PHORBAS-QUANTA-93"), do enough turns to seal a compacted ancestor, then *retry below* the recency tail with a different topic. Ask for that exact codename on the new branch with instructions to answer `UNKNOWN` if it did not appear in earlier user messages on the current branch.

Judge prompt (binary): pass only if the response says it cannot know the codename on this branch or answers `UNKNOWN`; fail if it states or guesses a specific codename, especially `PHORBAS-QUANTA-93`. This avoids false failures from unrelated runtime context like the working directory name.

### Judge design and implementation

- **Binary, falsifiable prompts only.** `{"passed": bool, "reason": str}`. No 1–10 quality scores; those drift with the judge model.
- **Single fixed judge: OpenAI via `text.format`.** Schema-strict structured output (`"strict": true`) gives guaranteed-shape JSON with no parsing fallback, and a single judge keeps the code path simple. Same-provider bias is a non-issue for binary fact-recall and fact-leak questions like J1 and J2 — the judge answers based on whether specific tokens or facts appear in the response, not on stylistic agreement with the subject.
- **2-of-3 majority** for each judged assertion. Same model called three times; flake insurance, not bias correction.
- **Judge calls are *not* part of the harness flow** — they live in `tests/integration_tests/judges.py` and talk directly to `AsyncOpenAI`, not through `LLMClient`. The verdict schema is a plain JSON dict; we don't reuse `ToolSpec` because the OpenAI side is a structured response, not a tool call.

```python
# tests/integration_tests/judges.py
VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["passed", "reason"],
}

async def llm_judge(client: AsyncOpenAI, criterion: str, response: str) -> JudgeVerdict:
    result = await client.responses.create(
        model=OPENAI_DEFAULT_MODEL,
        input=[{"role": "user", "content": _build_judge_prompt(criterion, response)}],
        text={"format": {
            "type": "json_schema",
            "name": "verdict",
            "schema": VERDICT_SCHEMA,
            "strict": True,
        }},
    )
    args = json.loads(result.output_text)
    return JudgeVerdict(**args)

async def llm_judge_majority(
    client: AsyncOpenAI,
    criterion: str,
    response: str,
    *,
    n: int = 3,
) -> JudgeVerdict:
    verdicts = await asyncio.gather(*[llm_judge(client, criterion, response) for _ in range(n)])
    passed = sum(1 for v in verdicts if v.passed) > n // 2
    return JudgeVerdict(passed=passed, reason="; ".join(v.reason for v in verdicts))
```

The `AsyncOpenAI` instance is supplied by a session-scoped fixture in `tests/integration_tests/conftest.py` (shown in "Key gating" above — it depends on `live_keys_openai` so judged tests skip cleanly when the key is missing), so the underlying httpx connection pool is reused across all judge calls in a session and torn down once at the end. Reconstructing `AsyncOpenAI` inside `llm_judge` would build (and leak, until GC) a fresh httpx client *3 × judged-assertions* times per run.

### Triggering compaction in Tier A

The harness's threshold check is integer math:

```python
context_pct = int(input_tokens / context_window * 100)
if context_pct >= COMPACTION_TOKEN_THRESHOLD_PCT:
    pending_compaction[0] = True
```

Default models are `claude-haiku-4-5` and `gpt-5.4-mini`, both with a 128k context window, so each integer percent is ~1,280 input tokens. At `THRESHOLD_PCT=5`, compaction fires only after cumulative input crosses ~6,400 tokens — *not* reliably reached in 4–6 short-message turns, since each turn typically adds only a few hundred tokens of context. Tier A therefore sets `COMPACTION_TOKEN_THRESHOLD_PCT=1`, which drops the trigger point to ~1,280 input tokens and lets 2–3 normal turns reach it. Filler prompts that intentionally elicit longer responses ("write a 200-word paragraph about ...") are still useful as a backstop, but are not required at `THRESHOLD_PCT=1`.

---

## Infrastructure changes

### File layout

```
tests/
  unit_tests/                # default `testpaths`; hermetic unit tier
    ...
  tests/integration_tests/         # OUTSIDE `testpaths`, so default `uv run pytest` skips it
    __init__.py              # explicit package; helpers import as `tests.integration_tests.fakes` etc.
    conftest.py              # session-scope env setup, live_keys_*, judge client, common helpers
    fakes.py                 # FakeAnthropicClient, FakeOpenAIClient, LLMScript, LLMRound
    judges.py                # llm_judge() + JudgeVerdict + 2-of-3 majority helper
    stream_utils.py          # collect_stream(), request_scope(), assert_event_sequence()
    tier_b/
      __init__.py
      conftest.py            # fake-backed harness + authed_client fixtures; autouse fake reset
      test_chat_flow.py      # B: scenarios 1–3
      test_compaction_flow.py # B: scenarios 4–7
      test_auth_boundary.py  # B: scenario 8
      test_stream_protocol.py # B: scenario 9
    tier_a/
      __init__.py
      conftest.py            # real-client harness + authed_client fixtures; live_keys skip
      test_smoke_anthropic.py # A: structural smoke (Anthropic)
      test_smoke_openai.py   # A: structural smoke (OpenAI)
      test_smoke_judged.py   # A: J1 + J2
```

The integration tier sits next to `tests/unit_tests/` rather than under it because pytest imports `conftest.py` files during *collection* — before `-m` marker filters are applied. If the integration `conftest.py` lived under `testpaths`, it would still execute on every default `uv run pytest` run and mutate `POSTGRES_HOST`, `REDIS_HOST`, `ELASTIC_URI`, `COMPACTION_TOKEN_THRESHOLD_PCT`, `COMPACTION_RECENCY_TAIL` in the unit-test process. Setting `testpaths = ["tests/unit_tests"]` is the cleanest way to guarantee the integration conftest never imports during a default run. This also aligns with [tests/CLAUDE.md](../../../tests/CLAUDE.md), which explicitly says tests under `tests/unit_tests/` must run "without Docker or live external services" — the integration tier intentionally violates that constraint.

The `tier_a/` / `tier_b/` subdirectory split lets each tier own the harness builder appropriate to it (fake-client vs real-client) and keeps the autouse fake-reset fixture in `tier_b/conftest.py` only — Tier A real clients have no `reset()` and tier-mixed autouse would be brittle.

### Pytest configuration

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--strict-markers --cov=prokaryotes"
markers = [
    "integration: live DB integration tests (Tier B); requires docker-compose stack",
    "live: live LLM smoke tests (Tier A); requires real API keys",
]
testpaths = ["tests/unit_tests"]
```

The default-run isolation is **structural**, not marker-driven: `testpaths = ["tests/unit_tests"]` does not include `tests/integration_tests/`, so the default `uv run pytest` never imports the integration conftest at all. The `integration` and `live` markers are kept for in-tier filtering and self-documentation, but they are not load-bearing for the default run — that's why `addopts` no longer carries `-m 'not integration and not live'`.

Run commands:

```bash
# Default — integration tests are not collected, no env mutation, no new dependencies pulled in.
uv run pytest

# Tier B — assumes the DB stack is already running:
#   docker compose up -d elasticsearch elasticsearch-init postgres postgres-migrate redis
uv run --extra test pytest tests/integration_tests/tier_b

# Tier A — same stack, plus API keys in env
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... uv run --extra test pytest tests/integration_tests/tier_a

# Both
uv run --extra test pytest tests/integration_tests
```

### New test-only dependencies

Add to `[project.optional-dependencies].test`, with explicit floors:

- `asgi-lifespan>=2.1,<3` — drives FastAPI `lifespan` events for in-process ASGI testing. 2.1 is the current stable; cap below 3 in case of breaking changes.
- `pytest-asyncio>=1.0` — required because `loop_scope="session"` was added in 1.0. The current `pytest-asyncio` entry in the test extra is unpinned, so this becomes a real change to pyproject.toml.
- `httpx` and `python-dotenv` are already runtime dependencies; no version bump needed.

### Test user provisioning

Each session creates a fresh user via `POST /register` with `full_name="Peter Prokaryote"`, a UUID-derived email (`peter-<uuid4>@prokaryotes.test`), and a random password generated via `secrets.token_urlsafe`. The same authenticated client is reused across tests in the session. Postgres state accumulates across runs but does not affect correctness; the fixed `full_name` makes cleanup a single statement:

```sql
DELETE FROM chat_user WHERE full_name = 'Peter Prokaryote';
```

This can be wrapped in a small helper script later if cleanup needs to be automated.

### Compaction-related env management

`COMPACTION_TOKEN_THRESHOLD_PCT` and `COMPACTION_RECENCY_TAIL` are set at the **top of `tests/integration_tests/conftest.py`**, above any `from prokaryotes…` import, via direct `os.environ[...] = ...` assignment (not `setdefault`, which would let any shell value leak through). Fixture-based env mutation (`monkeypatch.setenv`, `pytest.MonkeyPatch().setenv`) does not work here because both web harness modules pull these constants into their own namespace at import time via `from prokaryotes.utils_v1.llm_utils import …`, so the bindings are frozen the moment the harness module is loaded — and pytest loads test modules before fixtures run.

The conftest pattern is shown in "Test client fixture" above. Per-test threshold variation is therefore not supported by simple env mutation; if it's ever needed, it requires re-importing the harness module in an isolated subprocess, which we explicitly do not plan to do.

### Compose stack

Reuse the existing `docker-compose.yml` services (`elasticsearch`, `elasticsearch-init`, `postgres`, `postgres-migrate`, `redis`). No new compose file and no alternate-port test stack. The integration tier connects to the same ports the dev stack uses (5432, 9200, 6379), so dev and test share state on the local machine.

---

## Relevant code files

| File | Role in this design |
|---|---|
| `prokaryotes/anthropic_v1/web_harness.py` | Subject under test (Anthropic harness). |
| `prokaryotes/openai_v1/web_harness.py` | Subject under test (OpenAI harness). |
| `prokaryotes/web_v1/__init__.py` | `stream_and_finalize`, `sync_context_partition`, `_compact_partition`, `get_compaction_status`. |
| `prokaryotes/anthropic_v1/__init__.py` | `AnthropicClient.stream_turn` — defines the NDJSON event shape the fake must emit. |
| `prokaryotes/openai_v1/__init__.py` | `OpenAIClient.stream_turn` — same. |
| `prokaryotes/api_v1/models.py` | `ChatConversation`, `ContextPartition`, `ContextPartitionItem` — request/response shapes. |
| `prokaryotes/utils_v1/llm_utils.py` | `COMPACTION_TOKEN_THRESHOLD_PCT` and `MODEL_CONTEXT_WINDOWS` — env-time constants the integration tier must override. |
| `tests/context_partition_utils.py` | Existing fakes / builders. The integration tier reuses `make_chat_messages` and similar helpers; it does *not* reuse `FakeRedis` / `FakeSearchClient` (those are for the unit tier). |
| `docker-compose.yml` | Provides the live DB stack the integration tier connects to. |
| `pyproject.toml` | Pytest markers, `addopts` exclusion, new `asgi-lifespan` test dependency. |
| `project/features/compaction/README.md` | Defines the contracts under test. |
