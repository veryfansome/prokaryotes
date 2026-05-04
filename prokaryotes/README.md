# Codebase Overview

## Module layout

- `prokaryotes/api_v1/models.py` — Shared Pydantic models: `ContextPartition` (provider-agnostic conversation history; carries compaction chain fields), `ContextPartitionItem` (unified message/tool-call/tool-output), `ChatConversation` (HTTP request payload), `ToolSpec`/`ToolParameters` (provider-agnostic tool definitions, `.to_anthropic_tool_param()` / `.to_openai_function_tool_param()`), and the `FunctionToolCallback` protocol. Also defines hash functions and exception classes for the reconciliation layer.
- `prokaryotes/web_v1/` — `WebBase` base class providing auth (login/register via Postgres), Redis-backed sessions, `GraphClient`/`SearchClient` lifecycle, `sync_context_partition()` (three-tier reconciliation: Redis fast path → exact ES load → ancestor chain reconstruction), `stream_and_finalize()` (response streaming with optional background compaction), and common routes. LLM-specific harnesses extend this.
- `prokaryotes/anthropic_v1/` — `AnthropicClient` (`messages.stream()` in a `while True:` tool-call loop with `context_pct` emission), `WebHarness(WebBase)` wiring it into a FastAPI app with compaction triggering (`on_usage` callback, `_summarize_and_compact()`), and `ScriptHarness` for non-interactive one-off tasks.
- `prokaryotes/openai_v1/` — `OpenAIClient` (Responses API with event-driven tool-call continuation (`handle_response_stream_event()`); `context_pct` emission), `WebHarness(WebBase)` mirroring the Anthropic harness with developer-message assembly including ancestor summaries, and `ScriptHarness`.
- `prokaryotes/tools_v1/` — Reusable `FunctionToolCallback` implementations. `ThinkTool` takes structured `goal`, `context`, and `perspectives` parameters, makes focused LLM calls, and returns structured analysis the model can act on. `ShellCommandTool` lets the model run arbitrary shell commands (output truncated at 400 lines).
- `prokaryotes/eval_v1/` — Lightweight eval framework. `EvalTask` defines a task with setup files, a setup command, and a shell `check_command` that exits 0 on success. `EvalHarness` runs tasks sequentially in isolated workspaces under `/tmp/prokaryotes_eval/`, delegates to `ScriptHarness` per task, and produces an `EvalRun` with per-task pass/fail and a tier breakdown. `tasks.py` contains a curated set of 15 tasks across three tiers (basic file ops → multi-step → reasoning). Compaction is not implemented for script or eval harnesses.
- `prokaryotes/utils_v1/` — Shared utilities. `llm_utils.py` provides the `MODEL_CONTEXT_WINDOWS` lookup table and compaction configuration constants. `system_message_utils.py` assembles runtime context (time, platform, cwd, unix user) for injection into system/developer messages. `time_utils.py` handles timezone-aware datetime formatting. `os_utils.py`, `text_utils.py`, `http_utils.py`, and `logging_utils.py` provide additional support.
- `prokaryotes/emb_v1.py` — Standalone FastAPI embedding service using `sentence-transformers`. Runs separately on port 8001.
- `prokaryotes/graph_v1.py` — Neo4j async client for topic similarity graph operations.
- `prokaryotes/search_v1/` — `SearchClient` combining `ContextPartitionSearcher` (Elasticsearch CRUD for all partitions — active, forked, and compacted — with full-text indexing of summaries and message content), `NamedEntitySearcher`, and `TopicSearcher`.

## Key design patterns

**`ContextPartition` as the lingua franca**: Both LLM clients accept a mutable `ContextPartition` and append assistant messages and tool call records to it as streaming progresses. With compaction, a conversation is modeled as a chain of partitions rather than a single flat list; each partition carries a `partition_uuid`, an optional `parent_partition_uuid`, an `ancestor_summaries` list, and a `raw_message_start_index` marking where its raw items begin within the full conversation. See [project/features/compaction/README.md](../project/features/compaction/README.md) for the full data model.

**Three-tier partition reconciliation**: `sync_context_partition()` in `WebBase` produces the correct `ContextPartition` for each incoming request via a strict priority order: (1) Redis fast path if the cached partition matches or is the direct parent of the client's `partition_uuid`; (2) exact ES load by `partition_uuid` for non-compacted partitions; (3) ancestor chain reconstruction by walking `parent_partition_uuid` links through ES, validating each ancestor's `boundary_hash` before assembling summaries. See the compaction README for reconciliation details.

**Streaming with tool-call continuation**: `stream_turn()` yields text deltas while accumulating tool calls; after the stream ends it feeds results back into further LLM calls until no new tool calls are produced. `max_tool_call_rounds` caps the loop. After each round, `context_pct` usage is reported and background compaction is triggered when the configured threshold is reached. When the model streams text before stopping with `tool_use`, that text is stored in `text_preamble` on the *first* `function_call` `ContextPartitionItem` of that round — never as a standalone assistant item before it (subsequent `function_call` items in the same round carry no `text_preamble`). Storing preamble text as a standalone item causes `sync_from_conversation()` to truncate context on the next request; `to_anthropic_messages()` and `to_openai_input()` reconstruct the text block at API-call time from `text_preamble`, and `_summarize_and_compact()` must exclude it from the compaction payload.

## Dependencies

| Service | Env vars | Default port |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_MAX_TOKENS` (default 4096) | — |
| Elasticsearch | `ELASTIC_URI` | 9200 |
| Neo4j | `NEO4J_URI`, `NEO4J_AUTH` (user/pass) | 7687 |
| OpenAI | `OPENAI_API_KEY` | — |
| Postgres | `POSTGRES_HOST/PORT/DB/USER/PASSWORD/SSL_MODE` | 5432 |
| Redis | `REDIS_HOST/PORT` | 6379 |

Use a `.env` file at the project root (loaded via `python-dotenv` in `scripts/web.py`).

## Code organization

**Classes**: introduce a class when methods need to share mutable state. A set of related functions is not sufficient justification — keep them at module level.

**Method type**: use the least powerful form that satisfies the need — instance method if it needs `self`, `@classmethod` if it needs `cls` but not an instance (e.g. alternative constructors), `@staticmethod` if it logically belongs to the class but needs neither, and module-level if it has no meaningful connection to any class.

**Ordering**: `__init__` first within a class, then everything else — methods and module-level functions alike — sorted alphabetically. Alphabetical order gives a predictable location for any definition without requiring the reader to reconstruct call graphs or execution flow.

**Parameter ordering**: required parameters before optional ones, and alphabetical within each group.
