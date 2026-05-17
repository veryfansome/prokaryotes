# Codebase Overview

## Module layout

- `prokaryotes/api_v1/models.py` — Shared Pydantic models: `ContextPartition` (provider-agnostic conversation history; carries compaction chain fields), `ContextPartitionItem` (unified message/tool-call/tool-output), `ChatConversation` (HTTP request payload), `ToolSpec`/`ToolParameters` (provider-agnostic tool definitions, `.to_anthropic_tool_param()` / `.to_openai_function_tool_param()`), and the `FunctionToolCallback` protocol. Also defines hash functions and exception classes for the reconciliation layer.
- `prokaryotes/context_v1/` — Shared conversation-context lifecycle: `PartitionSyncer` (Redis/ES reconciliation), `PartitionCompactor` (background CAS compaction), and `get_redis_client()`.
- `prokaryotes/web_v1/` — `WebBase`: FastAPI base for the chat-over-HTTP harness. Owns auth, session middleware, the Postgres lifecycle, and the `/compaction-status` endpoint while re-exporting the shared context surface from `context_v1/`. See [web_v1/README.md](web_v1/README.md).
- `prokaryotes/anthropic_v1/` — `AnthropicClient`: `messages.stream()` in a `while True:` tool-call loop with `context_pct` emission.
- `prokaryotes/openai_v1/` — `OpenAIClient`: Responses API with event-driven tool-call continuation (`handle_response_stream_event()`) and `context_pct` emission.
- `prokaryotes/harness_v1/` — Provider-agnostic agent-loop harnesses: `HarnessBase` (shared Redis/Search/compaction lifecycle), `ScriptHarness` (non-interactive one-off tasks), `WebHarness(WebBase)` (chat-over-HTTP), and `EvalHarness` (orchestrates `ScriptHarness` runs over `EvalTask` fixtures). `WebHarness` dispatches on `impl` to select the LLM client (`AnthropicClient`/`OpenAIClient`) and instruction role (`"system"` / `"developer"`). Provider-specific summarization still lives in `WebHarness`.
- `prokaryotes/tools_v1/` — Reusable `FunctionToolCallback` implementations (`FileTool`, `ThinkTool`, `ShellCommandTool`). See [tools_v1/README.md](tools_v1/README.md).
- `prokaryotes/eval_v1/` — Eval framework data. `EvalTask` (in `models.py`) defines a task with setup files, an optional setup command, check files (helpers written into the workspace at check time only), and a shell `check_command` that exits 0 on success. `tasks.py` is a loader that builds `TASKS` from the per-task fixture directories under [evals/](../evals/README.md); the orchestrator (`EvalHarness`) lives in `prokaryotes/harness_v1/eval.py` and produces an `EvalRun` with per-task pass/fail and a tier breakdown.
- `prokaryotes/utils_v1/` — Shared utilities. `llm_utils.py` provides the `MODEL_CONTEXT_WINDOWS` lookup table and compaction configuration constants. `system_message_utils.py` assembles runtime context (time, platform, cwd, unix user) for injection into system/developer messages. `time_utils.py` handles timezone-aware datetime formatting. `os_utils.py`, `text_utils.py`, and `logging_utils.py` provide additional support.
- `prokaryotes/graph_v1.py` — Neo4j async client for topic similarity graph operations.
- `prokaryotes/search_v1/` — `SearchClient` combining `ContextPartitionSearcher` (ES CRUD for active/forked/compacted partitions with full-text indexing of summaries and message content) and `TopicSearcher` (tags records with `is_named_entity: bool` so queries can scope to topics, named entities, or both).

## Key design patterns

**`ContextPartition` as the lingua franca**: Both LLM clients accept a mutable `ContextPartition` and append assistant messages and tool-call records as streaming progresses. With compaction, a conversation becomes a chain of partitions rather than a single flat list; `HarnessBase.sync_context_partition()` reconciles the active partition on each request for harnesses that use the shared context lifecycle. See [project/features/compaction/README.md](../project/features/compaction/README.md) for the chain data model and reconciliation flow.

**Streaming with tool-call continuation**: `stream_turn()` buffers each provider round, dispatches tool calls when present, and feeds results back into further LLM calls until no new tool calls are produced or `max_tool_call_rounds` is reached. Per-round `context_pct` usage is reported and triggers background compaction at the configured threshold. The web harness surfaces transient `progress_message` and `tool_call` events for the UI; persisted `ContextPartition` items are only the `function_call` / `function_call_output` records and any final assistant message — subsequent provider calls are reconstructed from those persisted items rather than from transient progress narration.

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
