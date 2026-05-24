# Codebase Overview

## Module layout

- `prokaryotes/api_v1/models.py` — Wire-format Pydantic models and protocols: `IncomingMessage`/`IncomingConversation` (HTTP request payloads), `CompactionStatusResponse`, `ToolSpec`/`ToolParameters` (provider-agnostic tool definitions, `.to_anthropic_tool_param()` / `.to_openai_function_tool_param()`), and the `FunctionToolCallback` and `LLMClient` protocols.
- `prokaryotes/conversation_v1/models.py` — Internal conversation data model: `Conversation` (provider-agnostic conversation snapshot; carries compaction chain fields via `ancestor_summaries` and `parent_snapshot_uuid`), `ConversationMessage` (external dialogue messages), `TurnItem` (LLM-internal `function_call` / `function_call_output` records), `TurnExecution` (per-bot-reply log of `TurnItem`s), `ProjectedItem` (projection for provider wire-format), `NormalizedMessage`, plus reconciliation operations and hash functions.
- `prokaryotes/context_v1/` — Shared conversation-context lifecycle: `ConversationSyncer` (Redis/ES reconciliation, source_id assignment, unacknowledged-bot detection), `ConversationCompactor` (background CAS compaction over the conversation chain), and `get_redis_client()`.
- `prokaryotes/web_v1/` — `WebBase`: FastAPI base for the chat-over-HTTP harness. Owns auth, session middleware, the Postgres lifecycle, and the `/compaction-status` endpoint while re-exporting the shared context surface from `context_v1/`. See [web_v1/README.md](web_v1/README.md).
- `prokaryotes/slack_v1/` — `SlackBase`: Socket Mode lifecycle and inbound event dispatch. Owns the per-workspace WebSocket, the `_should_handle` trigger gate, the per-`conversation_uuid` dispatch lock, and three hooks (`_should_continue_threaded` / `_on_dispatch_accept` / `_dispatch_conversation_uuid`) that `SlackHarness` overrides to drive the single-human-thread continuation cache (`slack_thread_humans` SET + `slack_thread_engaged` flag, both 90d TTL). `replay.py` reconciles a fetched Slack thread into a `Conversation` (`sync_slack_thread`'s helpers); `streaming.py` ships bot replies via `chat.postMessage` with the `prokaryotes_in_flight` metadata-recovery contract.
- `prokaryotes/anthropic_v1/` — `AnthropicClient`: `messages.stream()` in a `while True:` tool-call loop with `context_pct` emission.
- `prokaryotes/openai_v1/` — `OpenAIClient`: Responses API with event-driven tool-call continuation (`handle_response_stream_event()`) and `context_pct` emission.
- `prokaryotes/harness_v1/` — Provider-agnostic agent-loop harnesses. `HarnessBase(ConversationCompactor)` provides the shared Redis/Search/compaction lifecycle for harnesses that need it. `WebHarness(WebBase)` is the chat-over-HTTP harness. `SlackHarness(SlackBase)` is the Socket Mode harness — one process per workspace; runs `_locked_turn` (sync → Site A continuation-cache write → prelude → `_run_turn`) under a per-thread turn lock. `ScriptHarness` is standalone (no persistence — in-memory conversation returned to the caller) for non-interactive one-off tasks. `EvalHarness` orchestrates `ScriptHarness` runs over `EvalTask` fixtures. Both `WebHarness` and `SlackHarness` dispatch on `impl` to select the LLM client (`AnthropicClient`/`OpenAIClient`) and instruction role (`"system"` / `"developer"`).
- `prokaryotes/tools_v1/` — Reusable `FunctionToolCallback` implementations (`FileTool`, `ThinkTool`, `ShellCommandTool`). See [tools_v1/README.md](tools_v1/README.md).
- `prokaryotes/eval_v1/` — Eval framework data. `EvalTask` (in `models.py`) defines a task with setup files, an optional setup command, check files (helpers written into the workspace at check time only), and a shell `check_command` that exits 0 on success. `tasks.py` is a loader that builds `TASKS` from the per-task fixture directories under [evals/](../evals/README.md); the orchestrator (`EvalHarness`) lives in `prokaryotes/harness_v1/eval.py` and produces an `EvalRun` with per-task pass/fail and a tier breakdown.
- `prokaryotes/utils_v1/` — Shared utilities. `llm_utils.py` provides the `MODEL_CONTEXT_WINDOWS` lookup table and compaction configuration constants. `system_message_utils.py` assembles runtime context (time, platform, cwd, unix user) for injection into system/developer messages. `time_utils.py` handles timezone-aware datetime formatting. `os_utils.py`, `text_utils.py`, and `logging_utils.py` provide additional support.
- `prokaryotes/graph_v1.py` — Neo4j async client for topic similarity graph operations.
- `prokaryotes/search_v1/` — `SearchClient` combining `ConversationSearcher` (ES CRUD for `Conversation` snapshots and `TurnExecution` records, with full-text indexing of summaries and message content) and `TopicSearcher` (tags records with `is_named_entity: bool` so queries can scope to topics, named entities, or both).

## Key design patterns

**`Conversation` as the lingua franca**: Both LLM clients receive a projected `list[ProjectedItem]` derived from the active `Conversation` snapshot. As streaming progresses, assistant messages and tool-call records are committed back as `TurnItem`s. With compaction, a conversation becomes a chain of snapshots rather than a single flat list; `HarnessBase.sync_conversation()` reconciles the active snapshot on each request for harnesses that use the shared context lifecycle. See [project/features/conversation/README.md](../project/features/conversation/README.md) for the conversation model, reconciliation, and branching, and [project/features/compaction/README.md](../project/features/compaction/README.md) for the snapshot-chain compaction lifecycle.

**Streaming with tool-call continuation**: `stream_turn()` buffers each provider round, dispatches tool calls when present, and feeds results back into further LLM calls until no new tool calls are produced or `max_tool_call_rounds` is reached. Per-round `context_pct` usage is reported and triggers background compaction at the configured threshold. The web harness surfaces transient `progress_message` and `tool_call` events for the UI; persisted `TurnItem`s are only the `function_call` / `function_call_output` records and any final assistant message — subsequent provider calls are reconstructed from those persisted items rather than from transient progress narration.

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

**Comment width**: wrap docstrings and long-form comments at 119 characters; `ruff format` won't reflow them.

**Method type**: use the least powerful form that satisfies the need — instance method if it needs `self`, `@classmethod` if it needs `cls` but not an instance (e.g. alternative constructors), `@staticmethod` if it logically belongs to the class but needs neither, and module-level if it has no meaningful connection to any class.

**Ordering**: `__init__` first within a class, then everything else — methods and module-level functions alike — sorted alphabetically. Alphabetical order gives a predictable location for any definition without requiring the reader to reconstruct call graphs or execution flow.

**Parameter ordering**: required parameters before optional ones, and alphabetical within each group.
