# prokaryotes

This project is about exploring agentic harnesses.

## Commands

### Python

```bash
# Install dependencies (including dev/test extras)
uv sync --extra dev --extra test

# Run all tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

### JavaScript

```bash
# Run JS tests
npm run test:js
```

### Running prokaryotes

```bash
# Run full stack including the web and embedding apps
docker compose up --build
```

```bash
# Run script-harness via CLI (Anthropic by default)
docker run --env-file .env prokaryotes:latest -- python -m scripts.cli "What's in the working directory?"

# Select provider, model, reasoning effort, and cap tool-call rounds
docker run --env-file .env prokaryotes:latest -- python -m scripts.cli "Summarise the repo" \
  --impl openai \
  --model gpt-5.4 \
  --reasoning-effort low \
  --max-tool-call-rounds 10 \
  --cwd /app
```

```bash
# Run the full eval suite
uv run python -m scripts.eval

# List available tasks (combinable with --tier)
uv run python -m scripts.eval --list
uv run python -m scripts.eval --list --tier 1

# Run a single tier or task
uv run python -m scripts.eval --tier 1
uv run python -m scripts.eval --task-id t1_implement_function

# Save results to JSON
uv run python -m scripts.eval --output results/run.json
```

## Architecture

This project is set up as a FastAPI-based application backed by multiple data stores. A web-harness provides a chat UI for human users. A script-harness enables running tasks non-interactively. An eval-harness can run a small curated evaluation set. Other kinds of harnesses can also be built.

### Module layout

- `prokaryotes/api_v1/models.py` — shared Pydantic models. The key abstractions are `ContextPartition` (provider-agnostic conversation history), `ToolSpec`/`ToolParameters` (provider-agnostic tool definition with `to_anthropic_tool_param()` / `to_openai_function_tool_param()` converters), and the `FunctionToolCallback` protocol (implementors expose `name`, `system_message_parts`, `tool_spec`, and an async `call()`). `ContextPartition.to_anthropic_messages()` converts to the `(system, messages)` format the Anthropic SDK expects.
- `prokaryotes/web_v1.py` — `WebBase` base class providing auth (login/register via Postgres), Redis-backed sessions, `GraphClient`/`SearchClient` lifecycle, `sync_context_partition()` (Redis cache + conversation sync), `stream_and_finalize()`, and common routes. LLM-specific harnesses extend this.
- `prokaryotes/anthropic_v1/` — Anthropic `LLMClient` (`messages.stream()` + tool-call loop), `WebHarness(WebBase)` wiring it into a FastAPI app, and `ScriptHarness` for non-interactive one-off tasks.
- `prokaryotes/openai_v1/` — OpenAI `LLMClient` (Responses API), `WebHarness(WebBase)`, and `ScriptHarness`.
- `prokaryotes/tools_v1/` — Reusable `FunctionToolCallback` implementations. `ThinkTool` gives the model a reasoning scratchpad between tool calls. `ShellCommandTool` lets the model run arbitrary shell commands.
- `prokaryotes/eval_v1/` — Lightweight eval framework. `EvalTask` defines a task with setup files, a setup command, and a shell `check_command` that exits 0 on success. `EvalHarness` runs tasks sequentially in isolated workspaces under `/tmp/prokaryotes_eval/`, delegates to `ScriptHarness` per task, and produces an `EvalRun` with per-task pass/fail and a tier breakdown. `tasks.py` contains a curated set of 15 tasks across three tiers (basic file ops → multi-step → reasoning).
- `prokaryotes/utils_v1/` — Shared utilities. `system_message_utils` assembles runtime context (time, platform, cwd, unix user) for injection into system/developer messages. `time_utils` handles timezone-aware datetime formatting.
- `prokaryotes/emb_v1.py` — Standalone FastAPI embedding service using `sentence-transformers`. Runs separately on port 8001.
- `prokaryotes/graph_v1.py` — Neo4j async client for topic similarity graph operations.
- `prokaryotes/search_v1/` — `SearchClient` combining `NamedEntitySearcher` and `TopicSearcher` via Elasticsearch.

### Key design patterns

**`ContextPartition` as the lingua franca**: Both LLM clients accept a mutable `ContextPartition` and append assistant messages and tool call records to it as streaming progresses. This same object is cached in Redis keyed by `context_partition:{conversation_uuid}` and synced against incoming `ChatConversation` payloads on each request.

**Streaming with tool-call continuation**: `stream_response()` yields text deltas while also accumulating tool calls. After the stream ends, it awaits all `FunctionToolCallback` tasks and—if all returned results—feeds them back into a second (or further) `create_response` call, yielding more text. This loop repeats until no new tool calls are produced. The Anthropic implementation uses `messages.stream()` natively and checks `stop_reason == "tool_use"` on the final message to drive the same loop, rather than a separate `create_response` method. A `\n` separator is emitted between rounds only when text was actually streamed in the preceding round. The optional `max_tool_call_rounds` parameter caps the loop to prevent runaway agents.

**`FunctionToolCallback` Protocol**: Defined once in `api_v1/models.py` and used by both LLM clients. Implementors expose `name`, `system_message_parts` (guidance lines for injection into the system/developer message), `tool_spec` (a `ToolSpec` that converts to the appropriate provider format), and an async `call(arguments, call_id)` returning a `ContextPartitionItem` of type `function_call_output`. Both web and script harnesses assemble the system message by iterating over all registered callbacks' `system_message_parts` alongside runtime context (time, platform, cwd, unix user).

**System message lifecycle**: System/developer messages are injected fresh on every request and must not be cached. `ContextPartition.pop_system_message()` strips the leading system/developer item before `finalize()` writes the partition to Redis, so the cache only ever holds the conversation turns.

### Infrastructure

| Service | Env vars | Default port |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_MAX_TOKENS` (default 4096) | — |
| Elasticsearch | `ELASTIC_URI` | 9200 |
| Neo4j | `NEO4J_URI`, `NEO4J_AUTH` (user/pass) | 7687 |
| OpenAI | `OPENAI_API_KEY` | — |
| Postgres | `POSTGRES_HOST/PORT/DB/USER/PASSWORD/SSL_MODE` | 5432 |
| Redis | `REDIS_HOST/PORT` | 6379 |

Use a `.env` file at the project root (loaded via `python-dotenv` in `scripts/web.py`).

Database migrations are run via `migrate/migrate` Docker containers pointed at `database/postgres/` and `database/neo4j/`. Elasticsearch index setup runs via `scripts/search_init.py`.
