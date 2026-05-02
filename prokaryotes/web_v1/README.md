# web_v1

`WebBase` is the base class for all web harnesses. It provides FastAPI setup, auth (login/register via Postgres), Redis-backed sessions, `GraphClient`/`SearchClient` lifecycle, partition reconciliation, and streaming response finalization. LLM-specific harnesses (`anthropic_v1/web_harness.py`, `openai_v1/web_harness.py`) extend it by adding an LLM client and a `/chat` route.

## Request lifecycle (web harness, single turn)

1. `POST /chat` arrives at `WebHarness.post_chat()`. `ChatConversation` (fields: `conversation_uuid`, optional `partition_uuid`, `messages`) is deserialized.

2. `WebBase.sync_context_partition()` produces the authoritative `ContextPartition` via three-tier reconciliation: Redis fast path → exact ES load → ancestor chain reconstruction. The returned partition's `items` contains only raw conversation turns — no system/developer item at position 0.

3. `post_chat()` assembles the system/developer `ContextPartitionItem` (core instructions, personality, tool guidance, runtime context, ancestor summaries per provider — see below) and inserts it at `context_partition.items[0]`.

4. `WebBase.stream_and_finalize()` wraps `stream_turn()`. It first yields `{"partition_uuid": "..."}` as the first NDJSON event so the client can track the active branch. Then it streams NDJSON events from `stream_turn()`: `{"text_delta": "..."}` per chunk, `{"context_pct": N}` after each LLM round.

5. Inside `stream_turn()`, tool calls are dispatched to `FunctionToolCallback.call()` after each round. Results are appended to the partition and fed back into the next LLM call. The loop repeats until no tool calls are produced or `max_tool_call_rounds` is reached.

6. After `stream_turn()` exhausts, `stream_and_finalize()` checks whether compaction was triggered. Two paths:
   - **Compaction path** (lock acquired): `finalize()` is awaited directly — not backgrounded, to prevent a race where a delayed finalize overwrites the compaction result. Then `{"compaction_pending": true}` is emitted and `_compact_partition()` is fired as a background task.
   - **Normal path**: `finalize()` runs as a background task via `background_and_forget()`.

7. `finalize()` calls `pop_system_message()` (strips `items[0]`), writes the partition to Redis keyed by `context_partition:{conversation_uuid}`, and calls `search_client.put_partition()` to upsert the Elasticsearch document.

## Extending WebBase

A new provider harness must implement three methods and follow their ordering constraints exactly.

### `init()` — synchronous setup

Call `super().init()` first. This creates `self.app`, registers all base routes, and initializes the Redis client, `GraphClient`, and `SearchClient`. Only after `super().init()` returns: initialize the LLM client, then register the `/chat` route via `self.app.add_api_route`. Registering before `super().init()` will fail because `self.app` does not exist yet.

```python
def init(self):
    super().init()
    self.llm_client.init_client()
    self.app.add_api_route("/chat", self.post_chat, methods=["POST"])
```

### `on_stop()` — asynchronous teardown

Call `await super().on_stop()` first. This closes `GraphClient` and `SearchClient`. Then close the LLM client. Reversing the order leaves search/graph clients open.

```python
async def on_stop(self):
    await super().on_stop()
    await self.llm_client.close()
```

### `post_chat()` — request handler

Must follow this shape, in order:

1. Authenticate via `load_session` and guard on empty `conversation.messages`.
2. Call `await self.sync_context_partition(conversation)` to get the authoritative partition.
3. Assemble the system/developer instruction item and insert it at `context_partition.items.insert(0, ...)`. See **Ancestor summary injection** below for the provider-specific difference.
4. Define an `on_usage` callback (receives `input_tokens`, `output_tokens`) that sets `pending_compaction[0] = True` when the context threshold is crossed. Use a single-element list — a plain `bool` would be an immutable copy unreachable from the closure.
5. Define a `compact_fn` coroutine that calls `_summarize_and_compact()`.
6. Return `StreamingResponse(self.stream_and_finalize(...), media_type="text/event-stream")`.

Do not call `finalize()` directly from `post_chat()`. `stream_and_finalize()` manages `finalize()` and the compaction lock; calling it directly bypasses the lock and causes a double-finalize race.

### `_summarize_and_compact()` — background compaction

Must produce a plain `str` summary using the provider's own API directly (not via `stream_turn()`). It is called from a background task after the streaming response has been finalized, so streaming is unavailable at this point.

## Ancestor summary injection

Ancestor summaries are injected differently per provider.

**Anthropic**: build the main system prompt from `get_core_instruction_parts(summaries=...)`, then personality, then the rest of the harness context. Do not include `ancestor_summaries` manually. `ContextPartition.to_anthropic_messages()` — called inside `stream_turn()` — appends them automatically as a trailing `# Compacted conversation summary` background-memory block after the core system instructions. Including them manually would duplicate them.

**OpenAI**: build the main developer prompt from `get_core_instruction_parts(summaries=...)`, then personality, then the rest of the harness context. `to_openai_input()` does not inject `ancestor_summaries`; append `context_partition.ancestor_summary_block()` as the final background-memory section when present.

```python
# Anthropic — summaries handled by to_anthropic_messages()
system_message_parts = []
system_message_parts.extend(system_message_utils.get_core_instruction_parts(
    summaries=bool(context_partition.ancestor_summaries)
))
system_message_parts.append("")
system_message_parts.extend(system_message_utils.get_personality_parts())
system_message_parts.append("")
system_message_parts.append("# Tool usage")
...
context_partition.items.insert(0, ContextPartitionItem(role="system", content="\n".join(system_message_parts)))

# OpenAI — summaries appended manually after the main instruction block
developer_message_parts = []
developer_message_parts.extend(system_message_utils.get_core_instruction_parts(
    summaries=bool(context_partition.ancestor_summaries)
))
developer_message_parts.append("")
developer_message_parts.extend(system_message_utils.get_personality_parts())
developer_message_parts.append("")
developer_message_parts.append("# Tool usage")
...
summary_block = context_partition.ancestor_summary_block()
if summary_block:
    developer_message_parts.append("")
    developer_message_parts.append(summary_block)
context_partition.items.insert(0, ContextPartitionItem(role="developer", content="\n".join(developer_message_parts)))
```
