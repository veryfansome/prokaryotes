# Conversation Model

## Overview

A `Conversation` is the single primitive for an external dialogue — one web chat session today, one Slack thread in the future. Single-user is the degenerate case of multi-user where the author cardinality is 1.

Three properties define the model:

1. **Source-ID-based reconciliation.** Each message carries a stable, server-assigned `source_id`. Reconciliation diffs an incoming message list against the stored snapshot by `source_id` and classifies the change (append, edit, delete, divergence). Position is irrelevant.
2. **External dialogue and turn execution are separate.** A `Conversation` is the append-only log of external messages (what users posted, what the bot posted). A `TurnExecution` is the LLM-internal log for one bot reply, holding tool-call records only. Intermediate narration the model produces between tool calls is transient and never persisted.
3. **Multi-author by default.** `ConversationMessage` carries `author_id`; the `user` vs `assistant` role is derived at projection time by comparing to the conversation's `bot_author_id`.

[Compaction](../compaction/README.md) — periodic summarization of old history so a conversation can run past the context window — is a lifecycle operation layered on this model; that doc owns the compaction-specific machinery.

---

## Core Concepts

### `Conversation`

A persistent snapshot of an external dialogue, identified by a stable `conversation_uuid`. A single `conversation_uuid` is a *DAG* of snapshots, each with its own `snapshot_uuid` — linear children via compaction, sibling branches via edit/regenerate (see [Branches and Snapshots](#branches-and-snapshots)).

A snapshot owns its ordered `ConversationMessage`s, the bot's `author_id`, the snapshot-DAG metadata (`parent_snapshot_uuid`, `ancestor_summaries`, `raw_message_start_index`), and the durable file-tool working memory (`working_file_windows` — see [file_tool/README.md](../file_tool/README.md)).

### `ConversationMessage`

One external dialogue message: `source_id` (identity *and* ordering key), `author_id` (opaque; bot vs human is derived against `Conversation.bot_author_id`), `content`, `display_name`, and the `deleted` / `edited` storage flags.

`source_id` is surface-assigned, sortable, and formatted `seconds.microseconds` so lexicographic order equals chronological order. The web syncer assigns one on first encounter; Slack supplies the message `ts`. Monotonicity within a conversation is enforced — a candidate `<=` the last assigned `source_id` is bumped by one microsecond.

### `TurnExecution` and `TurnItem`

`TurnExecution` is the LLM-internal log for one bot reply, stored in its own ES index keyed by the resulting bot message's `source_id`. It holds only `TurnItem`s — `function_call` / `function_call_output` records. The final assistant message itself lives on the `Conversation` (so external surfaces can see and edit it), not in the `TurnExecution`.

**Transient-narration invariant.** Intermediate assistant text between tool calls ("let me check…") is *not* persisted. The model writes it as in-the-moment scratch work; replaying it on a later turn would condition the model on past-tense versions of itself. The LLM client keeps it in a streaming buffer during the active turn and discards it at finalization — only `function_call` / `function_call_output` items are committed. `tests/unit_tests/test_anthropic_v1.py` and `test_openai_v1.py` guard this.

---

## Data Model

### Pydantic models — `prokaryotes/conversation_v1/models.py`

| Model | Role |
|---|---|
| `Conversation` | A snapshot: `conversation_uuid`, `snapshot_uuid`, `parent_snapshot_uuid`, `bot_author_id`, `ancestor_summaries`, `raw_message_start_index`, `messages`, `working_file_windows`. |
| `WorkingFileWindow` | Durable file-tool live-window state: `window_id`, `path`, `status`, `revision`, `view_start_line`, `view_end_line`, `requested_end_line`, `line_count`, `origin_call_ids`, `source_kind`, `rendered_output`. See [file_tool/README.md](../file_tool/README.md). |
| `ConversationMessage` | External message: `source_id`, `author_id`, `content`, `display_name`, `deleted`, `edited`. |
| `TurnExecution` | Per-bot-reply tool log: `conversation_uuid`, `bot_message_source_id`, `items`, `completed`. |
| `TurnItem` | One `function_call` / `function_call_output` record. Carries `prokaryotes_annotations` (file-tool annotations live here). |
| `ProjectedItem` | LLM-bound projection — the bridge from storage to provider wire format. |
| `NormalizedMessage` | What `reconcile()` operates on — fully-populated incoming message. |

Wire-format models live in `prokaryotes/api_v1/models.py`: `IncomingMessage` / `IncomingConversation` (the `POST /chat` body) and `CompactionStatusResponse`.

### Elasticsearch

Two indices, both `dynamic: strict` — `prokaryotes/search_v1/conversations.py`:

- **`conversations`** — one document per snapshot. Carries the model fields plus `compaction_state`, `is_compacted`, `summary`, `boundary_hash` / `tail_hash`, and a full-text-indexed `message_content`. `messages_json` is retained even after `is_compacted=True` so the [assistant-message guardrail](#assistant-message-guardrail) can read compacted ancestors' per-message identity.
- **`turn-executions`** — one document per bot reply that involved tool calls, keyed `conversation_uuid:bot_message_source_id`.

`scripts/search_init.py` bootstraps both indices (plus `topics`).

### Redis

| Key | Purpose |
|---|---|
| `conversation:{conversation_uuid}` | Cached active `Conversation` snapshot (one per conversation). |
| `assistant_index:{conversation_uuid}` | `{bot_source_id: content_hash}` for the assistant-message guardrail. |
| `compaction_lock:{conversation_uuid}` | Compaction CAS lock. |
| `compaction_status:{pending_snapshot_uuid}` | `/compaction-status` relabel target. |

TTLs default to `CONVERSATION_CACHE_EXPIRY_SECONDS` (7 days).

---

## Reconciliation

`reconcile(stored, incoming)` in `prokaryotes/conversation_v1/reconcile.py` diffs incoming `NormalizedMessage`s against `stored.messages` by `source_id` and returns a classification plus an operation list — it does not apply anything:

- **`match`** — no operations.
- **`append`** — every stored non-deleted `source_id` is in incoming with matching content, and incoming has additional `source_id`s at the end.
- **`edit`** — every stored `source_id` is in incoming, but at least one content differs.
- **`delete`** — stored `source_id`s missing from incoming, with no fresh `source_id`s and no content changes.
- **`divergence`** — anything else (e.g. an edit/delete that isn't a pure trailing change).

The **syncer** decides how to apply per surface. The web syncer (`ConversationSyncer._apply_result`) mutates in place for `match` / `append` and branches a new snapshot via `_apply_divergence` for `edit` / `delete` / `divergence` — this preserves the snapshot-DAG branch contract (the original snapshot stays intact in ES). `SlackApplyPolicy` overrides this to mutate in place for all classifications, since Slack threads are linear and authoritative; its delete path also re-keys a tombstoned bot's `TurnExecution` to the next non-tombstoned bot in the run.

### Three-tier load — `ConversationSyncer.sync_conversation`

`prokaryotes/context_v1/conversation_sync.py` reconciles the active snapshot per request:

1. **Redis fast path.** The cache "follows" the client when its `snapshot_uuid` equals the client's *or its parent* (a post-compaction relabel the client hasn't applied yet).
2. **Exact ES load.** Fetch the doc at the client's `snapshot_uuid` by `_id`. A compacted doc is forwarded to chain rebuild — a compacted snapshot can't be the active one directly.
3. **Ancestor-chain rebuild.** Walk `parent_snapshot_uuid` links; validate a compacted ancestor's `boundary_hash` against a prefix of incoming. No matching ancestor → a fresh root `Conversation`.

After load, the post-load pipeline runs: `_split_compacted_prefix` → `_detect_unacknowledged_bot_messages` → `_assign_source_ids` → `reconcile` → `_apply_result` → `_cache_and_persist_conversation`.

---

## Projection

`project_for_llm` (`prokaryotes/conversation_v1/project.py`) is the single bridge from storage to the LLM API and the only place role assignment, display-name prefixing, leading-block emission, and consecutive-same-role text merging happen. Everything else in the system is role-agnostic.

- Role is derived: `author_id == bot_author_id` → `assistant`, else `user`.
- In a multi-author conversation (more than one distinct human `author_id`), each human turn is prefixed `<display_name> `.
- Each bot message's `TurnExecution.items` are interleaved immediately before it. Historical file-tool outputs annotated `file_tool.persistence="working_file"` (and their paired `function_call`s) are filtered out — their durable relevance lives on `Conversation.working_file_windows`, which projects as a leading user-role block; frozen edit records (`persistence="history"`) ride the transcript forward.
- Consecutive same-role `message` items are merged (`\n\n`-joined) — this is what keeps the OpenAI Responses API's alternation requirement satisfied.

LLM clients translate `list[ProjectedItem]` to their wire format (`_items_to_anthropic_messages` / `_items_to_openai_input`). The instruction (system/developer) message is injected by the harness at position 0 *after* projection — it is not part of conversation storage and carries only trusted content (core instructions, runtime context, tool usage, personality, user context).

`FileTool` reads and mutates `Conversation.working_file_windows` directly via a `working_file_provider` callable (typically `lambda: conversation.working_file_windows`). Per-turn refresh runs at turn start via `reconcile_working_files(conversation.working_file_windows, ...)` before any FileTool call — see [file_tool/README.md](../file_tool/README.md) for the read/write/reconcile lifecycle.

### Background-context blocks

Untrusted-or-bot-summarized context projects into a **leading user-role slot** ahead of the conversation walk, using a shared XML delimiter convention:

```
<tag trust="…">
…body…
</tag>
```

Each block escapes any literal closing tag inside the body (`</tag>` → `<\/tag>`) so the structural boundary is unambiguous. Three sources flow through this slot, in this emission order:

- **Ancestor summaries.** `Conversation.ancestor_summary_block()` materializes `ancestor_summaries` as `<compacted_summary trust="bot-summarized">…</compacted_summary>` whenever the list is non-empty. The [compaction lifecycle](../compaction/README.md) owns the *content*; projection owns the *placement*.
- **Caller-supplied blocks.** `project_for_llm`'s `leading_context_blocks: list[str] | None = None` parameter accepts pre-delimited block strings (e.g. a future Slack `<channel_prelude trust="untrusted-user-data">…`). They emit between the summary and the working-files block, in list order. The parameter is `list[str]`, not `list[ProjectedItem]` — callers cannot inject assistant or function items at the head and break the merge invariant.
- **Working files.** `Conversation.working_files_block()` materializes `working_file_windows` as `<working_files trust="file-content">…</working_files>` whenever the list is non-empty. The [file_tool lifecycle](../file_tool/README.md) owns the *content*; projection owns the *placement*. Closing-tag escape applies the same way as the other blocks (`</working_files>` → `<\/working_files>`).

When the first stored message is user-role, the same-role merge collapses the summary, any caller-supplied blocks, the working-files block, and that first user message into one wire-level user-role message. Anthropic sees one user message with one text block whose body carries the merged content; OpenAI Responses sees one user input message with the same merged payload. When the first stored message is assistant-role (e.g. a snapshot that starts with a bot reply), the merge does not cross roles — the leading blocks form their own wire-level user-role message ahead of the assistant turn. Either way, the XML delimiters carry the structural separation inside any merged block; no wire-level non-merge guarantee is needed.

This slot sits below the instruction (system/developer) message but ahead of the conversation walk, so both providers grant it less authority than system/developer instructions while still seeing it before any user turn — the trust placement the convention exists for.

---

## Branches and Snapshots

A `conversation_uuid` is a DAG of snapshots. Two edge kinds, both via `parent_snapshot_uuid`, distinguished by the parent's state:

- **Compaction edge** (linear) — the parent is `is_compacted=true`; its raw messages were summarized into the child's `ancestor_summaries`.
- **Branch edge** (sibling) — the parent is non-compacted; the child holds a tail that diverged from the parent's. Web edit/regenerate produces these.

### Divergence creates a new snapshot

When the web syncer applies `edit` / `delete` / `divergence`, `_apply_divergence` roots a new snapshot at the shared prefix. The original snapshot is untouched in ES — prior branches survive server-side.

- **Case A — divergence within the parent's raw window** (the common edit/regenerate path). The child inherits the parent's `ancestor_summaries` and `raw_message_start_index` verbatim, and filters `working_file_windows` with the two-gate active-path + origin filter; see [file_tool/README.md](../file_tool/README.md#branch-divergence) for the filter details.
- **Case B — divergence before the raw window** (editing a message that was compacted away). The compacted-prefix split fails, the syncer discards the loaded snapshot, and a fresh root `Conversation` is built from incoming alone — no inherited summaries, `raw_message_start_index=0`.

Child branch `messages` are sorted by `source_id`, so the stored list stays in chronological order regardless of request shape.

### Client-side branch model — `scripts/static/`

The browser keeps a `messageTree`. Edit and regenerate create *new* sibling nodes; each node stores its server-assigned `source_id` and the `snapshot_uuid` of the snapshot it belongs to. `conversation_client.js` holds the pure protocol primitives (`applyHandshake`, `applyBotMessage`, `relabelSnapshotUuid`, `applyResyncHandshake`, `buildRequestMessages`); `ui.js` wires them into the chat UI.

Because branches share a prefix, a shared node belongs to several snapshots and `applyHandshake` restamps it each turn. The request's `snapshot_uuid` is therefore anchored on the **leaf of the branch the user is viewing**, captured before any tree mutation — not on the edit/regenerate parent, which is a shared node a sibling branch may have restamped.

Cross-session branch survival is out of scope: snapshots persist in ES, but the client's per-node `snapshot_uuid` mapping dies with the page session, so a reload resumes only the most recently active branch.

---

## Web Wire Protocol

### Request — `POST /chat`

`IncomingConversation`: `{conversation_uuid, snapshot_uuid?, messages: [{role, content, source_id?}]}`. `source_id` is omitted for newly-authored messages; the syncer assigns one. The route normalizes each message into a `NormalizedMessage` using session info (`author_id` from `chat_user.id`, `display_name` from `full_name`).

### Response stream (NDJSON) — event ordering is contractual

- **Handshake** — always first. `{snapshot_uuid, source_id_assignments: [{client_index, source_id}]}`. The client stamps both `source_id` and `snapshot_uuid` onto every submitted user node. Invariant: every node with a `source_id` also has a `snapshot_uuid`.
- `context_pct`, `text_delta`, `tool_call`, `progress_message` flow during the turn.
- **`bot_message`** — `{bot_message: {source_id}}`, emitted exactly once after the final assistant message commits. The client creates the assistant node only on this event; a stream that aborts before it creates no assistant node.
- **`compaction_pending`** — optional last event; see [compaction](../compaction/README.md).

### Stream-loss recovery

- **Pre-commit.** If the stream aborts after the handshake but before `bot_message`, the user node already carries the new `snapshot_uuid`, so a retry extends the branch instead of re-diverging.
- **Post-commit (resync).** If the server committed a bot message the client never saw, the next POST looks like a fork. The server detects the trailing-bot-skip pattern *before* reconcile, emits a resync handshake with `unacknowledged_bot_messages`, and closes the stream without an LLM call. The client reconstructs the missing assistant nodes; send-from-leaf auto-retries, edit/regenerate pops the pending message back to a draft.

### Assistant-message guardrail

Web clients only ever author `user` messages. `validate_assistant_messages` runs before `sync_conversation` and rejects (4xx) any `role="assistant"` entry whose `source_id` is unknown to the conversation DAG, or whose content hash doesn't match storage. Recognition is DAG-scoped — it reads compacted ancestors' retained `messages_json` — so the Case B retry path, which legitimately echoes compacted ancestors' assistant `source_id`s, is accepted.

---

## Surface Mappings

| Concept | Web | Slack |
|---|---|---|
| `conversation_uuid` | Server-generated on first message | `uuid5(SLACK_NS, team:channel:thread)` |
| `bot_author_id` | Constant `"__bot__"` | `bot_user_id` |
| `source_id` | Syncer-assigned `seconds.microseconds` | Slack `ts` |
| `author_id` | `chat_user.id` | Slack `user` |
| Apply policy | Branch on divergence | Overwrite in place |

`SlackApplyPolicy` lives in `prokaryotes/context_v1/conversation_sync.py` and is mixed into `SlackBase` (`prokaryotes/slack_v1/__init__.py`); `SlackHarness` (`prokaryotes/harness_v1/slack.py`) is the concrete subclass.

---

## Out of Scope

- Cross-surface conversation merging (web + Slack sharing one history).
- Per-message rich content (`content` is intentionally `str`).
- Cross-session branch survival (the ES DAG could support it later).
- Branch-prefix deduplication (sibling snapshots duplicate the shared prefix).
- `TurnExecution` garbage collection (orphaned turns linger in ES).

---

## Relevant Code Files

| File | Role |
|---|---|
| `prokaryotes/conversation_v1/models.py` | `Conversation`, `ConversationMessage`, `TurnExecution`, `TurnItem`, `ProjectedItem`, `NormalizedMessage`, `compute_boundary_hash` / `compute_tail_hash`. |
| `prokaryotes/conversation_v1/reconcile.py` | Source-ID-keyed `reconcile()`. |
| `prokaryotes/conversation_v1/project.py` | `project_for_llm` (role assignment, leading-block emission, historical working-file-output filter, same-role merge). |
| `prokaryotes/context_v1/conversation_sync.py` | `ConversationSyncer` (three-tier load, post-load pipeline, `_apply_divergence`, guardrail), `SlackApplyPolicy`. |
| `prokaryotes/api_v1/models.py` | `IncomingMessage` / `IncomingConversation`, `CompactionStatusResponse`. |
| `prokaryotes/search_v1/conversations.py` | `ConversationSearcher`: ES CRUD + index mappings. |
| `prokaryotes/harness_v1/base.py` / `web.py` | `stream_and_finalize`, `finalize_turn`; the `/chat` route, projection, instruction assembly. |
| `prokaryotes/anthropic_v1/`, `openai_v1/` | `ProjectedItem` → provider wire format; the transient-narration buffer. |
| `scripts/static/conversation_client.js`, `ui.js` | Client-side branch model, wire protocol, branch-anchored `snapshot_uuid` selection. |
| `tests/unit_tests/test_reconcile.py`, `test_conversation_*.py`, `test_project_for_llm.py` | Reconcile matrix, syncer, projection. |
| `tests/integration_tests/tier_b/test_unified_web_flow.py`, `test_unified_multi_author_flow.py` | Edit/regenerate branching and multi-author projection end to end. |
