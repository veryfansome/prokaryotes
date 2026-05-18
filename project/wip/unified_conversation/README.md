# Unified Conversation Model

## Goals

Replace `ContextPartition` with a single conversation primitive that serves both single-user (web) and multi-user (Slack, future platforms) surfaces. Single-user is the degenerate case of multi-user where the author cardinality is 1.

Three things change:

1. **Source-ID-based reconciliation** instead of position-based. Each message carries a stable, surface-provided ID. Reconcile classifies the change (append, edit, delete, divergence); the syncer applies per surface — Slack reconciles a thread in place, web edit/regenerate spawns a new branch snapshot. Position is irrelevant — non-mention chatter between two `@`-mentions can no longer be mistaken for a retry of the last user message.
2. **External dialogue and turn execution are separate constructs.** A `Conversation` is the append-only log of external messages (what users posted, what the bot posted). A `TurnExecution` is the LLM-internal log for one bot reply, holding tool calls and tool results only — intermediate narration the model produces between tool calls is transient and not persisted (preserving today's invariant; see [`TurnExecution`](#turnexecution)). Reconciliation only operates on the external log; turn execution is keyed by the resulting bot message's source ID and is never compared against the external surface.
3. **Multi-author by default.** `ConversationMessage` carries `author_id`; role (`user` vs `assistant`) is derived at projection time by comparing to the conversation's bot identity. The "<display_name> " prefix the Slack design proposes for multi-human threads becomes a property of the projection step, not the storage layer.

Out of scope: backwards compatibility with existing `ContextPartition` ES data — the app is not in production and a clean break is fine.

---

## Observed Repository Context

- `prokaryotes/api_v1/models.py` defines `ContextPartition` (provider-agnostic conversation history with compaction-chain fields), `ContextPartitionItem` (a single envelope for `message`, `function_call`, and `function_call_output`), and `ChatConversation`/`ChatMessage` (the HTTP request payload from the web client). `ContextPartition.message_items_for_sync()` already filters items down to `type == "message" and role in {"user", "assistant"}` — the latent split between "conversation messages" and "tool items" exists today; this design makes it explicit.
- `ContextPartition.find_context_divergence()` is positional and content-based: it walks `zip(context_items, conversation_items)`, returns the first index where content differs. This is what we replace with source-ID matching.
- `ContextPartition.to_anthropic_messages()` already coalesces consecutive same-role items via its `current_role` accumulator, so the Anthropic path is robust to multiple consecutive `user` items. `to_openai_input()` passes items through raw and is **not** robust to the same — Responses API will reject non-alternating sequences. Both projections need to live in the same place under the new model.
- `prokaryotes/context_v1/` already owns the shared lifecycle (`PartitionSyncer`, `PartitionCompactor`, `get_redis_client`). The `HarnessBase` extraction (separate wip) further decouples this from FastAPI. Under this design, primitives (models, reconciliation, projection) live in a new `conversation_v1/` module; `context_v1/` continues to own lifecycle, renamed and retyped against the new primitives.
- `prokaryotes/search_v1/context_partitions.py` defines the `context-partitions` ES index with `boundary_hash` / `tail_hash` for compaction-chain validation and a JSON-blob `items_json` field for the partition body. The index gets replaced; the hashing approach (boundary validation, tail heuristic fallback) survives but moves up a layer to operate on conversation messages.
- `WebHarness.post_chat` (in `prokaryotes/harness_v1/web.py`) is the single web entry point: it accepts a `ChatConversation`, calls `sync_context_partition` → builds an instruction message → streams via `stream_and_finalize`. This is the migration target on the web side.

---

## Core Concepts

### `Conversation`

A `Conversation` is a persistent snapshot of an external dialogue. The dialogue is per-surface (one Slack thread, one web chat session) and identified by a deterministic `conversation_uuid`; each snapshot has its own `snapshot_uuid`, and a single `conversation_uuid` typically has many snapshots over its life (see [Branches and Snapshots](#branches-and-snapshots)). A snapshot owns:

- A list of `ConversationMessage`s, ordered by `source_id` ascending.
- The bot's `author_id` in this conversation (so projection can derive role).
- Snapshot-DAG metadata (parent `snapshot_uuid`, ancestor summaries, raw window start).
- Tool-state lifted across compaction (`lifted_turn_items` + `lifted_anchor_source_id`) — see [Live windows across compaction](#live-windows-across-compaction).

```python
class Conversation(BaseModel):
    conversation_uuid: str
    snapshot_uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))  # was partition_uuid
    parent_snapshot_uuid: str | None = None
    bot_author_id: str
    ancestor_summaries: list[str] = Field(default_factory=list)
    lifted_turn_items: list[TurnItem] = Field(default_factory=list)
    lifted_anchor_source_id: str | None = None
    raw_message_start_index: int = 0
    messages: list[ConversationMessage] = Field(default_factory=list)
```

Naming: `snapshot_uuid` replaces `partition_uuid` as the per-snapshot identifier — a single `conversation_uuid` is a DAG of `Conversation` snapshots, generalizing today's linear partition chain to also cover sibling branches from edit/regenerate (see [Branches and Snapshots](#branches-and-snapshots)).

### `ConversationMessage`

```python
class ConversationMessage(BaseModel):
    source_id: str        # stable, surface-assigned, sortable; primary key AND ordering key
    author_id: str        # opaque; bot vs human distinguished against Conversation.bot_author_id
    content: str
    display_name: str | None = None   # for "<name> " prefixing in multi-author projections
    deleted: bool = False             # tombstone for surface-level deletes
    edited: bool = False              # informational; latest content is in `content`
```

Notes on the field choices:

- `source_id` is opaque to the harness but guaranteed sortable. Slack supplies the message `ts` (format: `seconds.microseconds`); the web syncer assigns one on first encounter using the same format from server `time.time()`. Lexicographic sort equals chronological sort, so `source_id` serves as both identity and ordering — no separate `sequence` field is needed.
- Monotonicity within a conversation is enforced by the syncer: if a would-be assignment is `<=` the last `source_id` in this conversation, bump by one microsecond. Slack handles this on their side; for web it's the syncer's job.
- `author_id` is also opaque per-surface. Slack uses `user_id` (workspace-scoped, fine because `conversation_uuid` already encodes the workspace). Web uses `chat_user.id` (as a string). The bot's `author_id` is whatever the surface chooses — Slack uses `bot_user_id`, web uses a constant like `"__bot__"`.
- `display_name` is denormalized into the message because Slack display names can change, and a fixed-at-write snapshot is good enough for projections.
- `deleted` is a tombstone — we keep the message in storage to preserve `source_id` uniqueness and audit trail; the projection skips it.

### `TurnExecution`

The internal log for one bot reply. Stored in a separate ES index, keyed by the resulting `ConversationMessage.source_id`. `TurnItem` is a renamed, narrower descendant of `ContextPartitionItem` — it holds only tool-call records (no `"message"` variant, no `content` field):

```python
class TurnExecution(BaseModel):
    conversation_uuid: str
    bot_message_source_id: str        # foreign key: which ConversationMessage this turn produced
    items: list[TurnItem]             # function_call / function_call_output only — see invariant below
    completed: bool = False           # marks the final assistant message has been finalized into the Conversation


class TurnItem(BaseModel):
    """Renamed from ContextPartitionItem. Holds only function_call / function_call_output records;
    the final assistant message lives in the Conversation, and intermediate assistant text is
    transient (see invariant below)."""
    arguments: str | None = None
    call_id: str | None = None
    id: str | None = None
    name: str | None = None
    output: str | None = None
    prokaryotes_annotations: dict[str, str] | None = None
    type: Literal["function_call", "function_call_output"] = "function_call"
    status: Literal["in_progress", "completed", "incomplete"] | None = None
```

The final assistant message lives in the `Conversation` (so external surfaces see and can edit it), not in the `TurnExecution`. The `TurnExecution` holds only the tool-call records.

**Transient-narration invariant (preserved from today's behavior).** Intermediate assistant text between tool calls — the "let me check…" / "looking at the result, I think…" narration the model produces alongside tool use — is *not* persisted to `TurnExecution`. The model writes this text expecting it to be in-the-moment scratch work, not durable output; replaying it on a later turn would condition the model on past-tense versions of itself "about to" do things it already did. The transient text is kept in the LLM client's streaming working memory during the active turn (so mid-tool-use-loop rounds can replay it within the same turn), and discarded at turn finalization. Only `function_call` / `function_call_output` items are committed to `TurnExecution`. Same effective behavior as today (`prokaryotes/README.md` documents the invariant; `tests/unit_tests/test_anthropic_v1.py` and `tests/unit_tests/test_openai_v1.py` guard it). Cleaner separation in the new model because the streaming buffer and the durable `TurnExecution` are different objects, so "strip before persisting" becomes just "don't copy."

This split makes compaction's job cleaner: it operates on `Conversation.messages` (external dialogue) and discards/summarizes the `TurnExecution`s associated with the compacted window. Tool-call internals from old turns never get replayed past compaction.

**Multi-post bot turns** (Slack only at v1). A single LLM turn whose final assistant text exceeds Slack's per-message size limit (~3500 chars) is split by the streamer into N continuation posts. The Slack thread ends up with N consecutive bot posts that all came from one LLM call — and therefore one `TurnExecution` worth of tool-call records. (Web doesn't hit this — there's no per-message size limit on the HTTP stream. The rule applies to *all* Slack contexts uniformly: channels, mpim, and 1:1 DMs, since they all share the same per-post limit.)

Ownership rule: the **first** post in the consecutive bot run (lowest `source_id` in the run) owns the `TurnExecution`. Subsequent posts in the same run are stored as additional `ConversationMessage`s with their own `source_id`s but have no associated `TurnExecution`. Projection's per-bot lookup finds the tool items at the first post; same-role text merging then concatenates the split bodies into one assistant block with the tool items preceding.

**Tombstone re-keying.** If the `TurnExecution`'s owner `ConversationMessage` is later tombstoned (Slack `message_deleted` — typically a workspace admin cleanup or a retention policy reaping old messages), the syncer's `message_deleted` handler re-keys the `TurnExecution`'s `bot_message_source_id` to the next non-tombstoned bot `ConversationMessage` in the *same consecutive run* (the chain of bot messages between the surrounding user messages, with tombstoned entries still counted for run-membership but skipped for selection). The lift anchor (`Conversation.lifted_anchor_source_id`) follows the same re-key rule when its target is tombstoned. If every bot in the run is tombstoned, the `TurnExecution` is orphaned (eligible for sweep on next compaction GC) and the lift anchor falls to `None`.

We fold this into v1 rather than deferring it because the implementation is trivial — a handful of lines in the `message_deleted` handler walking the conversation messages around the deleted `source_id` to find the next bot in the run, and one ES update for the affected `TurnExecution` (or `Conversation` doc for the anchor). The cost of *not* doing it is invisible-but-real: a deleted first-split-post means the next projection silently drops that turn's tool items (and any file-tool lift attached to it), with no signal back to the user. Cheap to do, cheap to test, and it removes a "known minor limitation" footnote.

---

## Data Model Changes

### Pydantic models

| Today | New | Notes |
|---|---|---|
| `ChatConversation` | `IncomingConversation` | Wire payload from web client. Carries `conversation_uuid`, optional `snapshot_uuid` (renamed from `partition_uuid`), and a `messages` list of `IncomingMessage`. Server-only fields like `bot_author_id` are not echoed by the client. |
| `ChatMessage` | `IncomingMessage` | **Web wire format only** (Slack consumes Slack events directly). Shape: `{role: "user" \| "assistant", content: str, source_id?: str}`. `source_id` is omitted on newly-authored messages (typed fresh, edit, regenerate); the syncer assigns one on first encounter. `author_id` and `display_name` are *not* on the wire — see `NormalizedMessage`. |
| _(new)_ | `NormalizedMessage` | What `reconcile()` operates on. Built by the syncer from `IncomingMessage` + session info (web) or from a Slack event (Slack). Carries `{source_id, author_id, content, display_name?}` with all fields populated. Storage-side flags (`deleted` / `edited`) live on `ConversationMessage` only and are never on incoming. |
| `ContextPartition` | `Conversation` | See above. Holds only external messages. |
| `ContextPartitionItem` | Split into `ConversationMessage` (external) and `TurnItem` (internal). | The `prokaryotes_annotations` extensibility hook moves to `TurnItem` since file-tool annotations live on tool-related items. |
| `compute_boundary_hash` | unchanged shape | Operates on `Conversation.messages` (filtered to non-deleted); the per-item payload becomes `{author_id, content}` because `role` is no longer stored — it's derived at projection time. |
| `compute_tail_hash` | unchanged shape | Operates on the last N non-bot messages by `author_id != bot_author_id`. |
| `CompactionStatusResponse` | unchanged shape, renamed field | `partition_uuid: str \| None` becomes `snapshot_uuid: str \| None`. Semantics unchanged: populated only when the swap committed a *direct child* of the polled-for pending id. |

### Elasticsearch

Two indices, replacing `context-partitions`:

**`conversations`** — one document per `Conversation` snapshot (compaction children and branch siblings alike).

```python
conversation_mappings = {
    "dynamic": "strict",
    "properties": {
        "snapshot_uuid": {"type": "keyword"},        # was partition_uuid
        "conversation_uuid": {"type": "keyword"},      # stable across the snapshot DAG
        "parent_snapshot_uuid": {"type": "keyword"},
        "bot_author_id": {"type": "keyword"},
        "compaction_state": {"type": "keyword"},
        "compaction_attempt_uuid": {"type": "keyword"},
        "is_compacted": {"type": "boolean"},
        "summary": {"type": "text", "analyzer": "standard"},
        "ancestor_summaries": {"type": "keyword", "index": False, "doc_values": False},
        "lifted_turn_items_json": {"type": "keyword", "index": False, "doc_values": False},
        "lifted_anchor_source_id": {"type": "keyword"},
        "messages_json": {"type": "keyword", "index": False, "doc_values": False},
        "message_content": {"type": "text", "analyzer": "standard"},
        "raw_message_start_index": {"type": "integer"},
        "boundary_message_count": {"type": "integer"},
        "boundary_user_count": {"type": "integer"},
        "boundary_hash": {"type": "keyword"},
        "tail_hash": {"type": "keyword"},
        "dt_created": {"type": "date"},
        "dt_modified": {"type": "date"},
    },
}
```

**`turn-executions`** — one document per bot reply that involved tool calls. A pure text completion (no tool calls at all) needs no turn-execution doc; `ThinkTool` and other tools each produce `function_call` / `function_call_output` items and so do require one.

```python
turn_execution_mappings = {
    "dynamic": "strict",
    "properties": {
        "bot_message_source_id": {"type": "keyword"},   # primary key
        "conversation_uuid": {"type": "keyword"},
        "items_json": {"type": "keyword", "index": False, "doc_values": False},
        "completed": {"type": "boolean"},
        "dt_created": {"type": "date"},
        "dt_modified": {"type": "date"},
    },
}
```

`turn-executions` are looked up only when projecting a `Conversation` for the LLM — and only for the most recent turn(s) inside the raw window, since compacted messages don't need their tool-call history. Cheap to query: `terms` filter on a small list of `bot_message_source_id`s.

### Redis

Keys:

| Today | New | Purpose |
|---|---|---|
| `context_partition:{conversation_uuid}` | `conversation:{conversation_uuid}` | Cached active `Conversation` snapshot. |
| `compaction_lock:{conversation_uuid}` | unchanged | Compaction CAS lock. |
| `compaction_status:{partition_uuid}` | `compaction_status:{snapshot_uuid}` | Polling endpoint state. Key is the *pending* (pre-swap) `snapshot_uuid`; value is the committed child's `snapshot_uuid` (the relabel target) or a tombstone when the swap concluded without a child commit. Written by the syncer at CAS-commit time; TTL matches `CONVERSATION_CACHE_EXPIRY_SECONDS` so a client returning after a long idle still finds the relabel target. Read by `GET /compaction-status` — see [Compaction completion and client relabel](#compaction-completion-and-client-relabel). |
| _(none)_ | `turn_execution:{bot_message_source_id}` | Cached `TurnExecution` for the most recent uncommitted turn. Optional — ES is authoritative. |

TTL semantics carry over unchanged.

---

## Reconciliation

The signature shifts from "compare `ChatConversation` to `ContextPartition` and find divergence" to "diff incoming messages against stored conversation by `source_id`":

```python
def reconcile(
    stored: Conversation,
    incoming: list[NormalizedMessage],
) -> ReconcileResult:
    """Diff incoming against stored by source_id and classify the changes.

    Preconditions: every incoming entry is a fully-populated `NormalizedMessage`
    (source_id assigned, author_id resolved). The syncer translates the raw surface
    payload — web `IncomingMessage` + session, or a Slack event — into
    `NormalizedMessage` and assigns source_ids to any newly-authored entries before
    calling reconcile. Storage-side flags (`deleted`, `edited`) live only on
    `ConversationMessage` and are never on incoming.

    Classifications (returned, not applied):
    - Append: incoming source_id is not in stored.
    - Edit: matching source_id exists in stored but content differs.
    - Delete: stored has a non-deleted source_id not in incoming.
    - Divergence: an Edit or Delete falls before the last shared source_id
        — i.e., the incoming list is not a pure-append extension of stored.

    No position-based truncation. Messages are sorted by `source_id` (lexicographic =
    chronological under the `seconds.microseconds` format).
    """
```

`ReconcileResult` carries the classification and the operation list. **The syncer**, not `reconcile` itself, decides how to apply: surfaces with different semantics (Slack in-place, web branch-on-divergence) consume the same classification and apply differently. See [Branches and Snapshots](#branches-and-snapshots).

Notable consequences:

- **In-place edits** (Slack `message_changed`) are first-class. Matching `source_id` with different content → server overwrites in place on the same `snapshot_uuid`; later messages are unaffected. Whether to invalidate the bot's subsequent turn-executions is a *policy* decision on the harness side; for v1 we leave them alone and the next bot turn sees the edited history naturally.
- **Divergence** (web edit/regenerate, where incoming includes new `source_id`s that didn't exist in stored and omits stored `source_id`s that the client has navigated away from) does *not* truncate the stored snapshot in place. The syncer creates a *new* snapshot (with a fresh `snapshot_uuid`) rooted at the shared prefix; the original snapshot stays in ES intact. This preserves prior branches server-side, in addition to the client's in-memory tree.
- **Slack-style deletes** (`message_deleted`): the stored source_id is marked `deleted=True`. Tombstones preserve uniqueness and audit trail; projection skips them.
- **Tool-execution preservation**: when a later user message arrives on the active branch, the stored `Conversation` is appended to. The most recent bot *turn*'s `TurnExecution` survives untouched (keyed to the first post of that turn's run — see [Multi-post bot turns](#turnexecution)). The fragility the Slack design surfaced — "non-mention chatter looks like a retry, throw away tool calls" — does not exist in this model.
- **Concurrent turns** (two background tasks racing on the same `conversation_uuid`): each does its own reconcile-and-append in sequence under a per-conversation `asyncio.Lock` held at the harness level, not at storage. Relevant only to surfaces where the harness can receive concurrent triggers for one conversation (e.g. Slack, when two users `@`-mention in the same thread); the web `/chat` route serializes by HTTP request.

### Cold-Redis recovery

Replaces the `find_latest_active_partition_uuid` query proposed in the Slack wip (the existing `find_partition_by_tail_hash` heuristic survives separately under a renamed `find_conversation_by_tail_hash`):

```python
async def find_latest_active_snapshot_uuid(self, conversation_uuid: str) -> str | None:
    """Return the snapshot_uuid of the most recently modified non-compacted snapshot
    for this conversation_uuid. Used when Redis has evicted the cache and the harness
    needs a head to start the chain walk from."""
```

ES query: `term: conversation_uuid` AND `term: is_compacted=false` AND `compaction_state in {committed, missing}`, sort `dt_modified desc`, size 1. Identical to the query the Slack wip proposed; just renamed.

---

## Projection

`project_for_llm` is the bridge from storage to the LLM API and the single place where role assignment, display-name prefixing, and consecutive same-role text merging happen. Everything else in the system is role-agnostic.

```python
class ProjectedItem(BaseModel):
    type: Literal["message", "function_call", "function_call_output"] = "message"
    role: Literal["user", "assistant", "system", "developer"] | None = None
    content: str | None = None
    # function_call / function_call_output fields:
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    output: str | None = None


def project_for_llm(
    conversation: Conversation,
    historical_turns: dict[str, TurnExecution] | None = None,
) -> list[ProjectedItem]:
    historical_turns = historical_turns or {}
    distinct_human_authors = {
        m.author_id for m in conversation.messages
        if not m.deleted and m.author_id != conversation.bot_author_id
    }
    needs_prefix = len(distinct_human_authors) > 1

    result: list[ProjectedItem] = []
    for msg in sorted(conversation.messages, key=lambda m: m.source_id):
        if msg.deleted:
            continue
        is_bot = msg.author_id == conversation.bot_author_id

        if is_bot:
            # When we reach the anchor bot message, emit lifted_turn_items immediately
            # before this turn's tool round — preserves today's lift_active_live_windows
            # placement (after leading user prefix, adjacent to first file activity).
            if msg.source_id == conversation.lifted_anchor_source_id:
                result.extend(_turn_items_to_projected(conversation.lifted_turn_items))
            # Interleave the historical turn's tool items before the final bot message.
            turn = historical_turns.get(msg.source_id)
            if turn:
                result.extend(_turn_items_to_projected(turn.items))
            result.append(ProjectedItem(type="message", role="assistant", content=msg.content))
        else:
            content = msg.content
            if needs_prefix and msg.display_name:
                content = f"<{msg.display_name}> {content}"
            result.append(ProjectedItem(type="message", role="user", content=content))

    return _merge_consecutive_same_role(result)


def _merge_consecutive_same_role(items: list[ProjectedItem]) -> list[ProjectedItem]:
    """Join consecutive type=message items with the same role, content joined by '\n\n'.
    Function-call items break the merge run."""
```

`project_for_llm` builds the input for an LLM call — used at turn start (the harness's main LLM call) and from the compactor for summarization (see [Compaction](#compaction)). Once a call is in progress, the LLM client takes over: inside the active turn's tool-use loop, the client maintains its own streaming working buffer (intermediate text, in-flight tool calls, provider-specific thinking/reasoning blocks) and feeds that back to the provider for each round. None of that working memory belongs in `project_for_llm`'s input; it's a continuation, not a fresh projection.

The instruction message (system/developer) is injected by the harness at position 0 *after* projection — it's not part of the conversation storage, so it doesn't belong in `project_for_llm`'s output.

LLM clients translate `list[ProjectedItem]` to their wire format. The Anthropic client retains its role-grouping pass (it builds Anthropic message dicts with `text` / `tool_use` / `tool_result` blocks); it just gets a cleaner input list with text-message merging already done. The OpenAI client passes items through largely unchanged — projection's same-role text merging is what makes the Responses API happy.

---

## Compaction

Compaction continues to operate on the raw window. The fields move:

| Today | New |
|---|---|
| `ContextPartition.parent_partition_uuid` | `Conversation.parent_snapshot_uuid` |
| `ContextPartition.ancestor_summaries` | `Conversation.ancestor_summaries` |
| `ContextPartition.raw_message_start_index` | `Conversation.raw_message_start_index` (now indexes into `messages`, not `items`) |
| `_compact_partition` CAS swap on `partition_uuid` | `_compact_conversation` CAS swap on `snapshot_uuid` |

The summarization prompt change: instead of summarizing a heterogeneous partition (messages + tool internals), summarize the `Conversation`'s raw window directly. Tool-call details from prior turns drop out of the summary entirely — which is what we want; tool calls are turn-local execution detail, not durable conversation content.

Live-window file bodies (the `strip_live_window_bodies` step in `WebHarness._summarize_and_compact`) still apply: the compactor loads the `TurnExecution`s for the messages in the compacted window, hands them to `strip_live_window_bodies` along with the conversation, then projects the stripped result for the summarization LLM call. The strip step operates on `TurnItem`s (file-tool reads live there), not on `ConversationMessage`s.

`PartitionSyncer` becomes `ConversationSyncer`. Its public surface stays similar; the implementation switches from positional reconciliation to `reconcile(stored, incoming)`.

`PartitionCompactor` becomes `ConversationCompactor`. The CAS-swap logic is unchanged in shape (snapshot → summarize → swap → notify); only the data types change.

**Compacted snapshots retain `messages_json`.** When a snapshot transitions to `is_compacted=true`, its `messages_json` is *not* cleared — the ES doc keeps the pre-summary `ConversationMessage` records (`source_id`, `author_id`, `content`, tombstone flags) alongside the new `summary` and boundary metadata. Two reasons:

- The [DAG-scoped assistant-message guardrail](#assistant-message-guardrails) needs per-message identity (`source_id` + content) for compacted ancestors so a [Case B](#when-divergence-creates-a-new-snapshot) retry-before-recency-tail POST that echoes those messages can be validated. Dropping `messages_json` would force the guardrail to either 4xx those legitimate POSTs or silently weaken to "source_id existence only" with no content check — both unacceptable for an asserted invariant.
- It preserves a debug/audit trail of what was in the window before summarization. The parent chain can't substitute: a compacted snapshot's own pre-summary messages only live on that snapshot.

Storage cost is modest. The retained data is short text bounded by the per-window message count; the `TurnExecution`s for the same window are still discarded (those are the bulky part), and `lifted_turn_items` carry forward only the live-window file bodies that need to remain visible.

### Live windows across compaction

Some tools produce results the agent treats as durable state — most notably the file tool's `function_call_output` "live windows," each holding the current view of a tracked file. Today, `lift_active_live_windows` carries those pairs forward across compaction so the LLM doesn't lose sight of tracked files after the raw window is summarized away. The new model preserves the same mechanic via two fields on the child snapshot:

- `Conversation.lifted_turn_items: list[TurnItem]` — function_call / function_call_output pairs lifted from the pre-compaction window, identity preserved by `(path, view_start_line, requested_end_line)` (today's `_live_window_stable_repr`). All pre-compaction live-window pairs whose path appears (non-stale) anywhere in the new raw window are lifted — multiple distinct windows over the same file are first-class, no per-path dedup.
- `Conversation.lifted_anchor_source_id: str | None` — the source_id of the first bot `ConversationMessage` in the new raw window whose `TurnExecution` has a `file_tool.path`-annotated item. Computed at compaction time using the same logic as today's `_tool_round_start_index`. Projection inserts the lifted pairs immediately before the start of that bot message's tool-call round, preserving today's "after the leading user prefix, adjacent to the first relevant file activity" placement. `None` when no file activity exists in the new tail. If the anchor bot is later tombstoned, the syncer re-keys via the same rule that re-keys orphaned `TurnExecution` owners — see [Multi-post bot turns](#turnexecution).

Compaction lift step (runs before summarization):

1. Walk the parent's `TurnExecution`s for messages in the soon-to-be-compacted window, plus the parent's existing `lifted_turn_items`.
2. Determine the set of active paths — paths appearing on any non-stale `file_tool.path`-annotated TurnItem in the new raw window's `TurnExecution`s.
3. For each pre-compaction live-window pair whose path is active and which isn't superseded by a fresh read in the new raw window, lift the `(function_call, function_call_output)` pair into the child's `lifted_turn_items`. Pairs are appended in source order.
4. Set `lifted_anchor_source_id` to the source_id of the new raw window's first bot message carrying a `file_tool.path` annotation in its `TurnExecution`.
5. Strip the live-window bodies from the parent's `TurnItem`s before handing to summarization (existing `strip_live_window_bodies` intent — the summary doesn't fossilize file contents that are already captured in the lifted set).

Projection (`project_for_llm`) emits lifted items at the anchor's tool-round start: when walking raw messages, on reaching the bot message identified by `lifted_anchor_source_id`, emit `lifted_turn_items` (in stored order) *before* the items from that bot message's `TurnExecution`. If `lifted_anchor_source_id` is `None`, no insertion happens.

### Tools that participate in lifted state

`reconcile_tracked_files`, `_find_covering_window`, and the post-write refresh loop in `FileTool` all scan the flat partition item list today. After the split they need a unified view across the new pieces. The conversation_v1 module exposes:

```python
def current_turn_items(
    conversation: Conversation,
    historical_turns: dict[str, TurnExecution],
    active_turn: TurnExecution,
) -> list[TurnItem]:
    """Flat list of TurnItems visible to tools during the current turn:
    `conversation.lifted_turn_items` + each `historical_turns[bot_message.source_id].items`
    in source_id order over conversation.messages + `active_turn.items`. Tools that need
    to see the durable state lifted across compaction must read from this view rather
    than from a single source."""
```

`FileTool` (today's only tool with lift semantics) is constructed against this view instead of `partition.items`. `reconcile_tracked_files` mutates lifted items in place; redundant-read detection sees them via `_find_covering_window`; same-turn write refresh covers them. This is a contract for any future tool that needs lift semantics: read from `current_turn_items`, not from a single source.

### Compaction completion and client relabel

The CAS swap is atomic on the server, but the *client's* knowledge of the new `snapshot_uuid` lags: nodes in `messageTree` created before compaction still carry the parent's id. Without an explicit relabel step, every subsequent POST from those nodes sends the stale id, the server's `_load_exact_partition` sees `is_compacted=true` and falls to chain rebuild, and the cost stacks across repeated compactions — eventually the multi-step lag exceeds the syncer's parent-match window and navigation breaks.

The polling endpoint and the client's `relabelSnapshotUuid` close that gap.

**Endpoint contract** — `GET /compaction-status?conversation_uuid={uuid}&pending_snapshot_uuid={old_id}`:

```python
class CompactionStatusResponse(BaseModel):
    done: bool
    snapshot_uuid: str | None = None
```

`snapshot_uuid` is populated only when the swap committed a *direct child* of `pending_snapshot_uuid` — that's the relabel target. In all other "done" outcomes (lock released without a commit, partition evicted, parent mismatch), `snapshot_uuid` is omitted; the client clears its indicator without relabeling.

**Server-side writes.** At the CAS-commit step the syncer writes `compaction_status:{pending_snapshot_uuid}` in Redis with the committed child's `snapshot_uuid` as the value (or a sentinel for "no relabel target"). TTL matches `CONVERSATION_CACHE_EXPIRY_SECONDS` (7 days by default) so a long-idle client returning to the conversation still finds the relabel target. The polling endpoint reads only Redis — no ES round-trip on the hot path.

**Client behavior** while a compaction-pending indicator is visible:

1. Poll `/compaction-status` every 5 seconds with the `pending_snapshot_uuid` that scheduled the compaction.
2. On `{done: false}`, keep polling.
3. On `{done: true, snapshot_uuid: new_id}`, run `relabelSnapshotUuid(pending_snapshot_uuid, new_id)` over `messageTree`, then clear the indicator.
4. On `{done: true}` with no `snapshot_uuid`, clear the indicator without relabeling.

Polling is the **only** indicator-clearing path. The previous (pre-unified) web design also cleared on any subsequent stream whose handshake carried a different `partition_uuid`, but that side-channel is unsafe once branches are first-class: switching to a sibling branch and sending a message there produces a "different `snapshot_uuid`" handshake even though the original branch's compaction isn't actually done, and the indicator would clear prematurely. If the 5-second polling lag becomes a noticeable UX issue, a safe optimization is to extend the handshake with an explicit `compaction_committed_from: pending_id` marker on compaction-child snapshots so the client can match against its pending indicators — deferred until needed.

**`relabelSnapshotUuid` is idempotent.** It walks `messageTree` and updates *only* nodes whose stored `snapshot_uuid == old_id`, leaving everything else untouched. Running it twice is a no-op on the second pass. That keeps it safe under polling races, repeated `{done: true}` responses on the same branch, and back-to-back compactions.

**Per-branch isolation.** Compaction is keyed on `snapshot_uuid`, so the relabel only touches `messageTree` nodes on the branch that just compacted. Sibling branches under the same `conversation_uuid` keep their own `snapshot_uuid` labels and are unaffected.

**The compaction-pending indicator is branch-scoped.** The indicator is tied to the specific `pending_snapshot_uuid` that scheduled the compaction, not to the conversation as a whole. If the user switches branches while the indicator is up, the client continues polling for that branch's relabel in the background and applies the relabel when it arrives; only the *visible* indicator follows the scheduling branch, not whichever branch happens to be active when the swap completes. Sibling-branch compactions get their own indicators and their own poll loops.

---

## Branches and Snapshots

A single `conversation_uuid` corresponds to a *DAG* of `Conversation` snapshots, not a linear sequence. Two kinds of edges connect snapshots, both using the same field (`parent_snapshot_uuid`); the distinction is in the parent's state:

- **Compaction edge** (linear). The parent has `is_compacted=true`; its raw `messages` got summarized into the child's `ancestor_summaries`. The child is a continuation of the same dialogue under context-window pressure.
- **Branch edge** (sibling). The parent is non-compacted. The child stores a new tail of messages that diverged from the parent's tail. Both children of the same parent represent alternate continuations from a shared prefix.

The compaction chain (today's mechanic) is the first kind. Web edit/regenerate produces the second.

### When divergence creates a new snapshot

`reconcile` classifies the incoming message list against the stored snapshot. If it returns Divergence (incoming has new `source_id`s where stored had different ones, *or* omits stored `source_id`s entirely), the syncer:

1. Finds the divergence point — the longest prefix of `source_id`s that appears in both stored and incoming, in the same order.
2. Creates a new `Conversation` snapshot. There are two cases, depending on where the divergence falls.

**Case A: divergence within the parent's raw window** (the common case — web edit/regenerate). The child inherits the parent's compacted-prefix state verbatim, and *recomputes* lifted state against its own raw window:

```python
shared_prefix = ...  # prefix of parent.messages whose source_ids are in incoming
new_tail = ...       # incoming messages after the divergence point
child_raw_window_turns = load_historical_turns(shared_prefix)
child_active_paths = active_paths_in_turns(child_raw_window_turns)

child = Conversation(
    conversation_uuid=parent.conversation_uuid,
    # snapshot_uuid auto-generated
    parent_snapshot_uuid=parent.snapshot_uuid,
    bot_author_id=parent.bot_author_id,
    ancestor_summaries=list(parent.ancestor_summaries),       # copy, don't alias
    raw_message_start_index=parent.raw_message_start_index,
    lifted_turn_items=filter_lifted_pairs_by_paths(
        list(parent.lifted_turn_items),                       # copy, don't alias
        child_active_paths,
    ),
    lifted_anchor_source_id=first_bot_source_id_with_file_activity(
        shared_prefix, child_raw_window_turns,
    ),
    messages=shared_prefix + new_tail,
)
# Invariant: lifted_anchor_source_id is None  iff  lifted_turn_items is empty.
# No anchor means no insertion point, so retaining items would be dead state.
```

The lifted-state recomputation has the same shape as the compaction lift step (filter to active paths, preserve multiple windows per path, pick the first relevant bot turn as the anchor), just over a different input — the parent's `lifted_turn_items` instead of the parent's `TurnExecution`s. If `child_active_paths` is empty, both `lifted_turn_items` and `lifted_anchor_source_id` end up empty/None; the invariant above is enforced explicitly by `filter_lifted_pairs_by_paths` returning `[]` when the active-path set is empty.

**Case B: divergence before the parent's raw window** (the "retry-before-recency-tail" path from [`project/features/compaction/README.md`](../../features/compaction/README.md)). The user is editing a message that was compacted away on this branch. The syncer falls out of reconcile's Divergence handler and into chain rebuild: a fresh `Conversation` is created from the incoming messages alone, with no inherited `ancestor_summaries`, no `lifted_turn_items`, and `raw_message_start_index=0`. This matches today's "clean partition with no ancestor summaries" recovery.

3. Persists the new snapshot to Redis and ES.
4. Returns the new `snapshot_uuid` in the response stream; the client labels the new branch tip's nodes with it.

The old snapshot is untouched in ES. Its `snapshot_uuid` remains valid; the client returns to it by sending that `snapshot_uuid` on a future POST. This is what fork navigation already does today: the client walks the active path of its `messageTree`, picks up the most recent `snapshot_uuid` stored on those nodes, and sends it. Switching forks reuses the existing per-node identifier rather than triggering a server roundtrip.

For v1, the new snapshot duplicates the shared prefix in `messages` rather than walking the parent chain on every projection. Storage cost is modest and projection stays simple; prefix deduplication is listed as a follow-up optimization.

### Compaction within branches

Each branch is independently compactable. The CAS swap is keyed on `snapshot_uuid`, so two non-compacted siblings under the same `conversation_uuid` don't interfere.

### Slack: divergence should not occur

Slack threads are linear by construction — `message_changed` carries the same `ts`, `message_deleted` is a tombstone, and no client ever creates a "branch tip" with a fresh `source_id` at a position where the thread already has one. If a Slack syncer ever sees `reconcile` return Divergence it means the stored snapshot drifted from Slack's authoritative thread; the recovery action is to overwrite the snapshot in place with the thread view (logged), not to fork.

### Cross-session branch survival

Snapshots persist in ES forever, but the client's per-node `snapshot_uuid` mapping lives in `messageTree` and dies with the page session. Surfacing prior branches after a reload would require persisting the tree (or rebuilding it from the ES snapshot DAG). Out of scope for v1; called out as a future improvement.

---

## Surface Mappings

### Web (single-user)

Field mappings:

| Concept | Value |
|---|---|
| `conversation_uuid` | Existing UUID, generated server-side on first message. |
| `bot_author_id` | Constant `"__bot__"`. |
| Per-message `source_id` | Server-assigned on first encounter, formatted `seconds.microseconds` from `time.time()` (matches Slack `ts`). Echoed back to the client in the response stream; the client echoes it on every subsequent POST. |
| Per-message `author_id` | The current session's `chat_user.id` as a string (for user messages); `"__bot__"` for bot replies. |
| `display_name` | `chat_user.full_name`. Unused in projection because cardinality is 1. |

#### Client model (`scripts/static/ui.js`)

The client maintains a `messageTree`. Edit and regenerate create *new* sibling nodes with fresh client-side ids — never in-place modifications of existing nodes. Each tree node stores the server-assigned `source_id` and the `snapshot_uuid` of the snapshot it belongs to; freshly-created (unsent) nodes have neither until the stream events surface them. Old-branch nodes survive in the tree, which is what enables fork navigation (see [Branches and Snapshots](#branches-and-snapshots) for the server-side mechanic).

#### Web wire protocol

**Request body** (`POST /chat`):

```json
{
  "conversation_uuid": "uuid-of-the-dialogue",
  "snapshot_uuid": "uuid-of-the-current-branch-snapshot-or-null",
  "messages": [
    {"role": "user", "content": "...", "source_id": "1717000000.123456"},
    {"role": "assistant", "content": "...", "source_id": "1717000000.123789"},
    {"role": "user", "content": "..."}
  ]
}
```

Per-message wire fields: `role`, `content`, and an optional `source_id`. The HTTP route normalizes each into a `NormalizedMessage` before calling reconcile:

- `author_id` from session: `role: "user"` → `chat_user.id` (as str); `role: "assistant"` → `"__bot__"`.
- `display_name` from session: `chat_user.full_name` for user messages; omitted for the bot.
- `source_id` assigned by the syncer for any message arriving without one (monotonicity bump enforced). Assignment is **deferred** if the syncer detects a stream-loss recovery scenario — see [Stream-loss recovery](#stream-loss-recovery-post-commit) — so the resync handshake never hands back a provisional `source_id` for a message the server hasn't accepted into the conversation yet.

**Response stream events** (NDJSON). Event ordering is contractual:

- **Handshake** is always the *first* event in the stream, before any `text_delta` / `tool_call` / `progress_message`.
- **Bot message** is emitted *exactly once*, after the final assistant text has been committed to the `Conversation`. If the turn fails before that commit (LLM error, tool crash, stream abort), no `bot_message` is emitted — and the client must *not* create an assistant node from whatever partial text it has buffered.

**Handshake** — first event:

```json
{
  "snapshot_uuid": "uuid-of-snapshot-the-bot-replies-into",
  "source_id_assignments": [
    {"client_index": 4, "source_id": "1717000000.123456"}
  ]
}
```

`snapshot_uuid` is the snapshot the bot is replying into — a fresh branch `snapshot_uuid` if reconcile classified Divergence, the existing one otherwise. `source_id_assignments` carries one entry per request message that arrived without a `source_id`; `client_index` is the 0-based position in the request's `messages` array.

For each `source_id_assignments` entry, the client stamps the corresponding `messageTree` user node with **both** the assigned `source_id` *and* the handshake's `snapshot_uuid`. The handshake's `snapshot_uuid` is authoritative for every newly submitted user node in the request — not just the branch-creation case. This matters under stream-failure recovery: if the stream aborts after handshake but before `bot_message`, no assistant node gets created and the user node remains the active branch tip. With the `snapshot_uuid` already stamped on it, a retry POST walks back from that user node and correctly carries the new branch's `snapshot_uuid`, extending the existing snapshot rather than re-Diverging from an ancestor and orphaning the just-created branch in ES.

**Bot message** — final persistence-relevant event:

```json
{
  "bot_message": {"source_id": "1717000000.123789"}
}
```

The client accumulates streamed text into `fullResponse` during the stream (as today). On `bot_message`, it creates the assistant node via the existing `createMessageNode('assistant', fullResponse, parentNodeId)` flow and stashes *both* this `source_id` *and* the handshake's `snapshot_uuid` on the new node.

**Invariant** (both stamping moments): every `messageTree` node that has a `source_id` also has a `snapshot_uuid`, set at the same time. User nodes get both at handshake; assistant nodes get both at `bot_message`. No node ever carries one without the other.

Other existing events (`text_delta`, `tool_call`, `progress_message`, `compaction_pending`) keep their current shapes.

#### Stream-loss recovery (post-commit)

The handshake-stamp invariant closes the failure mode where the stream aborts *before* `bot_message` (no commit happened). There is a peer failure mode after commit: the server commits the final assistant message to ES, emits `bot_message`, but the client never receives that event (network drop). The client's tree has the user node with the new `snapshot_uuid` but no following assistant node. A naive retry POST `[..., U2', U3]` looks identical on the wire to an intentional fork from U2' on the same `snapshot_uuid` — the server can't distinguish the two from the payload alone (the UI lets a user send from any node, including a user leaf).

Resolution: detect the pattern *before* calling `reconcile` and respond with a **resync handshake** that closes the stream early. The handshake event extends with an optional `unacknowledged_bot_messages` field:

```json
{
  "snapshot_uuid": "...",
  "source_id_assignments": [],
  "unacknowledged_bot_messages": [
    {"source_id": "s_b2'", "content": "<full assistant text>", "parent_source_id": "s2'"}
  ]
}
```

Server-side trigger: the stored snapshot has trailing bot-authored ConversationMessage(s) immediately after the last source_id shared with incoming, and incoming has appended content after that point. When this fires the server emits the resync handshake, *does not* start the LLM, and closes the stream — no `bot_message`, no text deltas.

**`source_id` assignment is deferred on the resync path.** Detection of the trailing-bot-skip pattern runs *before* the syncer assigns `source_id`s to incoming messages that arrived without one. The resync handshake's `source_id_assignments` is therefore always empty: no provisional `source_id` is handed back for the pending user message. Assignment happens only on the repaired retry, which goes through the normal reconcile path. This keeps client logic clean — in the pop-to-draft branch the discarded user node never carried a stale `source_id`, and in the send-from-leaf branch the retry assigns the `source_id` at the same time the recovered bot is integrated into the message path.

Client behavior on non-empty `unacknowledged_bot_messages`:

1. For each entry (in `source_id` ascending order), reconstruct the missing assistant node as a child of the `messageTree` node identified by `parent_source_id`. Stamp both the bot `source_id` and the handshake's `snapshot_uuid` on the new node. Chained entries (a later entry whose `parent_source_id` matches an earlier entry's `source_id`) reconstruct as a chain.
2. Handle the *pending user node* — the user node `sendMessage` created locally just before this POST went out — based on the compose mode at send time:
   - **Send-from-current-leaf** (`editingParentId === null` when `sendMessage` ran): reparent the pending user node so its `parentId` becomes the *last* (most recent in `source_id` order) reconstructed assistant node. Update `activeChildId` along the path so the reparented user node remains the active leaf. **Auto-retry** the original POST.
   - **Edit / regenerate** (`editingParentId !== null` when `sendMessage` ran, or the POST originated from `regenerateMessage`): pop the pending user node out of the tree, restore its content to the chat input as a draft, and surface the recovered assistant message(s) in the UI. **Do not auto-retry.** The user authored the pending message without seeing the recovered bot history; in an explicit non-leaf compose flow, auto-sending it would silently change their intent.

The compose-mode split is a default — client implementations may choose pop-to-draft universally as the conservative variant.

On the retry (in the send-from-leaf path), the pending user node now sits under the recovered bot in the tree, so the retry's `messages` list includes the recovered bot's `source_id`. `reconcile` classifies it as a clean Append.

What happens without the explicit pending-node rule: if the client only reconstructs the bot and flips `activeChildId` without reparenting the pending user node, the pending node and the recovered bot end up as *siblings* under the pre-bot parent. Auto-retrying then either walks to the bot (omitting the user's new message → message silently dropped) or walks back to the pending user node (sending the same wire shape that triggered resync → loop). Both are user-visible regressions.

Why this isn't a new `reconcile` classification: silently treating `[..., U2', U_new]` as recovery would override a legitimate fork-from-U2' intent that produces the same wire shape. Resync-then-retry surfaces the missed history to the client and lets the client (or user) decide between continuation and intentional branching — without the server guessing.

#### Assistant-message guardrails

Web clients only ever author `role: "user"` messages. A future POST whose `messages` list contains an assistant `source_id` that doesn't appear in *any* ConversationMessage for this `conversation_uuid` (across the entire snapshot DAG, including compacted ancestors), or a known assistant `source_id` whose content differs from what storage has, is corruption — not a legitimate edit — and the route handler rejects with a 4xx.

Recognition is DAG-scoped, not snapshot-scoped. This matters for [Case B](#when-divergence-creates-a-new-snapshot) — the retry-before-recency-tail path, where the client legitimately echoes assistant `source_id`s from compacted ancestors that aren't in the currently loaded snapshot. A snapshot-only check would 4xx a documented recovery path.

Implementation options: a Redis-cached `Set` of known assistant `source_id`s (and content hashes) per `conversation_uuid`, populated on conversation load by scanning the DAG and updated on every commit; or an ES `terms` query against the conversations index for the `conversation_uuid` at validation time. Either way, the contract is the same: an assistant `source_id` is "known" if it appears in *any* `ConversationMessage` for this `conversation_uuid`. Both implementations read from compacted snapshots' retained `messages_json` (see [Compaction](#compaction)) — the (`source_id`, `content`) pairs of compacted ancestors stay queryable in ES rather than living in a side index.

The guardrail forecloses a class of replay-style mistakes where a buggy or hostile client tries to rewrite bot history.

### Slack (multi-user)

Field mappings:

| Concept | Value |
|---|---|
| `conversation_uuid` | `uuid5(SLACK_NS, f"{team_id}:{channel_id}:{thread_ts}")` (the Slack wip's deterministic derivation, unchanged). |
| `bot_author_id` | Slack `bot_user_id` (resolved at startup via `auth.test`). |
| Per-message `source_id` | Slack `ts` (format: `seconds.microseconds`). Carried through unchanged — no separate ordering field needed. |
| Per-message `author_id` | Slack `user` field on the message event. |
| `display_name` | Resolved via `users.info` once per author per conversation; cached. |

The Slack wip can collapse significantly under this model: no `conversation_uuid → partition_uuid` mapping plumbing, no `collapse_consecutive_bot_messages` (Slack message splits are stored as separate `ConversationMessage`s; projection's same-role merge re-joins them — and the `TurnExecution` for the underlying LLM call is keyed by the *first* post's `source_id` per the [multi-post bot turns rule](#turnexecution), with tombstone re-keying if that first post is later deleted), no special handling of "non-mention chatter looks like a retry."

The "channel-tail prelude" feature is independent: it's a system-message section, not a conversation message. Stays as a property of the harness's instruction-building step.

---

## Migration Plan

Single-commit migration since the app is not in production. Order:

1. **Build the new primitives.** Add `prokaryotes/conversation_v1/` with `Conversation` (including `lifted_turn_items` / `lifted_anchor_source_id`), `ConversationMessage`, `TurnExecution`, `TurnItem`, `ProjectedItem`, `NormalizedMessage`, `IncomingConversation`, `IncomingMessage`, `reconcile`, `project_for_llm`, `current_turn_items`. Update `prokaryotes/context_v1/` to host `ConversationSyncer` (renamed `partition_sync.py` → `conversation_sync.py`) and `ConversationCompactor` (in `compaction.py`); the compactor's lift step replaces today's `lift_active_live_windows` and computes the child's `lifted_turn_items` / `lifted_anchor_source_id` before summarization. Tests at unit level — pure logic, no infra.
2. **Switch ES indices.** Replace `context-partitions` index with `conversations` (now carries `lifted_turn_items_json` and `lifted_anchor_source_id`) + `turn-executions`. Update `search_v1/` accordingly. Drop the old index on startup if it exists (acceptable because clean break).
3. **Update LLM clients.** `AnthropicClient.stream_turn` and `OpenAIClient.stream_turn` accept a pre-projected `list[ProjectedItem]` plus emitters/callbacks for: (a) the final assistant message (committed to the `Conversation`), and (b) committed turn items (`function_call` and `function_call_output`, committed to the `TurnExecution`). The client owns its own streaming working buffer for the in-flight turn — intermediate narration, in-progress function calls, and any provider-specific thinking/reasoning blocks — and discards it at turn finalization. Each client also owns its `ProjectedItem → wire format` translation.
4. **Update `HarnessBase`.** `sync_context_partition` becomes `sync_conversation`. Per-harness `stream_and_finalize` accepts the new types.
5. **Update `WebHarness` and `FileTool`.** `WebHarness` builds the instruction message, projects, streams. `FileTool` is constructed against the harness's `current_turn_items` view rather than `partition.items`; `reconcile_tracked_files`, `_find_covering_window`, and same-turn write refresh keep their semantics but iterate over the unified view (so they see lifted live windows from prior compactions).
6. **Update web client (`scripts/static/ui.js`).** Read server-assigned `source_id`s from the response stream and store them on the corresponding `messageTree` nodes; echo them back on subsequent POSTs for any node the client has seen before. Edit/regenerate continue to produce *new* sibling nodes with no `source_id` (the syncer assigns one on first encounter, and the server creates a new branch `snapshot_uuid` to hold the new tail). Rename `relabelPartitionUuid` → `relabelSnapshotUuid` and rewire the compaction-pending indicator to be branch-scoped (keyed on the `pending_snapshot_uuid` that scheduled the compaction, not the active branch); the 5-second poll loop stays, but the legacy side-channel clear (on any subsequent stream handshake whose id differed from the pending one) is *removed* — it's unsafe across branches. Polling is the sole clearing path.
7. **Delete legacy.** Remove `ContextPartition`, `ContextPartitionItem`, `ChatConversation.to_context_partition`, the position-based divergence helpers, and `lift_active_live_windows` / `_tool_round_start_index` from `tools_v1/file_tool/live_windows.py` (their logic now lives in the compactor's lift step).

Phase docs (`phase1/README.md`, …) will break this into deployable units once we agree on this overall design.

---

## Testing

Unit:

- `test_conversation_models.py` — model invariants (`source_id` ordering, tombstones, hash payloads, monotonicity bump).
- `test_reconcile.py` — append/edit/delete diff semantics, multi-author scenarios, no-op detection. **Tombstone re-keying**: when a bot `ConversationMessage` that owns a `TurnExecution` is tombstoned, the syncer re-keys `TurnExecution.bot_message_source_id` to the next non-tombstoned bot in the same consecutive run; chained tombstones (first then second post deleted) walk forward correctly; full-run tombstone leaves the `TurnExecution` orphaned; `Conversation.lifted_anchor_source_id` follows the same re-key path and falls to `None` when the run is fully tombstoned; tombstoning a *user* message does not trigger any re-keying; tombstoning a bot that has no `TurnExecution` (subsequent split-run post) is a no-op for ownership.
- `test_project_for_llm.py` — role derivation, prefixing on multi-author, same-role merging, turn-item interleaving, deleted-message skipping.
- `test_conversation_syncer.py` — Redis fast path, ES exact load, ancestor-chain rebuild for the new types, and divergence creating a new branch snapshot (fresh `snapshot_uuid`) while leaving the parent snapshot intact in ES. Branch-creation inheritance: from a compacted parent, the child carries `ancestor_summaries` and `raw_message_start_index` verbatim; `lifted_turn_items` is filtered to paths active in the child's raw window (multiple windows per path preserved); `lifted_anchor_source_id` is recomputed from the child's raw window. Invariant: `anchor=None` iff `lifted_turn_items == []` — no dead lifted state. Retry-before-recency-tail (divergence before `raw_message_start_index`) produces a fresh `Conversation` with no inherited compacted-prefix state, distinct from the Divergence path.
- `test_search_conversations.py` — new ES queries (`find_latest_active_snapshot_uuid`, `find_conversation_by_tail_hash`, `search_conversations`).
- `test_conversation_compactor.py` — CAS swap, summarization-input shape, live-window stripping moves to the turn-execution side. Lift coverage: all pre-compaction live-window pairs for paths active in the new raw window are lifted (multiple windows per path preserved, identity by `(path, view_start_line, requested_end_line)`); stale/tombstoned pairs are excluded; transitive roll-forward across a second compaction when the path isn't superseded; a fresh read in the post-compaction window supersedes the prior lifted pair on the next compaction; `lifted_anchor_source_id` matches the bot message identified by today's `_tool_round_start_index` logic. **`messages_json` preservation**: after the swap, the parent snapshot's `messages_json` is still populated with its pre-summary `ConversationMessage` records (specifically, `source_id` + `author_id` + `content` for bot messages survive the transition to `is_compacted=true`); the DAG-scoped guardrail can read those from ES on a Case B retry. **Compaction-status writes**: the CAS commit writes `compaction_status:{pending_snapshot_uuid}` in Redis with the child `snapshot_uuid` as the value and `CONVERSATION_CACHE_EXPIRY_SECONDS` TTL; "no relabel target" outcomes (no commit, evicted, parent mismatch) write the sentinel so the polling endpoint can return `done=true` without a `snapshot_uuid`.
- `test_compaction_status_endpoint.py` (new) — `GET /compaction-status` reads only from Redis (no ES touch); returns `{done: false}` while the lock is held; returns `{done: true, snapshot_uuid: child_id}` after a direct-child commit; returns `{done: true}` without `snapshot_uuid` for non-direct-child completions; tolerates polling after TTL expiry (returns `done: true` with no relabel target, matching the "long-idle return" recovery path).
- `test_file_tool_post_compaction.py` (new) — `reconcile_tracked_files` refreshes a lifted window in place after an external edit; `_find_covering_window` sees lifted windows and short-circuits redundant reads; same-turn write refresh updates the lifted window's revision; tombstoning a tracked file after compaction marks the lifted window stale. All flows read via `current_turn_items`.
- `test_projection_with_lifted.py` (new) — lifted pairs emit at the anchor's tool-round start; Anthropic projection stays user-first with `tool_use` preceding its `tool_result`; OpenAI Responses projection keeps alternation-safe order with lifted items inserted at the anchor.
- `test_anthropic_v1.py` / `test_openai_v1.py` — carry forward the existing transient-narration regression guards (intermediate assistant text is streamed but not committed to `TurnExecution`; the second provider round in a tool-use loop receives only persisted items).

- `test_web_wire_protocol.py` — handshake is the first stream event before any text; `source_id_assignments` map back to the right request indices; handshake stamps each affected user node with both `source_id` *and* `snapshot_uuid`; `bot_message` is emitted exactly once after the final assistant text commits and stamps both ids on the assistant node; on simulated mid-turn failure no `bot_message` is emitted; a retry POST after a handshake-then-aborted stream walks back from the un-bot-replied user node, picks up the new `snapshot_uuid`, and extends the just-created branch as an Append instead of triggering a second Divergence + duplicate snapshot. **Stream-loss recovery**: when the server has a trailing committed assistant message that's missing from incoming, the server emits a resync handshake with `unacknowledged_bot_messages` and closes the stream without invoking the LLM; the client reconstructs the missing assistant node under `parent_source_id` with both ids stamped. The resync response's `source_id_assignments` is empty even when the original POST included a new (no-source_id) user message — assignment is deferred to the repaired retry. **Send-from-current-leaf**: the pending user node is reparented under the recovered bot, the auto-retry walks the repaired path, the syncer assigns the `source_id` on the retry, and `reconcile` classifies as a clean Append on the same `snapshot_uuid` (no orphan snapshot, no resync loop). **Edit / regenerate**: the pending user node is popped from the tree, its content restored to the draft input, no auto-retry; verified that the discarded user node never carried a server-assigned `source_id`. **Failure-mode guards**: without reparenting in the send-from-leaf case, the retry either drops the user's message silently or loops on resync — both are explicitly tested as anti-cases against the documented behavior. **Guardrail**: route handler accepts a Case B retry-before-recency-tail POST whose echoed assistant `source_id`s come from compacted ancestors (recognized via the DAG-scoped index); rejects POSTs that carry a `conversation_uuid`-unknown assistant `source_id`, or a known assistant `source_id` whose content differs from any stored copy.

Integration (Tier B):

- `test_unified_web_flow.py` — replaces existing web partition tests. New conversation, retry/edit branching off into a new snapshot (parent preserved as a sibling), navigating back to the parent branch and continuing it, compaction triggered by repeated turns within a branch.
- `test_unified_multi_author_flow.py` — synthetic two-human conversation with bot replies, exercises display-name prefixing, edit-of-old-user-message preserves later turns.

Browser UI (`tests/ui_tests/`):

- `ui.test.js` — extend the existing fork-navigation block with **compaction relabel** coverage: `relabelSnapshotUuid(old, new)` updates only `messageTree` nodes where `snapshot_uuid === old` (idempotent on a second pass, no-op when no matches); sibling branches' `snapshot_uuid`s untouched after relabel; the compaction-pending indicator is keyed on the *scheduling* `pending_snapshot_uuid` so switching to a sibling branch mid-compaction does not clear or relocate the indicator (anti-case: sending on a sibling branch must *not* trigger a side-channel clear — polling is the only clearing path); polling-`{done: true, snapshot_uuid}` triggers relabel + clear; polling-`{done: true}` without `snapshot_uuid` clears without relabeling; back-to-back compactions on the same branch each get their own poll loop and relabel cycle, no stranding on a stale id.

The Slack flow tests proposed in `slack_harness/README.md` ride on these primitives once both wips land.

---

## Open Questions

1. **`author_id` namespace collisions.** Slack `user_id`s and web `chat_user.id`s could in theory share string values. Conversations are surface-scoped, so within a given conversation there's no collision — but if we ever want cross-surface analytics or a unified `author` directory, we'd want a namespaced form (`slack:T123:U456`, `web:42`). Defer; keep opaque-per-surface for v1.
2. **Multi-bot conversations.** `Conversation.bot_author_id` is single-valued. If we ever want a conversation with two bot personas (e.g., a "summarizer" and a "primary"), this becomes `bot_author_ids: set[str]`. Defer.
3. **TurnExecution garbage collection.** Two orphaning paths produce sweepable `TurnExecution`s: (a) compaction — when a `Conversation` snapshot's raw window is summarized, the `TurnExecution`s for messages in that window become unreachable; (b) multi-post tombstone-rekey — when every bot in a split run is tombstoned, the `TurnExecution` is orphaned regardless of compaction (see [Multi-post bot turns](#turnexecution)). Policy options: leave them in ES forever (small cost), sweep on compaction commit (covers path (a)) plus on full-run tombstone (covers path (b)), or make `turn-executions` index TTL-based. Probably the second — but worth confirming index-lifecycle conventions in the existing codebase.
4. **Hash function stability.** `compute_boundary_hash` payload changes from `{role, content}` to `{author_id, content}`. This is a deliberate stability win (role-assignment changes don't invalidate hashes) but it does mean the hash space is incompatible with any pre-migration data. Aligns with the clean-break decision.
5. **Web bot identity.** Constant `"__bot__"` is workable but ugly. Alternatives: `"assistant"`, `"bot:web"`, a per-deployment UUID. No strong preference — pick one and stick with it.
6. **`source_id` monotonicity collisions on the web side.** Two messages from the same client within the same microsecond is improbable but conceivable on fast retries. The syncer bumps by `+1` microsecond if the candidate `source_id` is not strictly greater than the last assigned one in the conversation. Simple and stable; documented here so future readers don't reinvent it.
7. **Branch snapshot accumulation in ES.** A heavily-forked conversation produces one ES doc per branch tip plus one per compaction step within each branch. Acceptable at v1 scale, but a long-lived prolific user could rack up hundreds of snapshots. Eventual sweep policy options: (a) hard-delete non-leaf, non-compacted snapshots that haven't been touched in N days, (b) reference-count from the client's persisted tree (when that exists), (c) leave forever. Defer.

---

## Out of Scope for v1

- Cross-surface conversation merging (one user talking to the same bot via web and Slack and seeing unified history).
- Per-message rich content (attachments, images, files). The `content: str` field is intentionally simple; a future `content_parts: list[ContentPart]` extension can wrap it.
- Cross-session branch survival. Snapshots persist in ES across reloads, but the client's per-node `snapshot_uuid` mapping does not — so after a page reload the user resumes the most recently active branch only, with no way to navigate to previous siblings. Restoring full tree navigation needs a persisted tree (out of scope for v1; the snapshot DAG in ES is already sufficient to support it later).
- Branch-prefix deduplication. Sibling snapshots currently duplicate the shared prefix in `messages` rather than walking the parent chain. Acceptable for v1; worth revisiting if storage becomes a concern.
- Cross-conversation references (mentioning another conversation by ID).
- Edit history beyond `edited: bool`. Slack and web both have access to prior versions via their own UIs; harness storage doesn't need to duplicate.

---

## Relevant Code Files (expected)

| File | Role |
|---|---|
| `prokaryotes/api_v1/models.py` | Delete `ContextPartition` / `ContextPartitionItem` / `ChatConversation` / `ChatMessage`. Add `IncomingConversation` / `IncomingMessage`. |
| `prokaryotes/conversation_v1/__init__.py` (new) | Exports for the new primitives. |
| `prokaryotes/conversation_v1/models.py` (new) | `Conversation` (gains `lifted_turn_items` and `lifted_anchor_source_id`), `ConversationMessage`, `TurnExecution`, `TurnItem`, `ProjectedItem`, `NormalizedMessage`. |
| `prokaryotes/conversation_v1/reconcile.py` (new) | `reconcile`, `ReconcileResult`. |
| `prokaryotes/conversation_v1/project.py` (new) | `project_for_llm`, `_merge_consecutive_same_role`, `current_turn_items` (the unified view tools with lift semantics read from). |
| `prokaryotes/context_v1/partition_sync.py` → `conversation_sync.py` | `ConversationSyncer` replaces `PartitionSyncer`. |
| `prokaryotes/context_v1/compaction.py` | `ConversationCompactor` replaces `PartitionCompactor`. Lift step computes `lifted_turn_items` / `lifted_anchor_source_id` for the child snapshot before summarization. |
| `prokaryotes/tools_v1/file_tool/` | `reconcile_tracked_files`, `_find_covering_window`, and post-write refresh read from `current_turn_items` instead of `partition.items`. `lift_active_live_windows` and `_tool_round_start_index` logic moves into `ConversationCompactor`'s lift step. |
| `prokaryotes/search_v1/context_partitions.py` → `conversations.py` | New ES queries against the new indices. |
| `prokaryotes/anthropic_v1/`, `prokaryotes/openai_v1/` | `stream_turn` accepts the new types; provider translation moves to `ProjectedItem` → wire format. |
| `prokaryotes/harness_v1/web.py` | Migrated; same UX. |
| `prokaryotes/web_v1/` | `WebBase` `/compaction-status` endpoint accepts `pending_snapshot_uuid` (renamed from `pending_partition_uuid`) and returns `CompactionStatusResponse { done, snapshot_uuid? }` against the new Redis key shape. Reads only Redis. |
| `scripts/static/ui.js` | Client-side `source_id` tracking — read assigned IDs from the response stream, store them on `messageTree` nodes, echo on subsequent POSTs for any node that already has one. New edit/regenerate nodes ship with no `source_id`. |
| `tests/unit_tests/test_conversation_*.py` (new) | Per the Testing section. |
| `tests/integration_tests/tier_b/test_unified_*.py` (new) | Per the Testing section. |
