"""Three-tier conversation reconciliation: Redis fast path → exact ES load → ancestor-chain rebuild.

Replaces `PartitionSyncer`. Reconciliation is source-ID-based via
`prokaryotes.conversation_v1.reconcile`; the syncer applies the result
per-surface — divergence creates a new branch snapshot on web; Slack-specific
subclasses can override `_apply_divergence` for in-place recovery.

The web flow also handles two stream-loss recovery scenarios:
- Pre-commit (handshake-stamp invariant): managed by the client; the syncer
  produces a `snapshot_uuid` in the handshake even before the bot message
  commits so retries from the un-bot-replied user node extend the branch
  instead of re-diverging.
- Post-commit (resync handshake): if the stored snapshot has trailing
  bot-authored messages immediately after the last source_id shared with
  incoming and incoming has new content beyond that, the syncer emits an
  `unacknowledged_bot_messages` payload and does *not* start the LLM. See
  the design doc.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from redis.asyncio import Redis

from prokaryotes.api_v1.models import IncomingMessage
from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    NormalizedMessage,
    compute_boundary_hash,
)
from prokaryotes.conversation_v1.reconcile import reconcile
from prokaryotes.search_v1 import SearchClient
from prokaryotes.search_v1.conversations import (
    conversation_from_doc,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceIdAssignment:
    """One entry per request message that arrived without a `source_id`.
    `client_index` is the 0-based position in the request's messages array."""

    client_index: int
    source_id: str


@dataclass
class UnacknowledgedBotMessage:
    """A bot message the server has committed but the client hasn't seen, surfaced
    in the resync handshake. The client reconstructs the assistant node under
    `parent_source_id`."""

    source_id: str
    content: str
    parent_source_id: str


@dataclass
class SyncResult:
    """Output of `sync_conversation`.

    When `resync=True`, `conversation` is the unchanged stored snapshot and the
    caller is expected to close the stream without starting the LLM. The client
    reconstructs the missing bot history from `unacknowledged_bot_messages`,
    repairs its tree per the compose-mode split rule, and retries.

    When `is_new_branch=True`, the caller's response stream's handshake must
    carry the new snapshot's `snapshot_uuid` so the client can stamp it on the
    pending user node.
    """

    conversation: Conversation
    source_id_assignments: list[SourceIdAssignment] = field(default_factory=list)
    is_new_branch: bool = False
    resync: bool = False
    unacknowledged_bot_messages: list[UnacknowledgedBotMessage] = field(default_factory=list)


@dataclass
class _PartialMessage:
    """Internal pre-assignment shape — `source_id` may be `None`."""

    author_id: str
    content: str
    client_index: int
    source_id: str | None = None
    display_name: str | None = None


class ConversationSyncer(ABC):
    """Source-ID-based reconciliation with per-surface apply hooks.

    Subclasses provide `conversation_cache_ex`, `redis_client`, `search_client`.
    Default `_apply_divergence` creates a new branch snapshot (web semantics).
    Slack subclasses can override to overwrite-in-place per the recovery rule.
    """

    @property
    @abstractmethod
    def conversation_cache_ex(self) -> int: ...

    @property
    @abstractmethod
    def redis_client(self) -> Redis: ...

    @property
    @abstractmethod
    def search_client(self) -> SearchClient: ...

    async def sync_conversation(
        self,
        *,
        conversation_uuid: str,
        snapshot_uuid: str | None,
        bot_author_id: str,
        incoming: list[IncomingMessage],
        session_user_id: str,
        session_display_name: str | None,
    ) -> SyncResult:
        partial = _partially_normalize(
            incoming,
            bot_author_id=bot_author_id,
            session_user_id=session_user_id,
            session_display_name=session_display_name,
        )

        stored = await self._load_stored(
            conversation_uuid=conversation_uuid,
            snapshot_uuid=snapshot_uuid,
            bot_author_id=bot_author_id,
            partial=partial,
        )

        unacknowledged = _detect_unacknowledged_bot_messages(stored, partial)
        if unacknowledged:
            return SyncResult(
                conversation=stored,
                resync=True,
                unacknowledged_bot_messages=unacknowledged,
            )

        assignments = self._assign_source_ids(partial, stored)
        normalized = [_to_normalized(m) for m in partial]
        result = reconcile(stored, normalized)

        final = await self._apply_result(
            stored=stored,
            result=result,
            normalized=normalized,
            bot_author_id=bot_author_id,
            conversation_uuid=conversation_uuid,
        )

        await self._cache_and_persist_conversation(final)

        return SyncResult(
            conversation=final,
            source_id_assignments=assignments,
            is_new_branch=(final.snapshot_uuid != stored.snapshot_uuid),
        )

    async def _apply_result(
        self,
        *,
        stored: Conversation,
        result,
        normalized: list[NormalizedMessage],
        bot_author_id: str,
        conversation_uuid: str,
    ) -> Conversation:
        if result.classification == "match":
            return stored
        if result.classification == "append":
            for op in result.operations:
                if op.kind == "append" and op.incoming is not None:
                    stored.messages.append(_normalized_to_message(op.incoming))
            return stored
        if result.classification == "edit":
            for op in result.operations:
                if op.kind == "edit" and op.incoming is not None:
                    msg = stored.message_by_source_id(op.source_id)
                    if msg is not None:
                        msg.content = op.incoming.content
                        msg.edited = True
            return stored
        if result.classification == "delete":
            for op in result.operations:
                if op.kind == "delete":
                    msg = stored.message_by_source_id(op.source_id)
                    if msg is not None:
                        msg.deleted = True
            return stored
        return await self._apply_divergence(
            stored=stored,
            result=result,
            normalized=normalized,
            bot_author_id=bot_author_id,
            conversation_uuid=conversation_uuid,
        )

    async def _apply_divergence(
        self,
        *,
        stored: Conversation,
        result,
        normalized: list[NormalizedMessage],
        bot_author_id: str,
        conversation_uuid: str,
    ) -> Conversation:
        """Default (web) divergence behavior: create a new branch snapshot rooted
        at the shared prefix. Case A — divergence within the parent's raw window —
        inherits compacted state and recomputes lifted state against the child's
        raw window. The Case B (retry-before-recency-tail) path materializes in
        `_rebuild_from_chain`'s fresh-Conversation fallback, never here.

        Slack subclasses can override for in-place recovery — see design doc.
        """
        shared_ids = set(result.shared_prefix_source_ids)
        shared_messages = [m for m in stored.messages if m.source_id in shared_ids]
        normalized_by_id = {m.source_id: m for m in normalized}
        new_tail_normalized = [
            normalized_by_id[m_id] for m_id in [n.source_id for n in normalized] if m_id not in shared_ids
        ]
        new_tail = [_normalized_to_message(m) for m in new_tail_normalized]

        active_paths = _active_paths_for_messages(shared_messages, {}, set())
        lifted = _filter_lifted_pairs_by_paths(stored.lifted_turn_items, active_paths)
        anchor = _recompute_lifted_anchor(shared_messages, {}, bot_author_id) if lifted else None

        return Conversation(
            conversation_uuid=conversation_uuid,
            parent_snapshot_uuid=stored.snapshot_uuid,
            bot_author_id=bot_author_id,
            ancestor_summaries=list(stored.ancestor_summaries),
            lifted_turn_items=lifted,
            lifted_anchor_source_id=anchor,
            raw_message_start_index=stored.raw_message_start_index,
            messages=shared_messages + new_tail,
        )

    def _assign_source_ids(self, partial: list[_PartialMessage], stored: Conversation) -> list[SourceIdAssignment]:
        """Assign `source_id`s to incoming messages that arrived without one.

        Format: `seconds.microseconds` from `time.time()`. Monotonicity within
        the conversation is enforced: if the candidate is `<=` the last
        source_id seen so far (in stored or earlier in this incoming list), it's
        bumped by 1 microsecond.
        """
        existing_ids: list[str] = [m.source_id for m in stored.messages if not m.deleted]
        for m in partial:
            if m.source_id is not None:
                existing_ids.append(m.source_id)
        last = max(existing_ids) if existing_ids else None

        assignments: list[SourceIdAssignment] = []
        for m in partial:
            if m.source_id is not None:
                continue
            candidate = _format_source_id(time.time())
            if last is not None and candidate <= last:
                candidate = _bump_source_id(last)
            m.source_id = candidate
            last = candidate
            assignments.append(SourceIdAssignment(client_index=m.client_index, source_id=candidate))
        return assignments

    async def _cache_and_persist_conversation(self, conversation: Conversation) -> None:
        await self.redis_client.set(
            f"conversation:{conversation.conversation_uuid}",
            conversation.model_dump_json(),
            ex=self.conversation_cache_ex,
        )
        await self.search_client.put_conversation(conversation)

    async def _load_exact_conversation(
        self,
        conversation_uuid: str,
        snapshot_uuid: str,
    ) -> tuple[Conversation | None, dict | None]:
        """Fetch the snapshot doc for `snapshot_uuid`.

        Returns `(conversation, None)` when the doc exists and is not compacted.
        Returns `(None, doc)` when the doc is compacted (caller forwards as
        head_doc to chain rebuild). Returns `(None, None)` when missing/invalid.
        """
        doc = await self.search_client.get_conversation(snapshot_uuid)
        if not doc:
            return None, None
        if doc.get("is_compacted"):
            return None, doc
        try:
            return conversation_from_doc(conversation_uuid, doc), None
        except Exception as exc:
            logger.warning("Failed to rebuild exact ES snapshot %s: %s", snapshot_uuid, exc)
            return None, None

    async def _load_stored(
        self,
        *,
        conversation_uuid: str,
        snapshot_uuid: str | None,
        bot_author_id: str,
        partial: list[_PartialMessage],
    ) -> Conversation:
        # Tier 1: Redis fast path. Must match client's snapshot_uuid (or its parent —
        # post-compaction relabel hasn't reached the client yet).
        cached_data = await self.redis_client.get(f"conversation:{conversation_uuid}")
        if cached_data:
            try:
                cached = Conversation.model_validate_json(cached_data)
            except Exception:
                cached = None
            if cached is not None and _conversation_can_follow_client(cached, snapshot_uuid):
                return cached
            elif cached is not None:
                logger.info(
                    "Cached snapshot %s does not match client snapshot %s; falling back to ES",
                    cached.snapshot_uuid,
                    snapshot_uuid,
                )

        # Tier 2: ES exact load.
        head_doc: dict | None = None
        if snapshot_uuid:
            exact, head_doc = await self._load_exact_conversation(conversation_uuid, snapshot_uuid)
            if exact is not None:
                return exact

        # Tier 3: ancestor-chain rebuild.
        return await self._rebuild_from_chain(
            conversation_uuid=conversation_uuid,
            snapshot_uuid=snapshot_uuid,
            bot_author_id=bot_author_id,
            partial=partial,
            head_doc=head_doc,
        )

    async def _rebuild_from_chain(
        self,
        *,
        conversation_uuid: str,
        snapshot_uuid: str | None,
        bot_author_id: str,
        partial: list[_PartialMessage],
        head_doc: dict | None,
    ) -> Conversation:
        """Walk the snapshot DAG back to a compacted ancestor whose boundary_hash
        matches a prefix of incoming; build a Conversation rooted there.

        If no matching ancestor (Case B retry-before-recency-tail) or no client
        snapshot_uuid at all (fresh conversation), return a fresh `Conversation`
        with no ancestors, no lifted state, `raw_message_start_index=0`,
        `messages=[]`. The reconcile-apply step then appends the full incoming
        list as plain appends.
        """
        if not snapshot_uuid:
            logger.info("Starting new conversation from request payload")
            return _fresh_conversation(conversation_uuid, bot_author_id)

        chain = await self._walk_snapshot_chain(
            conversation_uuid=conversation_uuid,
            snapshot_uuid=snapshot_uuid,
            head_doc=head_doc,
        )
        compacted_ancestors = [doc for doc in reversed(chain) if doc.get("is_compacted")]

        matched_ancestor: dict | None = None
        for doc in compacted_ancestors:
            boundary_count = doc.get("boundary_message_count") or 0
            boundary_hash = doc.get("boundary_hash")
            if not boundary_hash or boundary_count == 0:
                continue
            if boundary_count > len(partial):
                continue
            prefix_messages = [
                ConversationMessage(
                    source_id=m.source_id or "",
                    author_id=m.author_id,
                    content=m.content,
                )
                for m in partial[:boundary_count]
            ]
            if compute_boundary_hash(prefix_messages) == boundary_hash:
                matched_ancestor = doc

        if matched_ancestor is None:
            logger.info("No compacted ancestor validated; starting fresh from request payload")
            return _fresh_conversation(conversation_uuid, bot_author_id)

        ancestor_summaries: list[str] = []
        for doc in compacted_ancestors:
            summary = doc.get("summary")
            if summary:
                ancestor_summaries.append(summary)
            if doc["snapshot_uuid"] == matched_ancestor["snapshot_uuid"]:
                break

        raw_start = matched_ancestor.get("boundary_message_count") or 0
        echoed_messages: list[ConversationMessage] = []
        for m in partial[raw_start:]:
            if m.source_id is None:
                break
            echoed_messages.append(
                ConversationMessage(
                    source_id=m.source_id,
                    author_id=m.author_id,
                    content=m.content,
                    display_name=m.display_name,
                )
            )

        return Conversation(
            conversation_uuid=conversation_uuid,
            parent_snapshot_uuid=matched_ancestor["snapshot_uuid"],
            bot_author_id=bot_author_id,
            ancestor_summaries=ancestor_summaries,
            raw_message_start_index=raw_start,
            messages=echoed_messages,
        )

    async def _walk_snapshot_chain(
        self,
        *,
        conversation_uuid: str,
        snapshot_uuid: str,
        head_doc: dict | None = None,
    ) -> list[dict]:
        """Walk the parent chain from `snapshot_uuid` back to root, oldest-last.

        `head_doc`, if provided, avoids a redundant ES read for the head.
        """
        chain: list[dict] = []
        seen: set[str] = set()
        current = snapshot_uuid
        while current and current not in seen:
            seen.add(current)
            if head_doc is not None and head_doc.get("snapshot_uuid") == current:
                doc = head_doc
                head_doc = None
            else:
                doc = await self.search_client.get_conversation(current)
            if not doc:
                break
            if doc.get("conversation_uuid") and doc["conversation_uuid"] != conversation_uuid:
                break
            chain.append(doc)
            current = doc.get("parent_snapshot_uuid")
        return chain


def _active_paths_for_messages(
    messages: list[ConversationMessage],
    historical_turns: dict,
    seen: set[str],
) -> set[str]:
    """Active paths in a message window: any `file_tool.path`-annotated
    TurnItem in the window's TurnExecutions where status != 'stale'.

    Stubbed for v1 since branch-creation inheritance is a fresh Conversation
    when no TurnExecutions are loaded; the real recomputation runs in the
    compactor's lift step. Returning an empty set produces empty lifted
    state on the child, which matches the conservative-correctness rule
    (`anchor=None iff lifted_turn_items == []`).
    """
    return seen


def _bump_source_id(source_id: str) -> str:
    seconds_str, _, micros_str = source_id.partition(".")
    try:
        seconds = int(seconds_str)
        micros = int(micros_str or "0")
    except ValueError:
        return _format_source_id(time.time())
    micros += 1
    if micros >= 1_000_000:
        seconds += 1
        micros = 0
    return f"{seconds}.{micros:06d}"


def _conversation_can_follow_client(conversation: Conversation, client_snapshot_uuid: str | None) -> bool:
    """A cached snapshot can serve a client whose own `snapshot_uuid` is either
    this snapshot or its parent (the client may not have run `relabelSnapshotUuid`
    after a recent compaction yet)."""
    if client_snapshot_uuid is None:
        return True
    return (
        conversation.snapshot_uuid == client_snapshot_uuid or conversation.parent_snapshot_uuid == client_snapshot_uuid
    )


def _detect_unacknowledged_bot_messages(
    stored: Conversation,
    partial: list[_PartialMessage],
) -> list[UnacknowledgedBotMessage]:
    """Resync trigger: stored has trailing bot-authored messages immediately
    after the last source_id shared with incoming, and incoming has new content
    (a new source_id, or a message that arrived without one) beyond that point.

    Returns the unacknowledged bots in source_id order, or `[]` if not a resync.
    """
    stored_sorted = stored.sorted_messages()
    stored_non_deleted = [m for m in stored_sorted if not m.deleted]
    if not stored_non_deleted:
        return []

    partial_ids = {m.source_id for m in partial if m.source_id is not None}

    last_shared_idx = -1
    for idx, msg in enumerate(stored_non_deleted):
        if msg.source_id in partial_ids:
            last_shared_idx = idx
    if last_shared_idx == -1:
        return []

    trailing = stored_non_deleted[last_shared_idx + 1 :]
    if not trailing:
        return []
    if not all(m.author_id == stored.bot_author_id for m in trailing):
        return []

    has_new_incoming = any(
        m.source_id is None or m.source_id not in {n.source_id for n in stored_non_deleted} for m in partial
    )
    if not has_new_incoming:
        return []

    out: list[UnacknowledgedBotMessage] = []
    parent_id = stored_non_deleted[last_shared_idx].source_id
    for msg in trailing:
        out.append(
            UnacknowledgedBotMessage(
                source_id=msg.source_id,
                content=msg.content,
                parent_source_id=parent_id,
            )
        )
        parent_id = msg.source_id
    return out


def _filter_lifted_pairs_by_paths(lifted_items, active_paths: set[str]):
    """Return only those `(function_call, function_call_output)` pairs whose
    `file_tool.path` annotation is in `active_paths`. Empty when `active_paths`
    is empty, preserving the invariant `anchor=None iff lifted == []`."""
    if not active_paths:
        return []
    out = []
    by_call_id: dict[str, object] = {}
    for item in lifted_items:
        if item.type == "function_call":
            cid = item.call_id or item.id
            if cid is not None:
                by_call_id[cid] = item
            continue
        if item.type != "function_call_output":
            continue
        ann = item.prokaryotes_annotations or {}
        if ann.get("file_tool.path") not in active_paths:
            continue
        cid = item.call_id or item.id
        function_call_item = by_call_id.get(cid) if cid else None
        if function_call_item is None:
            continue
        out.append(function_call_item)
        out.append(item)
    return out


def _format_source_id(ts: float) -> str:
    seconds = int(ts)
    micros = int((ts - seconds) * 1_000_000)
    return f"{seconds}.{micros:06d}"


def _fresh_conversation(conversation_uuid: str, bot_author_id: str) -> Conversation:
    return Conversation(
        conversation_uuid=conversation_uuid,
        bot_author_id=bot_author_id,
        messages=[],
    )


def _normalized_to_message(m: NormalizedMessage) -> ConversationMessage:
    return ConversationMessage(
        source_id=m.source_id,
        author_id=m.author_id,
        content=m.content,
        display_name=m.display_name,
    )


def _partially_normalize(
    incoming: list[IncomingMessage],
    *,
    bot_author_id: str,
    session_user_id: str,
    session_display_name: str | None,
) -> list[_PartialMessage]:
    out: list[_PartialMessage] = []
    for idx, msg in enumerate(incoming):
        if msg.role == "assistant":
            author_id = bot_author_id
            display_name = None
        else:
            author_id = session_user_id
            display_name = session_display_name
        out.append(
            _PartialMessage(
                author_id=author_id,
                content=msg.content,
                source_id=msg.source_id,
                display_name=display_name,
                client_index=idx,
            )
        )
    return out


def _recompute_lifted_anchor(
    messages: list[ConversationMessage],
    historical_turns: dict,
    bot_author_id: str,
) -> str | None:
    """First bot message in the window whose TurnExecution has a `file_tool.path`-
    annotated item. Stub for v1 — the compactor's lift step provides the real
    computation. See `ConversationCompactor`."""
    return None


def _to_normalized(m: _PartialMessage) -> NormalizedMessage:
    if m.source_id is None:
        raise RuntimeError(
            "Internal error: NormalizedMessage requires source_id; assignment must happen before conversion"
        )
    return NormalizedMessage(
        source_id=m.source_id,
        author_id=m.author_id,
        content=m.content,
        display_name=m.display_name,
    )
