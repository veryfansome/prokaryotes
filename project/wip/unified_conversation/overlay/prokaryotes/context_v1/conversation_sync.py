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

import hashlib
import json
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
    TurnExecution,
    compute_boundary_hash,
    conversation_message_items,
)
from prokaryotes.conversation_v1.reconcile import reconcile
from prokaryotes.search_v1 import SearchClient
from prokaryotes.search_v1.conversations import (
    conversation_from_doc,
    messages_from_doc,
)

logger = logging.getLogger(__name__)


class AssistantMessageGuardrailError(ValueError):
    """Raised by `validate_assistant_messages` when an incoming `role="assistant"`
    entry is unknown to the conversation DAG or its content doesn't match what
    storage has. Web route handlers should catch and convert to a 400 response.
    """


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
        # Track the pre-split stored snapshot — Case B replaces `stored` with a
        # fresh Conversation, but the SyncResult's `is_new_branch` flag must still
        # report the divergence from the snapshot the client sent against.
        original_stored_snapshot_uuid = stored.snapshot_uuid

        # Compacted-prefix split: runs BEFORE resync detection so compacted-prefix
        # entries can't pollute `_detect_unacknowledged_bot_messages`'s
        # `has_new_incoming` heuristic. After the split, reconcile only sees the
        # raw-window suffix; any divergence it reports is Case A by construction.
        if stored.raw_message_start_index > 0:
            prefix_match, raw_suffix = await self._split_compacted_prefix(stored, partial)
            if prefix_match:
                partial = raw_suffix
            else:
                # Case B — the user is editing inside the compacted prefix.
                # Discard `stored` and work against a fresh root Conversation;
                # `partial` keeps the full incoming list so the reconcile-apply
                # path appends everything as plain appends.
                stored = _fresh_conversation(conversation_uuid, bot_author_id)

        unacknowledged = _detect_unacknowledged_bot_messages(stored, partial)
        if unacknowledged:
            return SyncResult(
                conversation=stored,
                resync=True,
                unacknowledged_bot_messages=unacknowledged,
            )

        # Source-id assignment for any raw-suffix (or Case B) entries that arrived
        # without one — `_to_normalized` would otherwise raise on `source_id=None`.
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
            is_new_branch=(final.snapshot_uuid != original_stored_snapshot_uuid),
        )

    async def _split_compacted_prefix(
        self,
        stored: Conversation,
        partial: list[_PartialMessage],
    ) -> tuple[bool, list[_PartialMessage]]:
        """Reconstruct the compacted-away prefix from the parent chain and compare
        it against the first `stored.raw_message_start_index` non-deleted entries
        of incoming.

        Returns `(True, raw_suffix)` when every `(source_id, author_id, content)`
        triple matches — strip the prefix before reconcile. Returns
        `(False, partial)` on any mismatch (length, source_id, author_id, content)
        — the caller routes to Case B (fresh Conversation, no inherited compacted
        state).

        `source_id` is included in the comparison deliberately: accepting different
        source-ids with the same content would silently weaken the
        source-ID-as-identity invariant the rest of the system relies on.
        """
        n = stored.raw_message_start_index
        if n <= 0:
            return True, partial
        if len(partial) < n:
            return False, partial
        expected_prefix = await self._reconstruct_compacted_prefix(stored)
        if len(expected_prefix) != n:
            return False, partial
        for expected, incoming in zip(expected_prefix, partial[:n], strict=True):
            if incoming.source_id != expected.source_id:
                return False, partial
            if incoming.author_id != expected.author_id:
                return False, partial
            if incoming.content != expected.content:
                return False, partial
        return True, partial[n:]

    async def _reconstruct_compacted_prefix(
        self,
        stored: Conversation,
    ) -> list[ConversationMessage]:
        """Walk the parent chain to return the first `stored.raw_message_start_index`
        non-deleted ConversationMessages from the global conversation history.

        Returns an empty list when the parent chain can't be walked (missing parent
        doc, missing messages_json) so the caller treats the split as failed.
        """
        n = stored.raw_message_start_index
        if n <= 0 or not stored.parent_snapshot_uuid:
            return []
        parent_doc = await self.search_client.get_conversation(stored.parent_snapshot_uuid)
        if parent_doc is None:
            return []
        boundary = await self._boundary_messages_for_doc(parent_doc)
        return boundary[:n]

    async def _boundary_messages_for_conversation(
        self,
        conversation: Conversation,
    ) -> list[ConversationMessage]:
        """Return `prefix + this snapshot's non-deleted messages` for boundary
        hashing. Walks the parent chain when this snapshot has a non-zero
        `raw_message_start_index`."""
        prefix: list[ConversationMessage] = []
        if conversation.parent_snapshot_uuid and conversation.raw_message_start_index > 0:
            parent_doc = await self.search_client.get_conversation(conversation.parent_snapshot_uuid)
            if parent_doc:
                parent_prefix = await self._boundary_messages_for_doc(parent_doc)
                prefix = parent_prefix[: conversation.raw_message_start_index]
        return prefix + conversation_message_items(conversation.messages)

    async def _boundary_messages_for_doc(
        self,
        doc: dict,
        memo: dict[str, list[ConversationMessage]] | None = None,
    ) -> list[ConversationMessage]:
        memo = memo or {}
        key = doc.get("snapshot_uuid")
        if key in memo:
            return memo[key]
        prefix: list[ConversationMessage] = []
        raw_start = doc.get("raw_message_start_index") or 0
        parent_uuid = doc.get("parent_snapshot_uuid")
        if parent_uuid and raw_start > 0:
            parent_doc = await self.search_client.get_conversation(parent_uuid)
            if parent_doc:
                parent_prefix = await self._boundary_messages_for_doc(parent_doc, memo)
                prefix = parent_prefix[:raw_start]
        own_messages = conversation_message_items(messages_from_doc(doc))
        result = prefix + own_messages
        if key:
            memo[key] = result
        return result

    async def _apply_result(
        self,
        *,
        stored: Conversation,
        result,
        normalized: list[NormalizedMessage],
        bot_author_id: str,
        conversation_uuid: str,
    ) -> Conversation:
        """Default (web) apply policy: `match` and `append` mutate in place;
        `edit`, `delete`, and `divergence` all branch to a new snapshot via
        `_apply_divergence`. This preserves the snapshot-DAG branch contract —
        the original snapshot stays intact in ES; the new branch carries the
        edit/delete/divergence.

        Slack overrides this in `SlackConversationSyncerMixin` to mutate in
        place (Slack threads are linear and authoritative).
        """
        if result.classification == "match":
            return stored
        if result.classification == "append":
            for op in result.operations:
                if op.kind == "append" and op.incoming is not None:
                    stored.messages.append(_normalized_to_message(op.incoming))
            return stored
        if result.classification in {"edit", "delete", "divergence"}:
            return await self._apply_divergence(
                stored=stored,
                result=result,
                normalized=normalized,
                bot_author_id=bot_author_id,
                conversation_uuid=conversation_uuid,
            )
        raise ValueError(f"Unknown reconcile classification: {result.classification!r}")

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
        # Build shared_messages in shared_prefix_source_ids order — not by set
        # membership over stored.messages — so the branch snapshot preserves
        # source-id ordering even if stored.messages's underlying order is ever
        # imperfect.
        stored_by_id = {m.source_id: m for m in stored.messages}
        shared_messages = [
            stored_by_id[sid] for sid in result.shared_prefix_source_ids if sid in stored_by_id
        ]
        shared_ids = set(result.shared_prefix_source_ids)
        normalized_by_id = {m.source_id: m for m in normalized}
        new_tail_normalized = [
            normalized_by_id[m_id] for m_id in [n.source_id for n in normalized] if m_id not in shared_ids
        ]
        new_tail = [_normalized_to_message(m) for m in new_tail_normalized]

        # Case A lifted-state recompute: load the shared prefix's bot
        # `TurnExecution`s, derive active paths, filter the parent's lifted
        # pairs to those paths, and pick the anchor as the first bot in the
        # shared prefix with non-stale file-tool activity. Invariant:
        # `anchor=None iff lifted_turn_items == []`.
        shared_bot_ids = [
            m.source_id
            for m in shared_messages
            if not m.deleted and m.author_id == bot_author_id
        ]
        shared_turns: dict[str, TurnExecution] = {}
        if shared_bot_ids:
            shared_turns = await self.search_client.get_turn_executions(
                stored.conversation_uuid, shared_bot_ids
            )
        child_active_paths = _active_paths_in_turns(shared_turns)
        lifted = _filter_lifted_pairs_by_paths(stored.lifted_turn_items, child_active_paths)
        anchor = (
            _first_bot_with_file_activity(shared_messages, shared_turns, bot_author_id)
            if lifted
            else None
        )

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

    async def load_assistant_index(self, conversation_uuid: str) -> dict[str, str]:
        """Return a `{bot_source_id: sha256(content)}` mapping for every
        assistant `ConversationMessage` reachable in the snapshot DAG for
        `conversation_uuid`.

        Cached in Redis at `assistant_index:{conversation_uuid}` with the same
        TTL as conversation snapshots. On miss, rebuild by walking every
        conversation doc (compacted ancestors + branch siblings) and indexing
        their bot messages.

        When the same `source_id` appears across multiple snapshots (branches
        duplicate the shared prefix) the index records one content_hash per
        source_id — last write wins. That's safe under the assumption that bot
        messages aren't edited per-branch; if a future feature changes that
        the index will need to become per-branch.
        """
        cached = await self.redis_client.get(f"assistant_index:{conversation_uuid}")
        if cached is not None:
            try:
                raw = cached.decode("utf-8") if isinstance(cached, bytes) else cached
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass

        docs = await self.search_client.find_all_conversation_docs(conversation_uuid)
        index: dict[str, str] = {}
        for doc in docs:
            bot_author_id = doc.get("bot_author_id")
            if not bot_author_id:
                continue
            for stored_msg in messages_from_doc(doc):
                if stored_msg.deleted:
                    continue
                if stored_msg.author_id != bot_author_id:
                    continue
                index[stored_msg.source_id] = _hash_content(stored_msg.content)

        await self.redis_client.set(
            f"assistant_index:{conversation_uuid}",
            json.dumps(index),
            ex=self.conversation_cache_ex,
        )
        return index

    async def refresh_assistant_index_with(
        self,
        conversation_uuid: str,
        source_id: str,
        content: str,
    ) -> None:
        """Add (or replace) a single bot message in the cached assistant index.
        Called by `finalize_turn` immediately after committing the bot's final
        `ConversationMessage` to ES."""
        index = await self.load_assistant_index(conversation_uuid)
        index[source_id] = _hash_content(content)
        await self.redis_client.set(
            f"assistant_index:{conversation_uuid}",
            json.dumps(index),
            ex=self.conversation_cache_ex,
        )

    async def validate_assistant_messages(
        self,
        conversation_uuid: str,
        incoming: list[IncomingMessage],
    ) -> None:
        """DAG-scoped guardrail for web clients.

        Raises `AssistantMessageGuardrailError` if any `role="assistant"` entry:
        - has no `source_id` (web assistants always must),
        - has a `source_id` not in the conversation's assistant index, or
        - has content whose hash doesn't match the indexed hash.

        No-op when `incoming` contains no assistant entries — the common case
        on the first turn or any pure-user POST.
        """
        if not any(msg.role == "assistant" for msg in incoming):
            return
        index = await self.load_assistant_index(conversation_uuid)
        for msg in incoming:
            if msg.role != "assistant":
                continue
            if msg.source_id is None:
                raise AssistantMessageGuardrailError(
                    "Assistant messages must carry a server-assigned source_id"
                )
            known_hash = index.get(msg.source_id)
            if known_hash is None:
                raise AssistantMessageGuardrailError(
                    f"Unknown assistant source_id: {msg.source_id}"
                )
            if known_hash != _hash_content(msg.content):
                raise AssistantMessageGuardrailError(
                    f"Assistant content mismatch for {msg.source_id}"
                )

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


class SlackConversationSyncerMixin:
    """Slack-flavored apply policy: edit, delete, and divergence all mutate
    the stored snapshot in place rather than branching.

    Slack threads are linear and authoritative — `message_changed` carries the
    same `ts`, `message_deleted` is a tombstone, and divergence indicates the
    stored snapshot drifted from Slack's view (recovery action is overwrite
    in place with a log warning, per the design doc).

    Delete also triggers tombstone re-keying: the deleted bot's `TurnExecution`
    is moved to the next non-tombstoned bot in the same consecutive run, and
    `Conversation.lifted_anchor_source_id` follows the same rule. See
    `_rekey_for_tombstone` below.

    Mixed in *before* `ConversationSyncer` in MRO so `_apply_result` overrides
    the default web behavior:

        class SlackSyncer(SlackConversationSyncerMixin, ConversationSyncer): ...
    """

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
                    if msg is None:
                        continue
                    msg.deleted = True
                    if msg.author_id == bot_author_id:
                        await self._rekey_for_tombstone(stored, op.source_id)
            return stored
        if result.classification == "divergence":
            # Stored snapshot drifted from Slack's authoritative thread — overwrite
            # in place with the incoming view. Surface a warning so operators see
            # this happened.
            logger.warning(
                "Slack syncer overwriting snapshot %s in place: reconcile returned divergence",
                stored.snapshot_uuid,
            )
            stored.messages = [_normalized_to_message(m) for m in normalized]
            return stored
        raise ValueError(f"Unknown reconcile classification: {result.classification!r}")

    async def _rekey_for_tombstone(
        self,
        stored: Conversation,
        deleted_source_id: str,
    ) -> None:
        """Move a tombstoned bot's `TurnExecution` ownership to the next
        non-tombstoned bot in the same consecutive run; same rule for
        `lifted_anchor_source_id`.

        Run membership: walk the bot run that contains `deleted_source_id`,
        stopping at non-bot messages. Tombstoned bots stay in the run for
        membership but are skipped for selection.

        If the run has no non-tombstoned bots left, the `TurnExecution` is
        orphaned (deleted from ES) and the anchor falls to `None` (which also
        clears `lifted_turn_items` to preserve the invariant
        `anchor=None iff lifted_turn_items == []`).
        """
        next_owner = _next_non_tombstoned_bot_in_run(stored, deleted_source_id)

        # Re-key the TurnExecution.
        try:
            existing = await self.search_client.get_turn_execution(deleted_source_id)
        except Exception:
            existing = None
        if existing is not None:
            if next_owner is not None:
                await self.search_client.rekey_turn_execution(deleted_source_id, next_owner)
            else:
                await self.search_client.delete_turn_execution(deleted_source_id)

        # Re-key the lifted anchor if it pointed at the deleted bot.
        if stored.lifted_anchor_source_id == deleted_source_id:
            if next_owner is not None:
                stored.lifted_anchor_source_id = next_owner
            else:
                stored.lifted_anchor_source_id = None
                stored.lifted_turn_items = []


def _next_non_tombstoned_bot_in_run(
    stored: Conversation,
    deleted_source_id: str,
) -> str | None:
    """Pick the replacement owner for a tombstoned bot's `TurnExecution`.

    Walks the bot run that contains `deleted_source_id` in `sorted_messages`
    order. Run boundaries: any non-bot message (a user / different-author
    message). Tombstoned bots are in the run for *membership* (they don't
    break the run) but are skipped for *selection*.

    Returns the source_id of the next non-tombstoned bot in the run, or `None`
    if the run is fully tombstoned (the orphan case).
    """
    sorted_msgs = stored.sorted_messages()
    bot_author_id = stored.bot_author_id

    # Find the deleted message's index in sorted order.
    deleted_idx: int | None = None
    for idx, m in enumerate(sorted_msgs):
        if m.source_id == deleted_source_id:
            deleted_idx = idx
            break
    if deleted_idx is None:
        return None
    # Defensive: the deleted message must be a bot for there to be a run to walk.
    # Callers (the Slack mixin's delete path) already guard on this, but if some
    # future caller hands us a user source_id we want a clean `None`.
    if sorted_msgs[deleted_idx].author_id != bot_author_id:
        return None

    # Walk forward through bot-author entries to find the first non-tombstoned one.
    for m in sorted_msgs[deleted_idx + 1 :]:
        if m.author_id != bot_author_id:
            break
        if not m.deleted:
            return m.source_id

    # Walk backward — but only through bots since the previous run ends at a non-bot.
    for m in reversed(sorted_msgs[:deleted_idx]):
        if m.author_id != bot_author_id:
            break
        if not m.deleted:
            return m.source_id

    return None


def _active_paths_in_turns(turns: dict[str, TurnExecution]) -> set[str]:
    """Paths active in a window: any `file_tool.path`-annotated TurnItem whose
    `file_tool.status` is not `"stale"`. Used by both the compactor's lift
    plan and the syncer's Case A branch-creation lifted-state recompute."""
    paths: set[str] = set()
    for turn in turns.values():
        for item in turn.items:
            ann = item.prokaryotes_annotations or {}
            if ann.get("file_tool.status") == "stale":
                continue
            path = ann.get("file_tool.path")
            if path:
                paths.add(path)
    return paths


def _first_bot_with_file_activity(
    messages: list[ConversationMessage],
    turns: dict[str, TurnExecution],
    bot_author_id: str,
) -> str | None:
    """Pick the source_id of the first non-tombstoned bot message in
    `messages` whose `TurnExecution` carries a non-stale file-tool annotation.

    Used as the lift anchor — projection emits `lifted_turn_items` just
    before this bot's tool round. Returns `None` if no qualifying bot exists.
    """
    for msg in messages:
        if msg.deleted or msg.author_id != bot_author_id:
            continue
        turn = turns.get(msg.source_id)
        if turn is None:
            continue
        for item in turn.items:
            ann = item.prokaryotes_annotations or {}
            if not ann.get("file_tool.path"):
                continue
            if ann.get("file_tool.status") == "stale":
                continue
            return msg.source_id
    return None


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


def _hash_content(content: str) -> str:
    """Stable per-message content hash used by the assistant-message guardrail."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
