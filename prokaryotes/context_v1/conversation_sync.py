"""Three-tier conversation reconciliation: Redis fast path → exact ES load → ancestor-chain rebuild.

Reconciliation is source-ID-based via `prokaryotes.conversation_v1.reconcile`; the syncer applies the result
per-surface — divergence creates a new branch snapshot on web; the `SlackApplyPolicy` mixin overrides
`_apply_result` to mutate the stored snapshot in place for Slack.

Case A branch divergence and cold rebuild apply a two-gate filter on `working_file_windows`
(`_filter_windows_by_active_path_and_origin`):
- `active_paths` = `file_tool.path` annotations on file-tool `function_call_output`s in the kept TurnExecutions.
- `kept_call_ids` = file-tool call_ids in TurnExecutions kept by the new branch (shared-prefix turns or rebuild
  target's own turns).
- `source_call_ids` = file-tool call_ids in the source/donor snapshot's own TurnExecution.items (post-compaction
  tail only — not the parent-snapshot chain).
- Keep a window if its `path` is in `active_paths` AND (`window_id ∈ kept_call_ids` OR `window_id ∉
  source_call_ids`). The origin clause's second disjunct is the carryforward bucket — a window whose `call_id`
  lives nowhere in the source's own turns originated in a compacted ancestor and rode through one or more
  compactions. Windows from discarded sibling turns (or donor-tail turns past the rebuild target) are in
  `source_call_ids` but not in `kept_call_ids` and are dropped.

The web flow also handles two stream-loss recovery scenarios:
- Pre-commit (handshake-stamp invariant): managed by the client; the syncer produces a `snapshot_uuid` in the
  handshake even before the bot message commits so retries from the un-bot-replied user node extend the branch
  instead of re-diverging.
- Post-commit (resync handshake): if the stored snapshot has trailing bot-authored messages immediately after the
  last source_id shared with incoming and incoming has new content beyond that, the syncer emits an
  `unacknowledged_bot_messages` payload and does *not* start the LLM.
"""

from __future__ import annotations

import bisect
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
    WorkingFileWindow,
    compute_boundary_hash,
    conversation_message_items,
)
from prokaryotes.conversation_v1.reconcile import reconcile
from prokaryotes.search_v1 import SearchClient
from prokaryotes.search_v1.conversations import (
    conversation_from_doc,
    messages_from_doc,
    working_file_windows_from_doc,
)

logger = logging.getLogger(__name__)


class AssistantMessageGuardrailError(ValueError):
    """Raised by `validate_assistant_messages` when an incoming `role="assistant"` entry is unknown to the
    conversation DAG or its content doesn't match what storage has. Web route handlers should catch and convert to a
    400 response.
    """


@dataclass
class SourceIdAssignment:
    """One entry per request message that arrived without a `source_id`. `client_index` is the 0-based position in
    the request's messages array."""

    client_index: int
    source_id: str


@dataclass
class UnacknowledgedBotMessage:
    """A bot message the server has committed but the client hasn't seen, surfaced in the resync handshake. The
    client reconstructs the assistant node under `parent_source_id`."""

    source_id: str
    content: str
    parent_source_id: str


@dataclass
class SyncResult:
    """Output of `sync_conversation`.

    When `resync=True`, `conversation` is the unchanged stored snapshot and the caller is expected to close the
    stream without starting the LLM. The client reconstructs the missing bot history from
    `unacknowledged_bot_messages`, repairs its tree per the compose-mode split rule, and retries.

    When `is_new_branch=True`, the caller's response stream's handshake must carry the new snapshot's
    `snapshot_uuid` so the client can stamp it on the pending user node.
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

    Subclasses provide `conversation_cache_ex`, `redis_client`, `search_client`. Default `_apply_divergence` creates
    a new branch snapshot (web semantics). Slack subclasses can override to overwrite-in-place per the recovery
    rule.
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
        original_stored_snapshot_uuid = stored.snapshot_uuid

        if stored.raw_message_start_index > 0:
            prefix_match, raw_suffix = await self._split_compacted_prefix(stored, partial)
            if prefix_match:
                partial = raw_suffix
            else:
                stored = _fresh_conversation(conversation_uuid, bot_author_id)

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
            is_new_branch=(final.snapshot_uuid != original_stored_snapshot_uuid),
        )

    async def _split_compacted_prefix(
        self,
        stored: Conversation,
        partial: list[_PartialMessage],
    ) -> tuple[bool, list[_PartialMessage]]:
        """Reconstruct the compacted-away prefix from the parent chain and compare it against the first
        `stored.raw_message_start_index` non-deleted entries of incoming."""
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
        raw_start = doc["raw_message_start_index"]
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
        """Default (web) apply policy: `match` and `append` mutate in place; `edit`, `delete`, and `divergence` all
        branch to a new snapshot via `_apply_divergence`."""
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
        """Default (web) divergence behavior: create a new branch snapshot rooted at the shared prefix.

        Case A — divergence within the parent's raw window — inherits `ancestor_summaries` and
        `raw_message_start_index` verbatim, and applies the two-set origin filter on `working_file_windows`:
        `kept_call_ids` from shared-prefix TurnExecutions, `source_call_ids` from the source's own
        TurnExecutions (post-compaction tail only). The keep rule (`in kept` OR `not in source`) drops
        discarded-sibling-origin windows while preserving carryforward windows whose call_id is from a compacted
        ancestor.
        """
        stored_by_id = {m.source_id: m for m in stored.messages}
        shared_messages = [stored_by_id[sid] for sid in result.shared_prefix_source_ids if sid in stored_by_id]
        shared_ids = set(result.shared_prefix_source_ids)
        normalized_by_id = {m.source_id: m for m in normalized}
        new_tail_normalized = [
            normalized_by_id[m_id] for m_id in [n.source_id for n in normalized] if m_id not in shared_ids
        ]
        new_tail = [_normalized_to_message(m) for m in new_tail_normalized]

        source_bot_ids = [m.source_id for m in stored.messages if not m.deleted and m.author_id == bot_author_id]
        source_turns: dict[str, TurnExecution] = {}
        if source_bot_ids:
            source_turns = await self.search_client.get_turn_executions(stored.conversation_uuid, source_bot_ids)
        shared_bot_id_set = {m.source_id for m in shared_messages if not m.deleted and m.author_id == bot_author_id}
        kept_turns = {bid: turn for bid, turn in source_turns.items() if bid in shared_bot_id_set}

        kept_call_ids = _file_tool_call_ids_in(kept_turns)
        source_call_ids = _file_tool_call_ids_in(source_turns)
        # Active paths come from the kept turns' file-tool output annotations — those cover every call shape
        # (read, edit, redundant_read, conflict, ...), not just calls that minted a fresh `WorkingFileWindow`.
        # Combined with the origin filter, this drops a carryforward window for a path no longer touched in the
        # shared prefix while preserving a carryforward window for a path the shared prefix still touches (e.g.
        # via an edit that refreshed the existing window without minting a new one).
        child_active_paths = _active_paths_in_turns(kept_turns)
        carried_windows = _filter_windows_by_active_path_and_origin(
            stored.working_file_windows,
            active_paths=child_active_paths,
            kept_call_ids=kept_call_ids,
            source_call_ids=source_call_ids,
        )

        branch_messages = sorted(shared_messages + new_tail, key=lambda m: m.source_id)
        return Conversation(
            conversation_uuid=conversation_uuid,
            parent_snapshot_uuid=stored.snapshot_uuid,
            bot_author_id=bot_author_id,
            ancestor_summaries=list(stored.ancestor_summaries),
            raw_message_start_index=stored.raw_message_start_index,
            messages=branch_messages,
            working_file_windows=carried_windows,
        )

    def _assign_source_ids(self, partial: list[_PartialMessage], stored: Conversation) -> list[SourceIdAssignment]:
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
        cached = await self.redis_client.get(f"assistant_index:{conversation_uuid}")
        if cached is not None:
            try:
                data = json.loads(cached)
                if isinstance(data, dict):
                    return data
            except (UnicodeDecodeError, json.JSONDecodeError):
                logger.warning(
                    "Corrupt assistant_index cache for %s; rebuilding from search.",
                    conversation_uuid,
                    exc_info=True,
                )

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
        if not any(msg.role == "assistant" for msg in incoming):
            return
        index = await self.load_assistant_index(conversation_uuid)
        for msg in incoming:
            if msg.role != "assistant":
                continue
            if msg.source_id is None:
                raise AssistantMessageGuardrailError("Assistant messages must carry a server-assigned source_id")
            known_hash = index.get(msg.source_id)
            if known_hash is None:
                raise AssistantMessageGuardrailError(f"Unknown assistant source_id: {msg.source_id}")
            if known_hash != _hash_content(msg.content):
                raise AssistantMessageGuardrailError(f"Assistant content mismatch for {msg.source_id}")

    async def _load_exact_conversation(
        self,
        conversation_uuid: str,
        snapshot_uuid: str,
    ) -> tuple[Conversation | None, dict | None]:
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

        head_doc: dict | None = None
        if snapshot_uuid:
            exact, head_doc = await self._load_exact_conversation(conversation_uuid, snapshot_uuid)
            if exact is not None:
                return exact

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
        """Walk the snapshot DAG back to a compacted ancestor whose boundary_hash matches a prefix of incoming;
        build a Conversation rooted there.

        Working-file state is restored from the latest active descendant (donor) of the matched ancestor, then
        filtered through the two-set origin filter against the rebuild target's own TurnExecutions and the donor's
        own TurnExecutions (neither set walks the parent-snapshot chain). Windows opened by donor-tail turns past
        the target are dropped; carryforward windows whose call_id lives in no current TurnExecution are kept.
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
            boundary_count = doc["boundary_message_count"]
            boundary_hash = doc["boundary_hash"]
            if boundary_count == 0:
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

        raw_start = matched_ancestor["boundary_message_count"]
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

        # Restore working_file_windows from the latest active descendant (donor) and apply the two-set origin
        # filter. The donor's working_file_windows live on its snapshot doc; the donor's own TurnExecutions are
        # loaded for its non-deleted bot messages. The rebuild target's own turns are the bot messages in
        # echoed_messages.
        working_file_windows: list[WorkingFileWindow] = []
        descendant_doc = await self.search_client.find_latest_active_child(
            conversation_uuid, matched_ancestor["snapshot_uuid"]
        )
        if descendant_doc is not None:
            donor_windows = working_file_windows_from_doc(descendant_doc)
            donor_bot_ids = [
                m.source_id for m in messages_from_doc(descendant_doc) if not m.deleted and m.author_id == bot_author_id
            ]
            target_bot_ids = [m.source_id for m in echoed_messages if not m.deleted and m.author_id == bot_author_id]
            donor_turns: dict[str, TurnExecution] = {}
            if donor_bot_ids:
                donor_turns = await self.search_client.get_turn_executions(conversation_uuid, donor_bot_ids)
            target_turns: dict[str, TurnExecution] = {}
            if target_bot_ids:
                target_turns = await self.search_client.get_turn_executions(conversation_uuid, target_bot_ids)
            target_kept_call_ids = _file_tool_call_ids_in(target_turns)
            donor_call_ids = _file_tool_call_ids_in(donor_turns)
            target_active_paths = _active_paths_in_turns(target_turns)
            working_file_windows = _filter_windows_by_active_path_and_origin(
                donor_windows,
                active_paths=target_active_paths,
                kept_call_ids=target_kept_call_ids,
                source_call_ids=donor_call_ids,
            )

        return Conversation(
            conversation_uuid=conversation_uuid,
            parent_snapshot_uuid=matched_ancestor["snapshot_uuid"],
            bot_author_id=bot_author_id,
            ancestor_summaries=ancestor_summaries,
            raw_message_start_index=raw_start,
            messages=echoed_messages,
            working_file_windows=working_file_windows,
        )

    async def _walk_snapshot_chain(
        self,
        *,
        conversation_uuid: str,
        snapshot_uuid: str,
        head_doc: dict | None = None,
    ) -> list[dict]:
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


class SlackApplyPolicy:
    """Slack-flavored apply policy: edit, delete, and divergence all mutate the stored snapshot in place rather than
    branching.

    Delete still triggers tombstone re-keying for the deleted bot's `TurnExecution` (the next non-tombstoned bot in
    the same consecutive run becomes the new owner). Working-file windows aren't positioned by bot source_id, so
    no anchor re-key is needed.

    `stored.messages` is kept in `source_id`-sorted order as an invariant: same-thread turn serialization can
    deliver messages out of chronological order (a later mention's sync sees a prior turn's bot reply that landed
    under the lock with a higher `ts`), so every append uses `_insert_message_sorted` rather than a tail-append. A
    tail-append would leave `stored.messages` out of order and make the next turn's reconcile diverge needlessly.

    `_split_compacted_prefix` is intentionally not exercised by the Slack surface — the bounded `oldest` fetch in
    `sync_slack_thread` already excludes the compacted prefix, so invoking it would mis-classify every
    post-compaction turn. Slack subclasses keep `raw_message_start_index == 0` on the snapshots they reconcile.

    Mixed in *before* `ConversationSyncer` in MRO so `_apply_result` overrides the default web behavior.
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
                    _insert_message_sorted(stored.messages, _normalized_to_message(op.incoming))
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
            # For Slack, `divergence` is the normal classification for a turn whose operation set mixes kinds
            # (combined edit/delete/append) — `reconcile._classify` only shortcuts single-kind sets. Apply each
            # operation per its kind, reusing the same logic as the `append`/`edit`/`delete` branches above so
            # tombstone re-keying still runs for deleted bot messages.
            handled_kinds = {"append", "edit", "delete"}
            if all(op.kind in handled_kinds for op in result.operations):
                for op in result.operations:
                    if op.kind == "append" and op.incoming is not None:
                        _insert_message_sorted(stored.messages, _normalized_to_message(op.incoming))
                    elif op.kind == "edit" and op.incoming is not None:
                        msg = stored.message_by_source_id(op.source_id)
                        if msg is not None:
                            msg.content = op.incoming.content
                            msg.edited = True
                    elif op.kind == "delete":
                        msg = stored.message_by_source_id(op.source_id)
                        if msg is None:
                            continue
                        msg.deleted = True
                        if msg.author_id == bot_author_id:
                            await self._rekey_for_tombstone(stored, op.source_id)
                return stored
            # Defensive fallback: an operation kind outside {append, edit, delete} genuinely should not occur on
            # the Slack surface. Overwrite wholesale rather than silently dropping the operation.
            logger.warning(
                "Slack syncer overwriting snapshot %s in place: divergence carried an unhandled operation kind",
                stored.snapshot_uuid,
            )
            stored.messages = sorted((_normalized_to_message(m) for m in normalized), key=lambda m: m.source_id)
            return stored
        raise ValueError(f"Unknown reconcile classification: {result.classification!r}")

    async def _rekey_for_tombstone(
        self,
        stored: Conversation,
        deleted_source_id: str,
    ) -> None:
        """Move a tombstoned bot's `TurnExecution` ownership to the next non-tombstoned bot in the same consecutive
        run. If the run has no non-tombstoned bots left, the `TurnExecution` is orphaned (deleted from ES).

        Working-file windows are not anchored to a bot source_id under the first-class working-files design, so no
        anchor re-key happens here — only the TurnExecution ownership re-key.
        """
        next_owner = _next_non_tombstoned_bot_in_run(stored, deleted_source_id)

        existing = await self.search_client.get_turn_execution(stored.conversation_uuid, deleted_source_id)
        if existing is not None:
            if next_owner is not None:
                await self.search_client.rekey_turn_execution(stored.conversation_uuid, deleted_source_id, next_owner)
            else:
                await self.search_client.delete_turn_execution(stored.conversation_uuid, deleted_source_id)


def _file_tool_call_ids_in(turns: dict[str, TurnExecution]) -> set[str]:
    """Collect file-tool `function_call.call_id`s across the given `TurnExecution`s. Used by the two-set origin
    filter at branch divergence and cold rebuild."""
    call_ids: set[str] = set()
    for turn in turns.values():
        for item in turn.items:
            if item.type != "function_call":
                continue
            if item.name != "file_tool":
                continue
            call_id = item.call_id or item.id
            if call_id is not None:
                call_ids.add(call_id)
    return call_ids


def _filter_windows_by_origin(
    windows: list[WorkingFileWindow],
    *,
    kept_call_ids: set[str],
    source_call_ids: set[str],
) -> list[WorkingFileWindow]:
    """Two-set origin filter: keep a window if `window_id ∈ kept_call_ids` OR `window_id ∉ source_call_ids`. The
    second clause is the carryforward bucket — a window whose call_id appears nowhere in the source's own turns
    must have originated in a compacted ancestor and ridden forward, so keep it."""
    return [w for w in windows if w.window_id in kept_call_ids or w.window_id not in source_call_ids]


def _active_paths_in_turns(turns: dict[str, TurnExecution]) -> set[str]:
    """Active paths in a set of kept turns: every `file_tool.path` annotation present on a file-tool
    `function_call_output` in those turns.

    Reads outputs (not function_call arguments) because outputs carry the **resolved absolute** path that
    `WorkingFileWindow.path` is also stored as — derived from the same `_resolve_path(...)` call. Annotations
    cover every call shape: successful reads, RANGE_TRUNCATED, ALREADY_EXISTS, CONFLICT, RANGE_ERROR,
    REDUNDANT_READ (which doesn't mint a new window), and CREATED/EDITED records (which refresh existing windows
    without minting new ones). Using outputs as the source-of-truth for "path was touched in this turn" keeps the
    active-path derivation working for edits and redundant reads even when no new window was minted for the call.
    """
    paths: set[str] = set()
    for turn in turns.values():
        for item in turn.items:
            if item.type != "function_call_output":
                continue
            ann = item.prokaryotes_annotations or {}
            path = ann.get("file_tool.path")
            if path:
                paths.add(path)
    return paths


def _filter_windows_by_active_path_and_origin(
    windows: list[WorkingFileWindow],
    *,
    active_paths: set[str],
    kept_call_ids: set[str],
    source_call_ids: set[str],
) -> list[WorkingFileWindow]:
    """Two-gate filter at Case A divergence and cold rebuild: keep a window if `path ∈ active_paths` AND
    (`window_id ∈ kept_call_ids` OR `window_id ∉ source_call_ids`).

    The active-path gate drops carryforward windows whose path is no longer touched in the kept turns; the origin
    gate drops windows minted by discarded sibling/tail turns even when their path remains active.
    """
    return [
        w
        for w in windows
        if w.path in active_paths and (w.window_id in kept_call_ids or w.window_id not in source_call_ids)
    ]


def _insert_message_sorted(messages: list[ConversationMessage], message: ConversationMessage) -> None:
    """Insert `message` into `messages` at its `source_id`-sorted position, preserving the Slack-side
    `source_id`-sorted invariant on `stored.messages`.

    Same-thread turn serialization can deliver an append out of chronological order (a later mention's sync sees a
    prior turn's bot reply with a higher `ts` already committed under the lock), so a tail-append would leave the
    list unsorted and make the next turn's reconcile diverge needlessly. `bisect.insort` against a parallel
    `source_id` key list keeps the insert ordered.
    """
    keys = [m.source_id for m in messages]
    index = bisect.bisect_right(keys, message.source_id)
    messages.insert(index, message)


def _next_non_tombstoned_bot_in_run(
    stored: Conversation,
    deleted_source_id: str,
) -> str | None:
    sorted_msgs = stored.sorted_messages()
    bot_author_id = stored.bot_author_id

    deleted_idx: int | None = None
    for idx, m in enumerate(sorted_msgs):
        if m.source_id == deleted_source_id:
            deleted_idx = idx
            break
    if deleted_idx is None:
        return None
    if sorted_msgs[deleted_idx].author_id != bot_author_id:
        return None

    for m in sorted_msgs[deleted_idx + 1 :]:
        if m.author_id != bot_author_id:
            break
        if not m.deleted:
            return m.source_id

    for m in reversed(sorted_msgs[:deleted_idx]):
        if m.author_id != bot_author_id:
            break
        if not m.deleted:
            return m.source_id

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
    if client_snapshot_uuid is None:
        return True
    return (
        conversation.snapshot_uuid == client_snapshot_uuid or conversation.parent_snapshot_uuid == client_snapshot_uuid
    )


def _detect_unacknowledged_bot_messages(
    stored: Conversation,
    partial: list[_PartialMessage],
) -> list[UnacknowledgedBotMessage]:
    stored_sorted = stored.sorted_messages()
    stored_non_deleted = [m for m in stored_sorted if not m.deleted]
    if not stored_non_deleted:
        return []

    incoming_by_id = {m.source_id: m for m in partial if m.source_id is not None}

    last_shared_idx = -1
    for idx, msg in enumerate(stored_non_deleted):
        incoming = incoming_by_id.get(msg.source_id)
        if incoming is None:
            continue
        if incoming.content != msg.content:
            continue
        if incoming.author_id != msg.author_id:
            continue
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


def _format_source_id(ts: float) -> str:
    seconds = int(ts)
    micros = int((ts - seconds) * 1_000_000)
    return f"{seconds}.{micros:06d}"


def _hash_content(content: str) -> str:
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
