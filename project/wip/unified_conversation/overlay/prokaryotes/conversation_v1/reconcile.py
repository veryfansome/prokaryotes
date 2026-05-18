"""Source-ID-based reconciliation.

Replaces the position-based `ContextPartition.find_context_divergence` /
`ContextPartition.sync_from_conversation`. Returns a classification + operation
list; the syncer interprets per surface (Slack overwrites in place, web
branches on divergence).
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import (
    Conversation,
    NormalizedMessage,
    ReconcileOperation,
    ReconcileResult,
)


def reconcile(
    stored: Conversation,
    incoming: list[NormalizedMessage],
) -> ReconcileResult:
    """Diff `incoming` against `stored.messages` by `source_id` and classify.

    Preconditions:
    - Every incoming entry is a fully-populated `NormalizedMessage`. The syncer
      translates raw surface payloads — web `IncomingMessage` + session, or a
      Slack event — into `NormalizedMessage` and assigns `source_id`s to any
      newly-authored entries before calling this function.
    - `source_id` is sortable; lexicographic order equals chronological under
      the `seconds.microseconds` format.

    Classifications:
    - `match`: no operations.
    - `append`: every stored non-deleted source_id appears in incoming with
      matching content, and incoming has additional source_ids at the end.
    - `edit`: every stored non-deleted source_id appears in incoming, but at
      least one content differs (Slack `message_changed`).
    - `delete`: stored has non-deleted source_ids that are not in incoming, and
      incoming has no fresh source_ids and no content changes.
    - `divergence`: anything else — typically incoming has new source_ids that
      didn't exist in stored AND omits stored source_ids (web edit/regenerate),
      or mixes edits/deletes/appends in a non-trailing way.

    `divergence_point_index` indexes into `stored.messages` (including deleted
    entries) at the position immediately after the longest matching prefix —
    the natural rooting point for a child snapshot.
    """
    stored_messages = stored.sorted_messages()
    stored_non_deleted = [msg for msg in stored_messages if not msg.deleted]
    incoming_sorted = sorted(incoming, key=lambda m: m.source_id)

    stored_by_id = {msg.source_id: msg for msg in stored_non_deleted}
    incoming_by_id = {msg.source_id: msg for msg in incoming_sorted}

    shared_prefix_source_ids: list[str] = []
    for stored_msg in stored_non_deleted:
        incoming_msg = incoming_by_id.get(stored_msg.source_id)
        if incoming_msg is None or incoming_msg.content != stored_msg.content:
            break
        shared_prefix_source_ids.append(stored_msg.source_id)

    operations: list[ReconcileOperation] = []
    for stored_msg in stored_non_deleted:
        incoming_msg = incoming_by_id.get(stored_msg.source_id)
        if incoming_msg is None:
            operations.append(ReconcileOperation(kind="delete", source_id=stored_msg.source_id))
        elif incoming_msg.content != stored_msg.content:
            operations.append(
                ReconcileOperation(
                    kind="edit",
                    source_id=stored_msg.source_id,
                    incoming=incoming_msg,
                )
            )
    for incoming_msg in incoming_sorted:
        if incoming_msg.source_id not in stored_by_id:
            operations.append(
                ReconcileOperation(
                    kind="append",
                    source_id=incoming_msg.source_id,
                    incoming=incoming_msg,
                )
            )

    divergence_point_index = _index_after_prefix(stored_messages, shared_prefix_source_ids)

    classification = _classify(operations, stored_non_deleted, incoming_by_id, shared_prefix_source_ids)

    return ReconcileResult(
        classification=classification,
        operations=operations,
        shared_prefix_source_ids=shared_prefix_source_ids,
        divergence_point_index=divergence_point_index,
    )


def _classify(operations, stored_non_deleted, incoming_by_id, shared_prefix_source_ids):
    if not operations:
        return "match"

    kinds = {op.kind for op in operations}
    all_stored_in_incoming = all(msg.source_id in incoming_by_id for msg in stored_non_deleted)
    all_contents_match = all(
        incoming_by_id[msg.source_id].content == msg.content
        for msg in stored_non_deleted
        if msg.source_id in incoming_by_id
    )
    last_shared = shared_prefix_source_ids[-1] if shared_prefix_source_ids else None

    if kinds == {"append"} and all_stored_in_incoming and all_contents_match:
        return "append"
    if kinds == {"edit"} and all_stored_in_incoming:
        return "edit"
    if (
        kinds == {"delete"}
        and all_contents_match
        and last_shared is not None
        and all(op.source_id > last_shared for op in operations)
    ):
        return "delete"
    return "divergence"


def _index_after_prefix(stored_messages, shared_prefix_source_ids):
    """Return the index in `stored_messages` (including tombstones) just after
    the last shared source_id. Returns 0 if no prefix is shared, or `None` if
    `stored_messages` is empty."""
    if not stored_messages:
        return None
    if not shared_prefix_source_ids:
        return 0
    last_shared = shared_prefix_source_ids[-1]
    for idx, msg in enumerate(stored_messages):
        if msg.source_id == last_shared:
            return idx + 1
    return 0
