"""Source-ID helpers shared by the conversation sync path and the per-harness commit paths.

Every `ConversationMessage.source_id` is a `"<seconds>.<micros:06d>"` string that must be monotonically
increasing within a conversation. These helpers are the single source of truth for that format and for the
sorted-insert invariant on `Conversation.messages`.
"""

from __future__ import annotations

import bisect
import time

from prokaryotes.conversation_v1.models import ConversationMessage


def bump_source_id(source_id: str) -> str:
    """Return the next `source_id` after `source_id` (one microsecond later).

    Falls back to a wall-clock-derived `source_id` if `source_id` cannot be parsed as `"<seconds>.<micros>"`.
    """
    seconds_str, _, micros_str = source_id.partition(".")
    try:
        seconds = int(seconds_str)
        micros = int(micros_str or "0")
    except ValueError:
        return format_source_id_now()
    micros += 1
    if micros >= 1_000_000:
        seconds += 1
        micros = 0
    return f"{seconds}.{micros:06d}"


def format_source_id(ts: float) -> str:
    """Format a Unix timestamp as a `source_id`."""
    seconds = int(ts)
    micros = int((ts - seconds) * 1_000_000)
    return f"{seconds}.{micros:06d}"


def format_source_id_now() -> str:
    """Format the current wall-clock time as a `source_id`."""
    return format_source_id(time.time())


def insert_message_sorted(messages: list[ConversationMessage], message: ConversationMessage) -> None:
    """Insert `message` into `messages` at its `source_id`-sorted position.

    Same-thread turn serialization can deliver an append out of chronological order (a later mention's sync sees
    a prior turn's bot reply with a higher `ts` already committed under the lock), so a tail-append would leave
    the list unsorted and make the next turn's reconcile diverge needlessly. `bisect.bisect_right` against a
    parallel `source_id` key list keeps the insert ordered.
    """
    keys = [m.source_id for m in messages]
    index = bisect.bisect_right(keys, message.source_id)
    messages.insert(index, message)
