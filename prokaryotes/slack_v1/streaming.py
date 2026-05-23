"""Translate the `stream_turn` NDJSON event stream into Slack thread posts.

`SlackStreamer` consumes the NDJSON `stream_turn` emits and posts / updates Slack messages in a thread as text
streams in. `PostedMessage` is the per-post record `SlackStreamer.finish()` returns — `_finalize_slack_turn`,
`replay.py`, and `harness_v1/slack.py` all import it from here.

The streamer is a producer that hands posts / updates to `SlackClientWithToken` and awaits acknowledgement. The
1Hz flush interval is a *hint* used to batch updates; throughput is governed by `SlackClient`'s shared rate
limiter, not the streamer.
"""

from __future__ import annotations

import json
import logging
import time

from pydantic import BaseModel

from prokaryotes.slack_v1.client import SlackClientWithToken

logger = logging.getLogger(__name__)


class PostedMessage(BaseModel):
    """One Slack post the streamer made: `source_id` is the post's Slack `ts`, `content` is the exact text left
    in that post after paragraph splitting, `mrkdwn` formatting, and status-line removal.

    `SlackStreamer.finish()` returns a `list[PostedMessage]`; `_finalize_slack_turn` commits one bot
    `ConversationMessage` per `PostedMessage` with `content` stored verbatim, so the next `conversations.replies`
    fetch reconciles cleanly against the as-posted text.
    """

    source_id: str
    content: str


def _format_for_slack(text: str) -> str:
    """Translate the subset of GitHub-flavored Markdown that differs from Slack `mrkdwn`.

    Slack `mrkdwn` uses single `*` for bold and single `_` for italic; code fences and inline code render as-is.
    This is a best-effort translator — `**bold**` → `*bold*` — not a full Markdown parser; unhandled constructs
    (tables, headings) degrade to plain text, which Slack tolerates.
    """
    return text.replace("**", "*")


def _split_at_paragraph_boundary(text: str, soft_limit: int) -> tuple[str, str]:
    """Split `text` into a sealed head (<= `soft_limit`) and a remaining tail.

    Prefers a paragraph break (`\\n\\n`), then a line break (`\\n`), then a hard character cut so a single
    unbroken run still fits. Returns `(text, "")` when `text` already fits.
    """
    if len(text) <= soft_limit:
        return text, ""
    window = text[:soft_limit]
    boundary = window.rfind("\n\n")
    if boundary <= 0:
        boundary = window.rfind("\n")
    if boundary <= 0:
        boundary = soft_limit
        return text[:boundary], text[boundary:]
    # `boundary` points at the start of the break — drop the break itself from both sides.
    head = text[:boundary]
    tail = text[boundary:].lstrip("\n")
    return head, tail


class SlackStreamer:
    """Consume the `stream_turn` NDJSON event stream and post / update Slack thread messages.

    Behavior:
    - `post_placeholder` posts a placeholder thread reply (`PLACEHOLDER_TEXT`) so the user gets immediate
      feedback. Every post / update the streamer makes carries `prokaryotes_in_flight` Slack message metadata
      (turn / conversation / channel / thread identity) so a crashed turn's orphan posts can be recovered.
    - `consume` buffers `text_delta` chunks and hints a flush at most every `FLUSH_INTERVAL_SECONDS` or when the
      buffer crosses `FLUSH_CHARS`; `SlackClient`'s shared limiter is the actual gate.
    - When buffered text would exceed `SLACK_MESSAGE_SOFT_LIMIT`, the current message is sealed at the last
      paragraph break and a fresh continuation message is posted. Continuations do not repeat the `<@user>`
      prefix — only the first message in a reply carries it — but they do carry the same in-flight metadata.
    - `tool_call` / `progress_message` events update an ephemeral italic status line on the most recent message
      rather than appending to the body. The status line is rewritten on each event and stripped at end-of-stream.
    - `finish` does a final flush, strips any status line, and returns a `list[PostedMessage]` — the turn's
      authoritative record. If the model streamed no text, `finish` rewrites the placeholder in place to
      `EMPTY_REPLY_NOTICE` and returns it as a single `PostedMessage`; it never returns an empty list.
    - `fail` is robust to partial post state: it rewrites the placeholder to `FAILURE_NOTICE`, `chat.delete`s any
      continuations, and returns the placeholder as a single `PostedMessage`. It recovers a Slack-accepted but
      streamer-unrecorded placeholder via `conversations.replies` keyed on the in-flight `turn_id`, and returns
      `None` when no placeholder ever made it to Slack.
    - `clear_in_flight_metadata` best-effort `chat.update`s each post to drop the `prokaryotes_in_flight`
      metadata after the durable persist.
    """

    FLUSH_INTERVAL_SECONDS = 1.0
    FLUSH_CHARS = 1500
    SLACK_MESSAGE_SOFT_LIMIT = 3500
    # Canonical bot-message texts. Posted verbatim and matched verbatim by reconcile on the next replay, so they
    # must be stable strings.
    PLACEHOLDER_TEXT = "_…working_"
    FAILURE_NOTICE = "_The agent hit an error and could not finish this turn._"
    EMPTY_REPLY_NOTICE = "_The agent finished without producing a reply._"

    def __init__(
        self,
        *,
        channel_id: str,
        conversation_uuid: str,
        slack_client: SlackClientWithToken,
        thread_ts: str,
        turn_id: str,
        reply_to_user_id: str | None = None,
    ) -> None:
        self._channel_id = channel_id
        self._conversation_uuid = conversation_uuid
        self._slack_client = slack_client
        self._thread_ts = thread_ts
        self._turn_id = turn_id
        # `<@user>` notification prefix — set in channels and mpim, `None` in DMs (which post unprefixed).
        self._reply_to_user_id = reply_to_user_id

        # `_posts[i]` is the `ts` of the i-th Slack message of this reply; `_posts[0]` is the placeholder.
        self._posts: list[str] = []
        # Body text per post, parallel to `_posts`. The status line is NOT included here — it is appended only
        # at flush time and stripped before recording, so `_bodies` always holds clean reply content.
        self._bodies: list[str] = []
        # Text streamed in but not yet flushed to a Slack message.
        self._buffer = ""
        # Current ephemeral status line (`tool_call` / `progress_message`), shown italicized on the last post.
        self._status_line: str | None = None
        # Monotonic timestamp of the last flush — gates the ~1Hz flush hint.
        self._last_flush_monotonic = 0.0
        # True once any `text_delta` arrived; drives the empty-output branch in `finish`.
        self._saw_text = False

    async def clear_in_flight_metadata(self, posted: list[PostedMessage]) -> None:
        """Best-effort remove the `prokaryotes_in_flight` metadata from each post after the durable persist.

        Runs after `_finalize_slack_turn`. A failure here is harmless — the next replay's orphan pre-pass sees
        the post's `source_id` is in stored and preserves the message — so each failure is logged and swallowed.
        """
        for post in posted:
            try:
                await self._slack_client.chat_update(
                    channel=self._channel_id,
                    ts=post.source_id,
                    text=post.content,
                    metadata={},
                )
            except Exception:
                logger.warning(
                    "failed to clear in-flight metadata for ts=%s (turn_id=%s) — next replay will re-clear",
                    post.source_id,
                    self._turn_id,
                )

    async def consume(self, ndjson_line: str) -> None:
        """Consume one NDJSON line from `stream_turn`.

        Slack consumes `stream_turn`'s NDJSON directly — `text_delta` / `tool_call` / `progress_message` /
        `context_pct`. The `handshake` / `bot_message` / `compaction_pending` events belong to
        `stream_and_finalize`, which Slack does not use; unknown events are ignored for forward compatibility.
        """
        event = json.loads(ndjson_line)
        match event:
            case {"text_delta": chunk}:
                await self._append_text(chunk)
            case {"tool_call": payload}:
                await self._render_tool_call(payload)
            case {"progress_message": payload}:
                await self._render_progress(payload)
            case {"context_pct": _}:
                pass  # usage signal — Slack renders no context meter
            case _:
                pass  # forward-compat: ignore unknown events

    async def fail(self) -> PostedMessage | None:
        """Collapse whatever was posted into a single `FAILURE_NOTICE` post and return it.

        Robust to partial post state:
        - placeholder posted → rewrite it to `FAILURE_NOTICE`, `chat.delete` any continuations, return it.
        - placeholder accepted by Slack but its `ts` not recorded (rare) → recover the `ts` via
          `conversations.replies` keyed on the in-flight `turn_id`, then proceed as above.
        - no placeholder ever reached Slack → return `None`; the caller skips `_finalize_slack_turn` entirely.

        The returned notice (when non-`None`) is persisted by `_run_turn` so the next replay treats it as
        `match`, not an orphan to delete.
        """
        if not self._posts:
            recovered_ts = await self._recover_placeholder_ts()
            if recovered_ts is None:
                return None
            self._posts.append(recovered_ts)
            self._bodies.append("")

        # Delete every continuation post — the failure collapses to one message on the placeholder.
        for ts in self._posts[1:]:
            try:
                await self._slack_client.chat_delete(channel=self._channel_id, ts=ts)
            except Exception:
                logger.warning("failed to delete continuation ts=%s during fail (turn_id=%s)", ts, self._turn_id)
        del self._posts[1:]
        del self._bodies[1:]

        placeholder_ts = self._posts[0]
        await self._slack_client.chat_update(
            channel=self._channel_id,
            ts=placeholder_ts,
            text=self.FAILURE_NOTICE,
            metadata=self._in_flight_metadata(),
        )
        self._bodies[0] = self.FAILURE_NOTICE
        return PostedMessage(source_id=placeholder_ts, content=self.FAILURE_NOTICE)

    async def finish(self) -> list[PostedMessage]:
        """Flush any buffered text, strip the status line, and return the turn's authoritative post record.

        When the model streamed no `text_delta`, rewrites the placeholder in place to `EMPTY_REPLY_NOTICE` and
        returns it as a single `PostedMessage` — never an empty list.
        """
        if self._buffer:
            await self._flush(force=True)
        # Drop the ephemeral status line and rewrite each post to its clean body.
        self._status_line = None
        for index, ts in enumerate(self._posts):
            await self._slack_client.chat_update(
                channel=self._channel_id,
                ts=ts,
                text=self._render_post_text(index),
                metadata=self._in_flight_metadata(),
            )

        if not self._saw_text:
            # Empty model output — rewrite the placeholder to the empty-reply notice so the turn still has a
            # stored bot message and the placeholder is never left dangling with in-flight metadata.
            placeholder_ts = self._posts[0]
            await self._slack_client.chat_update(
                channel=self._channel_id,
                ts=placeholder_ts,
                text=self.EMPTY_REPLY_NOTICE,
                metadata=self._in_flight_metadata(),
            )
            self._bodies[0] = self.EMPTY_REPLY_NOTICE
            return [PostedMessage(source_id=placeholder_ts, content=self.EMPTY_REPLY_NOTICE)]

        return [
            PostedMessage(source_id=ts, content=self._render_post_text(index)) for index, ts in enumerate(self._posts)
        ]

    async def post_placeholder(self) -> None:
        """Post the placeholder thread reply so the user gets immediate feedback.

        The placeholder carries the `<@user>` prefix in channels / mpim and the `prokaryotes_in_flight`
        metadata. It becomes `_posts[0]` — the message every later flush, `finish`, and `fail` rewrites.
        """
        text = self._with_mention_prefix(self.PLACEHOLDER_TEXT, is_first=True)
        response = await self._slack_client.chat_post_message(
            channel=self._channel_id,
            thread_ts=self._thread_ts,
            text=text,
            metadata=self._in_flight_metadata(),
        )
        self._posts.append(response["ts"])
        self._bodies.append("")
        self._last_flush_monotonic = time.monotonic()

    async def _append_text(self, chunk: str) -> None:
        """Buffer a streamed text chunk and flush if the ~1Hz / `FLUSH_CHARS` hint fires."""
        self._saw_text = True
        self._buffer += chunk
        now = time.monotonic()
        should_flush = (
            len(self._buffer) >= self.FLUSH_CHARS or now - self._last_flush_monotonic >= self.FLUSH_INTERVAL_SECONDS
        )
        if should_flush:
            await self._flush(force=False)

    async def _flush(self, *, force: bool) -> None:
        """Write buffered text into the current Slack post, splitting into continuations past the soft limit.

        `force` is informational — the limiter in `SlackClient` is the real gate — but a non-forced flush with
        an empty buffer is a no-op.
        """
        if not self._buffer and not force:
            return
        # Merge the buffer into the last post's body, then peel off as many full posts as the soft limit allows.
        combined = self._bodies[-1] + self._buffer
        self._buffer = ""
        head, tail = _split_at_paragraph_boundary(_format_for_slack(combined), self.SLACK_MESSAGE_SOFT_LIMIT)
        self._bodies[-1] = head
        await self._slack_client.chat_update(
            channel=self._channel_id,
            ts=self._posts[-1],
            text=self._render_post_text(len(self._posts) - 1),
            metadata=self._in_flight_metadata(),
        )
        # Any overflow becomes one or more continuation posts.
        while tail:
            head, tail = _split_at_paragraph_boundary(tail, self.SLACK_MESSAGE_SOFT_LIMIT)
            response = await self._slack_client.chat_post_message(
                channel=self._channel_id,
                thread_ts=self._thread_ts,
                text=head,
                metadata=self._in_flight_metadata(),
            )
            self._posts.append(response["ts"])
            self._bodies.append(head)
        self._last_flush_monotonic = time.monotonic()

    def _in_flight_metadata(self) -> dict:
        """Build the `prokaryotes_in_flight` Slack message metadata stamped on every post the streamer makes.

        The `(turn_id, source_id-in-stored)` pair is the load-bearing primitive of the orphan-placeholder
        recovery contract — see `_strip_in_flight_orphans` and the failure-window matrix.
        """
        return {
            "event_type": "prokaryotes_in_flight",
            "event_payload": {
                "turn_id": self._turn_id,
                "conversation_uuid": self._conversation_uuid,
                "channel_id": self._channel_id,
                "thread_ts": self._thread_ts,
            },
        }

    async def _recover_placeholder_ts(self) -> str | None:
        """Recover a Slack-accepted but streamer-unrecorded placeholder `ts` for the `fail` path.

        Scans `conversations.replies` (thread replies do not appear in `conversations.history`) for a message
        carrying this turn's `prokaryotes_in_flight` metadata. Returns the `ts` or `None` if none is found.
        """
        try:
            thread = await self._slack_client.conversations_replies(
                channel=self._channel_id,
                ts=self._thread_ts,
                include_all_metadata=True,
            )
        except Exception:
            logger.warning("placeholder recovery lookup failed for turn_id=%s", self._turn_id)
            return None
        for message in thread:
            metadata = message.get("metadata") or {}
            payload = metadata.get("event_payload") or {}
            if metadata.get("event_type") == "prokaryotes_in_flight" and payload.get("turn_id") == self._turn_id:
                return message.get("ts")
        return None

    async def _render_progress(self, payload: dict) -> None:
        """Render a `progress_message` event as the ephemeral status line on the most recent post."""
        message = payload.get("message") or payload.get("text") or ""
        await self._update_status_line(f"_{message}_" if message else None)

    async def _render_tool_call(self, payload: dict) -> None:
        """Render a `tool_call` event as the ephemeral status line on the most recent post."""
        name = payload.get("name") or payload.get("tool") or "tool"
        await self._update_status_line(f"_…running {name}_")

    def _render_post_text(self, index: int) -> str:
        """Compose the wire text for post `index`: mention prefix (first post only), body, and status line.

        The status line is appended only to the *last* post and only while streaming; `finish` clears it before
        the final rewrite so stored content is clean.
        """
        body = self._with_mention_prefix(self._bodies[index], is_first=index == 0)
        if self._status_line is not None and index == len(self._posts) - 1:
            body = f"{body}\n\n{self._status_line}" if body else self._status_line
        # A first post with no body yet still shows the placeholder text so the user is not staring at a blank.
        return body or self._with_mention_prefix(self.PLACEHOLDER_TEXT, is_first=index == 0)

    async def _update_status_line(self, status_line: str | None) -> None:
        """Set the ephemeral status line and rewrite the most recent post to show it."""
        if not self._posts:
            return
        self._status_line = status_line
        await self._slack_client.chat_update(
            channel=self._channel_id,
            ts=self._posts[-1],
            text=self._render_post_text(len(self._posts) - 1),
            metadata=self._in_flight_metadata(),
        )

    def _with_mention_prefix(self, text: str, *, is_first: bool) -> str:
        """Prepend `<@user> ` to the first post of a reply in channels / mpim; leave continuations and DMs bare."""
        if is_first and self._reply_to_user_id is not None:
            return f"<@{self._reply_to_user_id}> {text}"
        return text
