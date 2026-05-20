"""Async wrappers around the Slack Web API.

`SlackClient` is a token-less wrapper over Slack's Web API. Every public method takes an explicit `bot_token`
keyword so a single client instance can, in principle, be shared; in practice one harness serves one workspace and
`SlackClientWithToken` curries the workspace's bot token onto every call.

All requests go through `_call`, which enforces the shared rate limiter described in the Slack-harness design:
a per-`(method, channel_id)` bucket at ~1 req/sec plus workspace-global `Retry-After` honoring on 429s. Slack
documents rate limits per method and per workspace and recommends staying around 1 request/sec; the limiter keeps
two streamers in different threads of the same channel from bursting Slack.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)


# Per-`(method, channel_id)` bucket spacing. Slack recommends ~1 request/sec for Web API calls; `chat.postMessage`
# is special-tier and `chat.update` has its own bucket, so a 1-second floor between consecutive calls in the same
# bucket keeps the harness comfortably under the documented limits.
_BUCKET_MIN_INTERVAL_SECONDS = 1.0

# Slack base URL — every method path is appended to this.
_SLACK_API_BASE_URL = "https://slack.com/api/"

# Default per-request timeout. Generous enough for a paginated `conversations.replies` page, short enough that a
# wedged call cannot hold a per-thread turn lock indefinitely (the harness also wraps turns in `wait_for`).
_HTTP_TIMEOUT_SECONDS = 20.0


class SlackApiError(Exception):
    """A Slack Web API method returned HTTP 200 with `ok=false`.

    Slack reports most application-level failures (`channel_not_found`, `message_not_found`, `not_in_channel`,
    `msg_too_long`, `cant_update_message`, …) as a 200 response carrying `{"ok": false, "error": ...}`, not an
    HTTP error status. `_call` raises this so a failed call surfaces as a failure instead of masquerading as
    empty data (`conversations.replies`) or degrading into a missing-field `KeyError` downstream
    (`chat.postMessage`). Best-effort callers — `_strip_in_flight_orphans`, `resolve_display_names`,
    `resolve_app_id`, the streamer's metadata cleanup — catch it and degrade; load-bearing calls let it
    propagate to the turn's failure path.
    """

    def __init__(self, method: str, error: str, response: dict):
        self.method = method
        self.error = error
        self.response = response
        super().__init__(f"Slack API call {method!r} failed: {error}")


class SlackClient:
    """Token-less async wrapper over the Slack Web API.

    Built on a single `httpx.AsyncClient` bound to `https://slack.com/api/`. Every public method takes an explicit
    `bot_token` keyword and routes through `_call`, which applies the shared rate limiter. `SlackClientWithToken`
    curries the bot token so per-turn code does not have to thread it through.
    """

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(base_url=_SLACK_API_BASE_URL, timeout=_HTTP_TIMEOUT_SECONDS)
        # Per-`(method, channel_id)` bucket: maps a bucket key to the monotonic timestamp at which the next call
        # in that bucket is allowed to start. `channel_id` is `None` for method-keyed (non-channel-scoped) calls.
        self._bucket_next_allowed: dict[tuple[str, str | None], float] = {}
        # Workspace-global per-method `Retry-After` floors. A 429 on a method pauses every later call of that
        # method workspace-wide until the recorded monotonic deadline passes.
        self._method_retry_after_until: dict[str, float] = {}
        # One lock serializes the limiter's read-modify-write of the bucket / retry-after maps so concurrent turns
        # cannot both read a stale "next allowed" timestamp and burst Slack.
        self._limiter_lock = asyncio.Lock()

    async def auth_test(self, *, bot_token: str) -> dict:
        """Call `auth.test` to resolve workspace identity (`team_id`, `user_id`, `bot_id`, `team`)."""
        return await self._call("auth.test", bot_token=bot_token, data={})

    async def chat_delete(self, *, bot_token: str, channel: str, ts: str) -> dict:
        """Delete a message previously posted by the bot."""
        return await self._call(
            "chat.delete",
            bot_token=bot_token,
            channel_id=channel,
            data={"channel": channel, "ts": ts},
        )

    async def chat_post_message(
        self,
        *,
        bot_token: str,
        channel: str,
        thread_ts: str | None = None,
        text: str | None = None,
        blocks: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Post a message into `channel` (optionally as a reply on `thread_ts`).

        `metadata` is sent as a JSON-encoded `metadata` field so the harness can stamp `prokaryotes_in_flight`
        markers on every post.
        """
        data: dict = {"channel": channel}
        if thread_ts is not None:
            data["thread_ts"] = thread_ts
        if text is not None:
            data["text"] = text
        if blocks is not None:
            data["blocks"] = json.dumps(blocks)
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        return await self._call("chat.postMessage", bot_token=bot_token, channel_id=channel, data=data)

    async def chat_update(
        self,
        *,
        bot_token: str,
        channel: str,
        ts: str,
        text: str | None = None,
        blocks: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Update an existing message identified by `(channel, ts)`.

        Passing `metadata={}` clears any previously stamped metadata — that is how `clear_in_flight_metadata`
        removes the `prokaryotes_in_flight` marker after a durable persist.
        """
        data: dict = {"channel": channel, "ts": ts}
        if text is not None:
            data["text"] = text
        if blocks is not None:
            data["blocks"] = json.dumps(blocks)
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        return await self._call("chat.update", bot_token=bot_token, channel_id=channel, data=data)

    async def close(self) -> None:
        """Close the underlying HTTP client. Call once at harness shutdown after the in-flight drain finishes."""
        await self._http.aclose()

    async def conversations_history(
        self,
        *,
        bot_token: str,
        channel: str,
        latest: str | None = None,
        oldest: str | None = None,
        inclusive: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch top-level channel messages. Used to build the pre-mention channel prelude.

        Returns the `messages` list (newest-first, as Slack returns it). This is a single-page fetch — the
        prelude only needs a short tail — so no cursor pagination is performed.
        """
        data: dict = {"channel": channel, "limit": limit}
        if latest is not None:
            data["latest"] = latest
        if oldest is not None:
            data["oldest"] = oldest
        if inclusive:
            data["inclusive"] = "true"
        response = await self._call("conversations.history", bot_token=bot_token, channel_id=channel, data=data)
        return response.get("messages", [])

    async def conversations_replies(
        self,
        *,
        bot_token: str,
        channel: str,
        ts: str,
        oldest: str | None = None,
        inclusive: bool = False,
        include_all_metadata: bool = False,
        paginate_until_ts: str | None = None,
    ) -> list[dict]:
        """Fetch every message in a thread, following Slack's `next_cursor` pagination internally.

        Returns the flattened `messages` list in chronological (oldest-first) order. `paginate_until_ts` caps
        pagination: once a fetched page's newest message reaches that timestamp, no further pages are requested.
        With `oldest` set to the active raw window's first `source_id` the fetch typically returns one short
        page; the cap keeps even a cold-recovery full-fetch on a very long thread bounded.
        """
        messages: list[dict] = []
        cursor: str | None = None
        while True:
            data: dict = {"channel": channel, "ts": ts, "limit": 200}
            if oldest is not None:
                data["oldest"] = oldest
            if inclusive:
                data["inclusive"] = "true"
            if include_all_metadata:
                data["include_all_metadata"] = "true"
            if cursor is not None:
                data["cursor"] = cursor
            response = await self._call("conversations.replies", bot_token=bot_token, channel_id=channel, data=data)
            page = response.get("messages", [])
            messages.extend(page)
            # Stop once the cap is crossed: the newest message on this page has reached `paginate_until_ts`, so
            # any later page can only hold messages past the cap.
            if paginate_until_ts is not None and page and page[-1].get("ts", "") >= paginate_until_ts:
                break
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        return messages

    async def resolve_app_id(self, *, bot_token: str, bot_user_id: str) -> str | None:
        """Resolve the bot's `app_id` via `users.info(bot_user_id).profile.api_app_id`.

        `app_id` is not on `auth.test`; one `users.info` call surfaces it. It is only a fallback own-bot identity
        check (the primary path is `bot_id` matching), so a lookup failure degrades to `None` rather than failing
        startup — `_should_handle` and `_slack_author_id` already guard `app_id is None`. Returns `None` when the
        field is absent or the lookup fails.
        """
        try:
            info = await self.users_info(bot_token=bot_token, user=bot_user_id)
        except SlackApiError:
            logger.warning("users.info failed resolving app_id for %s — app_id fallback disabled", bot_user_id)
            return None
        return info.get("user", {}).get("profile", {}).get("api_app_id")

    async def users_info(self, *, bot_token: str, user: str) -> dict:
        """Call `users.info` to resolve a user's profile (display name, `api_app_id`)."""
        return await self._call(
            "users.info",
            bot_token=bot_token,
            data={"user": user},
        )

    async def _call(self, method: str, *, bot_token: str, data: dict, channel_id: str | None = None) -> dict:
        """Issue one Slack Web API POST through the shared rate limiter.

        `channel_id` selects the bucket: channel-scoped methods (`chat.postMessage`, `chat.update`,
        `chat.delete`) bucket by `(method, channel_id)`; method-keyed calls (`auth.test`, `users.info`,
        `conversations.*`) pass `channel_id=None` and bucket by `(method, None)`. A 429 with `Retry-After`
        records a workspace-global floor for that method and the call is retried once the floor passes.
        """
        while True:
            await self._wait_for_slot(method, channel_id)
            response = await self._http.post(
                method,
                data={**data, "token": bot_token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if response.status_code == 429:
                # Workspace-global backoff: pause every later call of this method until `Retry-After` elapses,
                # then retry this call.
                retry_after = float(response.headers.get("Retry-After", "1"))
                logger.warning("Slack 429 on %s — backing off %.0fs workspace-wide", method, retry_after)
                async with self._limiter_lock:
                    self._method_retry_after_until[method] = time.monotonic() + retry_after
                continue
            response.raise_for_status()
            payload: dict = response.json()
            # Slack signals application-level failures as HTTP 200 + `ok=false`; surface them as an error rather
            # than handing back a body with no `messages` / `ts` for callers to misread as success.
            if not payload.get("ok", False):
                raise SlackApiError(method, payload.get("error", "unknown"), payload)
            return payload

    async def _wait_for_slot(self, method: str, channel_id: str | None) -> None:
        """Block until this `(method, channel_id)` bucket and the method's workspace-global floor both clear.

        Computes the sleep deadline under `_limiter_lock` and reserves the bucket's next slot, then sleeps
        outside the lock so concurrent calls to other buckets are not serialized behind this one's wait.
        """
        bucket = (method, channel_id)
        async with self._limiter_lock:
            now = time.monotonic()
            # Wait for the later of the per-bucket spacing floor and the workspace-global `Retry-After` floor.
            ready_at = max(
                self._bucket_next_allowed.get(bucket, now),
                self._method_retry_after_until.get(method, now),
            )
            start_at = max(now, ready_at)
            # Reserve this bucket's next slot so a concurrent caller queues behind us rather than racing.
            self._bucket_next_allowed[bucket] = start_at + _BUCKET_MIN_INTERVAL_SECONDS
        delay = start_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


class SlackClientWithToken:
    """Curries the harness's `bot_token` onto every `SlackClient` call.

    The per-turn code path (`sync_slack_thread`, `SlackStreamer`, prelude building) holds one of these so it does
    not have to thread the bot token through every call. Each method mirrors the corresponding `SlackClient`
    method with `bot_token` dropped.
    """

    def __init__(self, base: SlackClient, bot_token: str) -> None:
        self._base = base
        self._token = bot_token

    async def auth_test(self) -> dict:
        return await self._base.auth_test(bot_token=self._token)

    async def chat_delete(self, *, channel: str, ts: str) -> dict:
        return await self._base.chat_delete(bot_token=self._token, channel=channel, ts=ts)

    async def chat_post_message(
        self,
        *,
        channel: str,
        thread_ts: str | None = None,
        text: str | None = None,
        blocks: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        return await self._base.chat_post_message(
            bot_token=self._token,
            channel=channel,
            thread_ts=thread_ts,
            text=text,
            blocks=blocks,
            metadata=metadata,
        )

    async def chat_update(
        self,
        *,
        channel: str,
        ts: str,
        text: str | None = None,
        blocks: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        return await self._base.chat_update(
            bot_token=self._token,
            channel=channel,
            ts=ts,
            text=text,
            blocks=blocks,
            metadata=metadata,
        )

    async def conversations_history(
        self,
        *,
        channel: str,
        latest: str | None = None,
        oldest: str | None = None,
        inclusive: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        return await self._base.conversations_history(
            bot_token=self._token,
            channel=channel,
            latest=latest,
            oldest=oldest,
            inclusive=inclusive,
            limit=limit,
        )

    async def conversations_replies(
        self,
        *,
        channel: str,
        ts: str,
        oldest: str | None = None,
        inclusive: bool = False,
        include_all_metadata: bool = False,
        paginate_until_ts: str | None = None,
    ) -> list[dict]:
        return await self._base.conversations_replies(
            bot_token=self._token,
            channel=channel,
            ts=ts,
            oldest=oldest,
            inclusive=inclusive,
            include_all_metadata=include_all_metadata,
            paginate_until_ts=paginate_until_ts,
        )

    async def resolve_app_id(self, *, bot_user_id: str) -> str | None:
        return await self._base.resolve_app_id(bot_token=self._token, bot_user_id=bot_user_id)

    async def users_info(self, *, user: str) -> dict:
        return await self._base.users_info(bot_token=self._token, user=user)
