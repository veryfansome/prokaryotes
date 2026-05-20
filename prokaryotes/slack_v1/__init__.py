"""`SlackBase` — Socket Mode lifecycle and inbound event dispatch for the Slack harness.

`SlackBase(SlackConversationSyncerMixin, HarnessBase, ABC)` owns the single per-workspace Socket Mode connection,
acks envelopes inside the listener, dedupes by `event_id`, applies the trigger gate (`_should_handle`), and fires
`handle_event` into a background task. `SlackConversationSyncerMixin` precedes `HarnessBase` in the MRO so its
in-place `_apply_result` (edit / delete / op-aware divergence, `TurnExecution` re-keying) overrides the default
web branch-on-divergence policy.

The LLM / tool wiring and the per-turn flow live in `prokaryotes.harness_v1.slack.SlackHarness`.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import certifi
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

from prokaryotes.context_v1.conversation_sync import SlackConversationSyncerMixin
from prokaryotes.harness_v1.base import HarnessBase
from prokaryotes.slack_v1.client import SlackClient

logger = logging.getLogger(__name__)

# `_turn_locks` reclamation tuning. The idle threshold is deliberately far longer than any turn could run, so the
# sweep only ever removes truly cold entries.
_TURN_LOCK_IDLE_SECONDS = 2 * 24 * 60 * 60  # reclaim locks untouched for 2 days
_TURN_LOCK_SWEEP_SECONDS = 60 * 60  # run the sweep at most hourly


def build_socket_mode_client(app_token: str) -> SocketModeClient:
    """Construct the `SocketModeClient` bound to the workspace's app-level (`xapp-`) token.

    A thin constructor seam so `SlackBase.on_start` and tests share one place that decides how the Socket Mode
    client is built. An explicit certifi-backed `ssl.SSLContext` is threaded onto the inner `AsyncWebClient` so
    both `apps.connections.open` and the Socket Mode WebSocket use a populated CA bundle — aiohttp falls back to
    the system trust store by default, which is empty on the python.org macOS installer until
    `Install Certificates.command` is run, producing an infinite retry loop on `CERTIFICATE_VERIFY_FAILED`.
    """
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    return SocketModeClient(app_token=app_token, web_client=AsyncWebClient(ssl=ssl_context))


@dataclass
class _LockEntry:
    """Per-`conversation_uuid` turn lock plus its last-use timestamp.

    `last_used_monotonic` is refreshed on every turn that touches the entry; the coarse `_sweep_turn_locks` pass
    reclaims entries that are both unlocked and idle past `_TURN_LOCK_IDLE_SECONDS`. No refcount, no per-release
    bookkeeping.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used_monotonic: float = field(default_factory=time.monotonic)


class SlackBase(SlackConversationSyncerMixin, HarnessBase, ABC):
    """Worker harness bound to a single Slack workspace.

    `SlackConversationSyncerMixin` precedes `HarnessBase` in the MRO so its `_apply_result` (in-place edit /
    delete / divergence, `TurnExecution` re-keying) overrides the default web branch-on-divergence policy.
    """

    def __init__(self, app_token: str, bot_token: str):
        super().__init__()
        self._app_token = app_token
        self._bot_token = bot_token
        self.slack_client = SlackClient()  # bot-API wrapper, token bound per call
        self.socket: SocketModeClient | None = None
        # Per-conversation_uuid turn locks — see "Serializing same-thread turns". Entries are never popped on
        # release; `_sweep_turn_locks` reclaims those idle past `_TURN_LOCK_IDLE_SECONDS`, bounding the dict over a
        # long run.
        self._turn_locks: dict[str, _LockEntry] = {}
        self._last_lock_sweep_monotonic = time.monotonic()

        # Resolved at startup via auth.test:
        self.team_id: str | None = None
        self.bot_user_id: str | None = None
        self.bot_id: str | None = None  # primary identifier for own-bot replay normalization
        self.team_name: str | None = None
        self.app_id: str | None = None  # fallback identifier — some bot posts include bot_profile.app_id only

    @staticmethod
    async def _await_finalize_through_cancel(coro):
        """Run `coro` as a Task and await it under `asyncio.shield`, re-entering the await whenever this coroutine
        is itself cancelled, until the task is `done()`. Then propagate whichever signal is more actionable — the
        finalize's exception if it raised, or `CancelledError` if it succeeded under cancellation.

        The only thing that cancels this coroutine is the per-turn `asyncio.wait_for(SLACK_TURN_TIMEOUT_SECONDS)`
        in `handle_event`, which fires once. The `while` loop is still required for that single cancellation:
        `await shield(task)` raises `CancelledError` the moment the cancellation lands — before the shielded
        finalize task is `done()` — so the loop re-enters the await and keeps waiting until the task completes.

        Why this instead of `asyncio.shield(coro)`: shield alone protects the inner task's lifetime but the outer
        `await shield(...)` still raises CancelledError on cancellation, releasing the per-thread lock and letting
        the next turn race the still-running commit.

        Exception ordering matters: if the finalize task raised AND we were cancelled, we surface the finalize
        exception. A storage failure (ES down, Redis CAS conflict, etc.) is the operationally meaningful signal —
        the outer cancellation will land again on the next await of the cancelled task. The reverse ordering
        (raise CancelledError first, swallow the finalize exception) would hide the actual durable-write failure
        behind a generic cancellation, making the failure matrix lie about persistence outcomes.
        """
        task = asyncio.create_task(coro)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        # task is done — `task.result()` re-raises any exception the task raised. If finalize raised, this
        # propagates that exception (and the recorded `cancelled` flag is intentionally dropped: the cancellation
        # will land again on the caller's next await of the cancelled task).
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError()
        return result

    async def _claim_event_id(self, event_id: str) -> bool:
        return bool(await self.redis_client.set(f"slack_event_seen:{event_id}", "1", ex=600, nx=True))

    @asynccontextmanager
    async def _conversation_turn_lock(self, conversation_uuid: str):
        """Serialize turns for one thread. v1: in-process `asyncio.Lock`, valid because one harness process serves
        one workspace. The Redis-lease swap for multi-container workspaces is localized to this method.

        Entries are not deleted on release. `last_used_monotonic` is refreshed on every use and
        `_sweep_turn_locks` reclaims cold entries — so there is no refcount discipline and no
        `Lock.release()`/queued-waiter resume race (the sweep only touches entries idle for days, never a
        contended lock).
        """
        self._sweep_turn_locks()
        entry = self._turn_locks.get(conversation_uuid)
        if entry is None:
            entry = _LockEntry()
            self._turn_locks[conversation_uuid] = entry
        entry.last_used_monotonic = time.monotonic()
        async with entry.lock:
            yield

    async def _dispatch_event(self, *, envelope: dict) -> None:
        if envelope.get("type") != "events_api":
            return
        payload = envelope["payload"]
        event_id = payload.get("event_id")
        if event_id and not await self._claim_event_id(event_id):
            return  # duplicate envelope delivery — Slack redelivers if ack timed out

        event = payload["event"]
        if event.get("type") in {"tokens_revoked", "app_uninstalled"}:
            logger.error("Slack app removed or token revoked for team_id=%s", self.team_id)
            return

        # Sanity check team_id; mismatch means the bot token was mis-configured against an app from a different
        # workspace.
        if payload.get("team_id") and payload["team_id"] != self.team_id:
            logger.warning(
                "Dropping event with team_id=%s — harness is bound to %s",
                payload["team_id"],
                self.team_id,
            )
            return

        if not self._should_handle(event):
            return

        self.background_and_forget(self.handle_event(event=event))

    async def _listener(self, client, request) -> None:
        # Ack first so Slack doesn't redeliver while we run the LLM turn.
        await client.send_socket_mode_response(SocketModeResponse(envelope_id=request.envelope_id))
        await self._dispatch_event(envelope=request.to_dict())

    def _should_handle(self, event: dict) -> bool:
        # Hidden events (message_replied and other metadata-only deliveries) carry no real message content and
        # should never reach handle_event. Slack flags them with hidden=true.
        if event.get("hidden"):
            return False
        # Required-field guard. handle_event and _run_turn assume these are present (channel for routing, ts for
        # source_id derivation, user for the originator display name, text for content). A malformed event
        # without them would KeyError mid-turn — drop it cleanly at the gate.
        if not (event.get("channel") and event.get("ts") and event.get("user") and event.get("text") is not None):
            return False

        # Bot self-guard runs ahead of the app_mention branch. A bot post can identify itself via any of three
        # fields (Slack's chat.postMessage and event shapes are not uniform):
        # - event["user"] for the standard user-attributed path
        # - event["bot_id"] for bot_message-subtype posts (no `user` field)
        # - event["bot_profile"]["app_id"] when bot_profile is included
        # Every comparison guards on `self.<field>` being non-None first — otherwise a failed startup resolution
        # (e.g. users.info couldn't find profile.api_app_id) would compare None == None and drop every normal
        # human event. The app_id check is strictly a fallback for events with no bot_id: multiple bot users can
        # share one Slack app (e.g. a test user-bot installed under the same app), so a mismatched-bot_id +
        # matching-app_id event is a *different* bot, not us — must not be filtered.
        if self.bot_user_id and event.get("user") == self.bot_user_id:
            return False
        if self.bot_id and event.get("bot_id") == self.bot_id:
            return False
        if (
            self.app_id
            and event.get("bot_id") is None
            and event.get("bot_profile", {}).get("app_id") == self.app_id
        ):
            return False
        if event.get("subtype") == "bot_message":
            return False  # defensive — bot_message from another integration also drops here
        if event.get("type") == "app_mention":
            # app_mention subtypes other than the plain message form (e.g. `document_mention` for Canvas
            # mentions) need product decisions we haven't made yet. Drop them rather than processing as if they
            # were normal thread messages — they carry different field shapes and would break downstream
            # assumptions.
            if event.get("subtype") is not None:
                return False
            return True
        if event.get("type") != "message":
            return False
        if event.get("subtype") is not None:
            return False
        if event.get("channel_type") == "im":
            return True  # every non-bot DM message — DMs are unambiguous
        return False  # all other non-mention channel/mpim/thread chatter — ignored

    def _sweep_turn_locks(self) -> None:
        """Reclaim cold `_turn_locks` entries. Runs at most once per `_TURN_LOCK_SWEEP_SECONDS`; removes entries
        that are both unlocked and idle past `_TURN_LOCK_IDLE_SECONDS`. A contended lock is `locked()` and a
        recently-used one has a fresh `last_used_monotonic`, so neither is ever swept — which is why no
        holder/waiter accounting is needed.
        """
        now = time.monotonic()
        if now - self._last_lock_sweep_monotonic < _TURN_LOCK_SWEEP_SECONDS:
            return
        self._last_lock_sweep_monotonic = now
        cold = [
            conversation_uuid
            for conversation_uuid, entry in self._turn_locks.items()
            if not entry.lock.locked() and now - entry.last_used_monotonic > _TURN_LOCK_IDLE_SECONDS
        ]
        for conversation_uuid in cold:
            self._turn_locks.pop(conversation_uuid, None)

    @abstractmethod
    async def handle_event(self, *, event: dict) -> None: ...

    async def on_start(self):
        await super().on_start()
        info = await self.slack_client.auth_test(bot_token=self._bot_token)
        if not info.get("ok"):
            raise RuntimeError(f"auth.test failed: {info.get('error')}")
        self.team_id = info["team_id"]
        self.bot_user_id = info["user_id"]  # "user_id" is the bot user's Slack ID
        self.bot_id = info["bot_id"]  # auth.test returns this directly for bot tokens
        self.team_name = info["team"]
        # app_id is not on auth.test — one users.info call surfaces it via profile.api_app_id. It's used only as
        # a fallback identity check; the primary path is bot_id matching.
        self.app_id = await self.slack_client.resolve_app_id(bot_token=self._bot_token, bot_user_id=self.bot_user_id)
        logger.info(
            "Slack harness bound to team_id=%s team_name=%r bot_id=%s app_id=%s",
            self.team_id,
            self.team_name,
            self.bot_id,
            self.app_id,
        )

        self.socket = build_socket_mode_client(self._app_token)
        self.socket.socket_mode_request_listeners.append(self._listener)
        await self.socket.connect()

    async def on_stop(self):
        if self.socket is not None:
            await self.socket.disconnect()  # stop new events first
            await self.socket.close()
        # super().on_stop() drains in-flight background handle_event tasks (up to 30s). Those tasks still call
        # Slack — keep the shared HTTP client open until the drain finishes, then close it.
        await super().on_stop()
        await self.slack_client.close()
