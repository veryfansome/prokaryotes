"""`SlackTierBHarness` — the `SlackHarness` under test in the Tier B suite, plus per-turn delivery helpers.

The subclass keeps every production code path — `_dispatch_event`, `_should_handle`, `_conversation_turn_lock`,
`sync_slack_thread`, `_run_turn`, `_finalize_slack_turn`, compaction — and only swaps the two real external
boundaries that Tier B cannot reach:

- the LLM provider, replaced by `FakeAnthropicClient` (scripted per turn);
- the Slack Web API, replaced per-turn by a `FakeSlackThreadClient` the test stages.

`handle_event` is overridden so the per-turn `slack` object is the test-supplied `FakeSlackThreadClient` rather
than a `SlackClientWithToken` over a real `httpx` client. Everything inside the lock — `sync_slack_thread`,
`build_prelude`, `_run_turn`, `_finalize_slack_turn` — is the real harness code.
"""

from __future__ import annotations

import asyncio
import uuid

from prokaryotes.harness_v1.slack import SLACK_TURN_TIMEOUT_SECONDS, SlackHarness, slack_conversation_uuid
from prokaryotes.slack_v1 import SlackBase
from tests.unit_tests._llm_fakes import FakeAnthropicClient, LLMRound, LLMScript
from tests.unit_tests._slack_fakes import FakeSlackClient, FakeSlackThreadClient

# Re-exported so `test_slack_flow.py` can import `LLMRound` / `LLMScript` from this module.
__all__ = ["APP_ID", "BOT_ID", "BOT_USER_ID", "LLMRound", "LLMScript", "SlackTierBHarness", "TEAM_ID"]

# Workspace identity the `FakeSlackClient.auth_test` reports — Tier B binds the harness to this synthetic team.
TEAM_ID = "T_TEAM"
BOT_USER_ID = "U_BOT"
BOT_ID = "B_BOT"
APP_ID = "A_APP"


def _ensure_event_in_thread(event: dict, thread_client: FakeSlackThreadClient) -> None:
    """Idempotently append a message-shaped inbound event into `thread_client.thread`.

    Real Slack delivers an event to the bot *after* writing the message to the thread; the per-turn
    `conversations.replies` fetch then returns it. The Tier B fakes don't model that automatically, so tests
    used to have to remember to stage the trigger into the fake thread by hand — a frequent omission that made
    `_run_turn`'s tombstoned-trigger guard fire mid-turn. This helper closes the gap. No-op for non-message
    events (`app_uninstalled`, `tokens_revoked`) and idempotent when the test has already staged the message.
    """
    if event.get("type") not in {"app_mention", "message"}:
        return
    ts = event.get("ts")
    if not ts or any(m.get("ts") == ts for m in thread_client.thread):
        return
    staged = {"ts": ts, "user": event.get("user"), "text": event.get("text", "")}
    if event.get("thread_ts"):
        staged["thread_ts"] = event["thread_ts"]
    thread_client.thread.append(staged)
    thread_client.thread.sort(key=lambda m: m["ts"])


class SlackTierBHarness(SlackHarness):
    """`SlackHarness` wired to fake Slack / LLM boundaries but real Redis + Elasticsearch."""

    def __init__(self) -> None:
        # Skip SlackHarness.__init__ — it builds a real provider LLM client. Run SlackBase.__init__ for the
        # Socket Mode / lock / identity state, then install the fakes.
        SlackBase.__init__(self, app_token="xapp-tier-b", bot_token="xoxb-tier-b")
        self.impl = "anthropic"
        self.llm_client = FakeAnthropicClient()
        from prokaryotes.utils_v1.llm_utils import ANTHROPIC_DEFAULT_MODEL

        self.default_model = ANTHROPIC_DEFAULT_MODEL
        self.slack_client = FakeSlackClient(app_id=APP_ID)
        # The test stages a `FakeSlackThreadClient` here before each delivery; `handle_event` reads it.
        self._thread_client: FakeSlackThreadClient | None = None

    async def deliver(self, event: dict, *, thread_client: FakeSlackThreadClient) -> None:
        """Run one inbound event end-to-end against `thread_client`, awaiting the turn to completion.

        Mirrors the `_dispatch_event` → `background_and_forget(handle_event(...))` path but awaits the turn so a
        test can assert on its committed state synchronously. The triggering message is auto-staged into
        `thread_client.thread` (idempotently) so the per-turn `conversations.replies` fetch sees the event Slack
        would have already delivered.
        """
        self._thread_client = thread_client
        _ensure_event_in_thread(event, thread_client)
        await self.handle_event(event=event)

    async def deliver_via_socket(self, event: dict, *, thread_client: FakeSlackThreadClient) -> None:
        """Run one inbound event through the real `_dispatch_event` gate (ack / dedup / `_should_handle`), then
        drain the background `handle_event` task. Used where the dispatch path itself is under test. The
        triggering message is auto-staged into `thread_client.thread` (idempotently). Default `event_id` is a
        fresh UUID so tests do not silently collide on the per-event dedupe key in Redis."""
        from tests.unit_tests._slack_fakes import envelope

        self._thread_client = thread_client
        _ensure_event_in_thread(event, thread_client)
        event_id = event.get("event_id", str(uuid.uuid4()))
        await self._dispatch_event(envelope=envelope(event, event_id=event_id))
        await self.drain_background_tasks()

    async def handle_event(self, *, event: dict) -> None:
        """Per-thread-locked turn driver — the production body with the per-turn `slack` object replaced by the
        test-staged `FakeSlackThreadClient`."""
        thread_ts = event.get("thread_ts") or event["ts"]
        channel_id = event["channel"]
        conversation_uuid = slack_conversation_uuid(self.team_id, channel_id, thread_ts)
        channel_type = event.get("channel_type", "channel")
        slack = self._thread_client
        async with self._conversation_turn_lock(conversation_uuid):
            await asyncio.wait_for(
                self._locked_turn(
                    channel_id=channel_id,
                    channel_type=channel_type,
                    conversation_uuid=conversation_uuid,
                    event=event,
                    slack=slack,
                    thread_ts=thread_ts,
                ),
                timeout=SLACK_TURN_TIMEOUT_SECONDS,
            )
