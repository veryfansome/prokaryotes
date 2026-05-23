"""Tier A — live Slack smoke for the Slack harness.

Each test posts into `SLACK_TEST_CHANNEL` as the user behind `SLACK_USER_TOKEN`, then waits for the harness —
running against a real Slack Socket Mode connection and a real LLM provider — to reply in the expected shape.
Skipped unless all four Slack env vars and `ANTHROPIC_API_KEY` are present (see `tier_a/conftest.py`).

Touches live Slack and a live LLM provider, so each scenario is slow (10–30 seconds) and rate-limited by both
APIs. Run by hand against a sandbox workspace, not in CI.
"""

from __future__ import annotations

import uuid

import pytest

from tests.integration_tests.tier_a.conftest import slack_api_call, wait_for_bot_reply

# Both markers: `integration` lets the README's "not integration" unit filter exclude this file alongside the
# rest of the integration tier; `live` is the existing per-tier label for live-LLM-and-network tests.
pytestmark = [pytest.mark.integration, pytest.mark.live]


@pytest.mark.asyncio(loop_scope="session")
async def test_channel_mention_round_trip(
    live_slack_harness,
    slack_live_env,
    slack_messages_to_cleanup,
    slack_test_user_id,
):
    """A user @-mention reaches the harness over Socket Mode and the bot replies in the thread, prefixed with
    `<@user>` and not still showing the placeholder."""
    bot_token = slack_live_env["bot_token"]
    bot_user_id = live_slack_harness.bot_user_id
    channel = slack_live_env["channel"]
    user_token = slack_live_env["user_token"]
    nonce = uuid.uuid4().hex[:8]

    posted = await slack_api_call(
        "chat.postMessage",
        token=user_token,
        data={
            "channel": channel,
            "text": f"<@{bot_user_id}> tier-a smoke {nonce} — reply with the single word 'pong'",
        },
    )
    assert posted.get("ok"), f"user post failed: {posted}"
    parent_ts = posted["ts"]
    slack_messages_to_cleanup.append((channel, parent_ts, user_token))

    reply = await wait_for_bot_reply(
        bot_token=bot_token,
        bot_user_id=bot_user_id,
        channel=channel,
        parent_ts=parent_ts,
    )

    text = reply["text"]
    assert text.startswith(f"<@{slack_test_user_id}>"), f"reply missing user mention prefix: {text!r}"
    assert "_…working_" not in text, f"reply still shows placeholder: {text!r}"
    body_after_prefix = text[len(f"<@{slack_test_user_id}>") :].strip()
    assert body_after_prefix, f"reply has prefix only, no body: {text!r}"


@pytest.mark.asyncio(loop_scope="session")
async def test_same_thread_continuation(
    live_slack_harness,
    slack_live_env,
    slack_messages_to_cleanup,
    slack_test_user_id,
):
    """Two @-mentions in the same thread both get replies — exercising the per-thread lock and the Redis
    fast-path continuation. The second reply is a distinct message from the first."""
    bot_token = slack_live_env["bot_token"]
    bot_user_id = live_slack_harness.bot_user_id
    channel = slack_live_env["channel"]
    user_token = slack_live_env["user_token"]
    nonce = uuid.uuid4().hex[:8]

    first = await slack_api_call(
        "chat.postMessage",
        token=user_token,
        data={
            "channel": channel,
            "text": f"<@{bot_user_id}> tier-a thread {nonce} — say the single word 'first'",
        },
    )
    assert first.get("ok"), f"first user post failed: {first}"
    root_ts = first["ts"]
    slack_messages_to_cleanup.append((channel, root_ts, user_token))

    first_reply = await wait_for_bot_reply(
        bot_token=bot_token,
        bot_user_id=bot_user_id,
        channel=channel,
        parent_ts=root_ts,
    )

    second = await slack_api_call(
        "chat.postMessage",
        token=user_token,
        data={
            "channel": channel,
            "thread_ts": root_ts,
            "text": f"<@{bot_user_id}> and now say the single word 'second'",
        },
    )
    assert second.get("ok"), f"second user post failed: {second}"
    slack_messages_to_cleanup.append((channel, second["ts"], user_token))

    second_reply = await wait_for_bot_reply(
        after_ts=first_reply["ts"],
        bot_token=bot_token,
        bot_user_id=bot_user_id,
        channel=channel,
        parent_ts=root_ts,
    )

    assert second_reply["ts"] != first_reply["ts"], "second reply is the same message as the first"
    assert second_reply["text"].startswith(f"<@{slack_test_user_id}>"), (
        f"second reply missing user mention prefix: {second_reply['text']!r}"
    )
