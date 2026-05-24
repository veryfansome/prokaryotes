"""Tests for the single-human-thread continuation feature.

Covers all three Redis write sites (A in `_locked_turn` post-sync, B in `_should_continue_threaded`, C in
`_on_dispatch_accept`), the dispatch gate's predicate behavior under the humans / engaged state machine, the
dispatch lock, and the first-turn / cross-site / cross-envelope races.
"""

from __future__ import annotations

import asyncio

import pytest

from prokaryotes.conversation_v1.models import Conversation, ConversationMessage
from prokaryotes.harness_v1.slack import SlackHarness, slack_conversation_uuid
from prokaryotes.slack_v1 import SlackBase
from tests.unit_tests._slack_fakes import FakeRedis, FakeSearchClient, envelope

BOT_USER = "U_BOT"
BOT_ID = "B_BOT"
APP_ID = "A_APP"
TEAM_ID = "T_TEAM"
CHANNEL = "C_CHAN"
THREAD_TS = "100.000000"
CONV_UUID = slack_conversation_uuid(TEAM_ID, CHANNEL, THREAD_TS)
HUMANS_KEY = f"slack_thread_humans:{CONV_UUID}"
ENGAGED_KEY = f"slack_thread_engaged:{CONV_UUID}"


class _TestHarness(SlackHarness):
    """`SlackHarness` with fakes injected, bypassing `__init__`'s LLM-client construction."""

    def __init__(self) -> None:
        # Skip SlackHarness/SlackBase __init__ — they construct an LLM client and the SlackClient. We just need
        # the lock dicts, the fakes, and the identity fields the new code paths read.
        self._redis_client = FakeRedis()
        self._search_client = FakeSearchClient()
        self._conversation_cache_ex = 60 * 60 * 24 * 7
        self.bot_user_id = BOT_USER
        self.bot_id = BOT_ID
        self.app_id = APP_ID
        self.team_id = TEAM_ID
        self._turn_locks: dict = {}
        self._dispatch_locks: dict = {}
        self._last_lock_sweep_monotonic = 0.0
        self.background_tasks: set[asyncio.Task] = set()
        self.handled: list[dict] = []

    async def handle_event(self, *, event: dict) -> None:
        # Default stub — tests can override per-instance to inject blocking / inspection behavior.
        self.handled.append(event)


@pytest.fixture
def harness() -> _TestHarness:
    return _TestHarness()


def _mention(user: str = "U_ALICE", ts: str = THREAD_TS, **extra) -> dict:
    e = {"type": "app_mention", "channel": CHANNEL, "ts": ts, "user": user, "text": f"<@{BOT_USER}> hi"}
    e.update(extra)
    return e


def _channel_reply(user: str, ts: str, thread_ts: str = THREAD_TS, **extra) -> dict:
    e = {
        "type": "message",
        "channel": CHANNEL,
        "ts": ts,
        "thread_ts": thread_ts,
        "user": user,
        "text": "follow-up",
        "channel_type": "channel",
    }
    e.update(extra)
    return e


async def _smembers_str(harness: _TestHarness, key: str) -> set[str]:
    return {m.decode() if isinstance(m, bytes) else m for m in await harness._redis_client.smembers(key)}


# -----------------------------------------------------------------------------
# Site C — eager humans seed in `_on_dispatch_accept`
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_site_c_writes_humans_for_channel_app_mention(harness: _TestHarness):
    """Site C SADDs the originator into humans and refreshes TTL via EXPIRE. engaged stays absent — only
    Site A sets it."""
    await harness._on_dispatch_accept(_mention())

    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}
    assert await harness._redis_client.exists(ENGAGED_KEY) == 0
    assert (HUMANS_KEY, 60 * 60 * 24 * 90) in harness._redis_client.expire_calls


@pytest.mark.asyncio
async def test_site_c_does_not_write_for_non_app_mention(harness: _TestHarness):
    """Site C's only trigger is `app_mention`. A dispatched `message` event (the single-human continuation
    path) does not re-write the cache — humans already exists, Site A will refresh."""
    await harness._on_dispatch_accept(_channel_reply("U_ALICE", "101.000000"))

    assert harness._redis_client.sadd_calls == []
    assert harness._redis_client.expire_calls == []


@pytest.mark.asyncio
async def test_site_c_no_channel_type_filter_mpim_dead_state_tolerated(harness: _TestHarness):
    """Site C does NOT filter by `channel_type` because `app_mention` does not carry it. An mpim app_mention
    therefore *does* write the humans cache. That entry is dead state — Site B's eligibility check vetoes
    incoming non-mention messages with `channel_type == "mpim"` before reading the cache."""
    # Site C writes despite the mpim channel_type.
    await harness._on_dispatch_accept(_mention(channel_type="mpim"))
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}

    # The incoming non-mention message — which DOES carry channel_type — drops at Site B's eligibility
    # check before reading the cache.
    dispatched = await harness._should_continue_threaded(
        _channel_reply("U_ALICE", "101.000000", channel_type="mpim")
    )
    assert dispatched is False
    # Site B did not touch the cache on this drop.
    assert all(call[0] != HUMANS_KEY or "101.000000" not in str(call) for call in harness._redis_client.sadd_calls[1:])


@pytest.mark.asyncio
async def test_site_c_skips_event_without_channel_or_thread(harness: _TestHarness):
    """Defensive: an app_mention missing `channel` or `ts` (shouldn't happen at this point, but) is a no-op."""
    await harness._on_dispatch_accept({"type": "app_mention", "user": "U_ALICE"})

    assert harness._redis_client.sadd_calls == []


# -----------------------------------------------------------------------------
# Site A — `_write_thread_state_post_sync`
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_site_a_seeds_humans_and_engaged_for_single_human_thread(harness: _TestHarness):
    """After Site A on a single-human conversation, humans contains the human author and engaged is set."""
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="hi")],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")

    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}
    assert await harness._redis_client.exists(ENGAGED_KEY) == 1


@pytest.mark.asyncio
async def test_site_a_seeds_all_historical_humans(harness: _TestHarness):
    """Pre-existing multi-human thread: Site A SADDs every human in the reconciled conversation, including
    participants from before the bot was mentioned."""
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="90.000000", author_id="U_BOB", content="earlier"),
            ConversationMessage(source_id="95.000000", author_id="U_CAROL", content="earlier 2"),
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="<@U_BOT> hi"),
        ],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")

    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB", "U_CAROL"}


@pytest.mark.asyncio
async def test_site_a_excludes_bot_and_foreign_bot_authors(harness: _TestHarness):
    """Bot author IDs and `bot:` foreign-bot IDs do not enter the humans set."""
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="<@U_BOT> hi"),
            ConversationMessage(
                source_id="101.000000", author_id=BOT_USER, content="bot reply", reply_to_source_id=THREAD_TS
            ),
            ConversationMessage(source_id="102.000000", author_id="bot:B_FOREIGN", content="foreign bot post"),
            ConversationMessage(source_id="103.000000", author_id="unknown", content="anonymous"),
        ],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")

    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}


@pytest.mark.asyncio
async def test_site_a_stickiness_across_deletion(harness: _TestHarness):
    """Pre-existing humans `{U_ALICE, U_BOB}`; next reconciliation sees only Alice (Bob's message tombstoned).
    Site A's SADD is union-only — Bob survives in the cache."""
    # Seed the cache with both humans.
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE", "U_BOB")

    # Conversation now reflects only Alice (Bob's message is tombstoned).
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="90.000000", author_id="U_BOB", content="gone", deleted=True),
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="<@U_BOT> hi"),
        ],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")

    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}


@pytest.mark.asyncio
async def test_site_a_noops_for_dm(harness: _TestHarness):
    """Site A no-ops for DMs: the continuation gate only reads the cache for channel/group threads, so
    writing for `channel_type == "im"` would leak 90-day Redis entries that are never consulted. (`mpim`
    mentions still write here because `app_mention` events don't carry `channel_type` and `handle_event`
    falls back to `"channel"` — the resulting cache is harmless dead state since Site B vetoes mpim reads;
    see `_write_thread_state_post_sync`'s docstring.)"""
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="hi")],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="im")

    assert await harness._redis_client.exists(HUMANS_KEY) == 0
    assert await harness._redis_client.exists(ENGAGED_KEY) == 0


@pytest.mark.asyncio
async def test_site_a_stickiness_across_compaction(harness: _TestHarness):
    """Pre-existing humans `{U_ALICE, U_BOB}`; the next reconciliation produces a `Conversation` whose
    `messages` shows only Alice (Bob's posts in `ancestor_summaries` only). Site A still SADDs only Alice; the
    cache retains Bob because SADD is union-only."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE", "U_BOB")

    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        ancestor_summaries=["...earlier Bob/Alice exchange..."],
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="continuation")],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")

    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}


# -----------------------------------------------------------------------------
# Site B — `_should_continue_threaded` predicate
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_site_b_drops_when_humans_absent(harness: _TestHarness):
    """No humans key → drop. Cache stays absent — Site B refuses to seed on the drop path."""
    dispatched = await harness._should_continue_threaded(_channel_reply("U_ALICE", "101.000000"))

    assert dispatched is False
    assert await harness._redis_client.exists(HUMANS_KEY) == 0


@pytest.mark.asyncio
async def test_site_b_dispatches_single_human_continuation(harness: _TestHarness):
    """humans `{U_ALICE}`, engaged set, sender `U_ALICE` → dispatched. SADD is a no-op for the already-present
    member."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)

    dispatched = await harness._should_continue_threaded(_channel_reply("U_ALICE", "101.000000"))

    assert dispatched is True
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}


@pytest.mark.asyncio
async def test_site_b_drops_pending_engagement(harness: _TestHarness):
    """humans `{U_ALICE}`, engaged absent (sync window) → dropped even when sender == only human. Sender is
    still recorded (no-op SADD here, but the EXPIRE landed)."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")

    dispatched = await harness._should_continue_threaded(_channel_reply("U_ALICE", "101.000000"))

    assert dispatched is False
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}


@pytest.mark.asyncio
async def test_site_b_drops_sticky_multi_human(harness: _TestHarness):
    """humans `{U_ALICE, U_BOB}`, engaged set, sender `U_ALICE` → dropped (multi-human sticky). SADD is a
    no-op; SMEMBERS post-condition is unchanged."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE", "U_BOB")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)

    dispatched = await harness._should_continue_threaded(_channel_reply("U_ALICE", "101.000000"))

    assert dispatched is False
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}


@pytest.mark.asyncio
async def test_site_b_records_second_human_on_drop_engaged(harness: _TestHarness):
    """humans `{U_ALICE}`, engaged set, sender `U_BOB` → dropped AND humans becomes `{U_ALICE, U_BOB}`. The
    write-through is what keeps the sticky transition correct."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)

    dispatched = await harness._should_continue_threaded(_channel_reply("U_BOB", "101.000000"))

    assert dispatched is False
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}


@pytest.mark.asyncio
async def test_site_b_records_second_human_during_pending_engagement(harness: _TestHarness):
    """humans `{U_ALICE}`, engaged absent, sender `U_BOB` → dropped AND humans becomes `{U_ALICE, U_BOB}`.
    Recording the dropper into humans during the sync window is load-bearing: without it, Bob's signal would
    be lost and Alice's later follow-up could incorrectly dispatch."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")

    dispatched = await harness._should_continue_threaded(_channel_reply("U_BOB", "101.000000"))

    assert dispatched is False
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}


@pytest.mark.asyncio
async def test_site_b_idempotent_sadd_and_ttl_refresh(harness: _TestHarness):
    """humans `{U_ALICE, U_BOB}`, engaged set, sender already in humans → dropped (multi-human), set
    unchanged, but EXPIRE recorded on both keys (TTL refresh on every relevant hit)."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE", "U_BOB")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)
    # Reset call recorders so we measure just this round.
    harness._redis_client.expire_calls.clear()

    dispatched = await harness._should_continue_threaded(_channel_reply("U_BOB", "101.000000"))

    assert dispatched is False
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}
    refreshed_keys = {key for key, _ in harness._redis_client.expire_calls}
    assert HUMANS_KEY in refreshed_keys
    assert ENGAGED_KEY in refreshed_keys


@pytest.mark.asyncio
async def test_site_b_mpim_vetoed_without_touching_cache(harness: _TestHarness):
    """mpim non-mention thread reply drops at the eligibility check before any cache lookup. No SADD, no
    EXPIRE, no SMEMBERS."""
    # Pre-seed humans/engaged so a buggy implementation that read the cache before the mpim check would
    # appear to "succeed" if the cache happened to match — this guards against that regression.
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)
    harness._redis_client.expire_calls.clear()
    harness._redis_client.sadd_calls.clear()

    dispatched = await harness._should_continue_threaded(
        _channel_reply("U_ALICE", "101.000000", channel_type="mpim")
    )

    assert dispatched is False
    assert harness._redis_client.sadd_calls == []
    assert harness._redis_client.expire_calls == []


@pytest.mark.asyncio
async def test_site_b_rejects_top_level_message(harness: _TestHarness):
    """A `message` event with `thread_ts == ts` (top-level post, not a reply) drops at the eligibility check."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)

    event = _channel_reply("U_ALICE", THREAD_TS, thread_ts=THREAD_TS)
    dispatched = await harness._should_continue_threaded(event)

    assert dispatched is False


@pytest.mark.asyncio
async def test_site_b_rejects_subtyped_message(harness: _TestHarness):
    """A `message` event with any subtype (channel_join, etc.) drops at the eligibility check."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)

    event = _channel_reply("U_ALICE", "101.000000")
    event["subtype"] = "channel_join"
    assert await harness._should_continue_threaded(event) is False


@pytest.mark.asyncio
async def test_site_b_rejects_hidden_event(harness: _TestHarness):
    """A `hidden=True` event (e.g. metadata-only `message_replied`) drops before the eligibility checks even
    if its other fields would otherwise pass — parity with the base `_should_handle` guard."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)

    event = _channel_reply("U_ALICE", "101.000000")
    event["hidden"] = True
    assert await harness._should_continue_threaded(event) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["channel", "ts", "user", "text"])
async def test_site_b_rejects_missing_required_field(harness: _TestHarness, missing: str):
    """A reply missing `channel`, `ts`, `user`, or `text` drops before the cache lookup so `handle_event` /
    `_locked_turn` never have to read a key that isn't there. Without this guard a sender whose Redis
    predicate matches could still pass and KeyError downstream."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)

    event = _channel_reply("U_ALICE", "101.000000")
    del event[missing]
    assert await harness._should_continue_threaded(event) is False


# -----------------------------------------------------------------------------
# Site B — bot-identity guard
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        {"user": BOT_USER},
        {"user": "U_ALICE", "bot_id": BOT_ID},
        {"user": "U_ALICE", "bot_profile": {"app_id": APP_ID}},
        {"user": "U_ALICE", "subtype": "bot_message"},
    ],
)
async def test_site_b_bot_guard_drops_without_mutating_cache(harness: _TestHarness, shape: dict):
    """A bot-shaped event must drop without polluting the human cache with the bot's ID."""
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)
    harness._redis_client.sadd_calls.clear()
    harness._redis_client.expire_calls.clear()

    event = _channel_reply("placeholder", "101.000000")
    event.update(shape)

    dispatched = await harness._should_continue_threaded(event)

    assert dispatched is False
    assert harness._redis_client.sadd_calls == []
    assert harness._redis_client.expire_calls == []
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}


# -----------------------------------------------------------------------------
# End-to-end behavior — multi-human transition
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_multi_human_transition(harness: _TestHarness):
    """Alice mentions, Site A fires post-sync → humans {Alice}, engaged. Bob non-mentions → dropped, humans
    becomes {Alice, Bob}. Alice non-mentions → dropped (sticky)."""
    # Alice's @-mention is dispatched: Site C writes humans {Alice} eagerly.
    await harness._on_dispatch_accept(_mention("U_ALICE"))
    # Then `_locked_turn` runs sync (modeled as a Conversation with just Alice in single-human) and Site A
    # fires.
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="<@U_BOT> hi")],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}
    assert await harness._redis_client.exists(ENGAGED_KEY) == 1

    # Bob's non-mention follow-up → dropped (sender not in humans), humans becomes {Alice, Bob}.
    bob_dispatched = await harness._should_continue_threaded(_channel_reply("U_BOB", "101.000000"))
    assert bob_dispatched is False
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}

    # Alice's next non-mention → dropped (sticky multi-human).
    alice_dispatched = await harness._should_continue_threaded(_channel_reply("U_ALICE", "102.000000"))
    assert alice_dispatched is False


# -----------------------------------------------------------------------------
# End-to-end behavior — pre-existing multi-human thread
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_existing_multi_human_thread_blocks_first_continuation(harness: _TestHarness):
    """The thread had prior Bob and Carol posts before Alice mentioned the bot. Site A's SADD seeds humans
    with all three; Alice's first non-mention follow-up drops correctly."""
    await harness._on_dispatch_accept(_mention("U_ALICE"))

    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="50.000000", author_id="U_BOB", content="prior bob"),
            ConversationMessage(source_id="60.000000", author_id="U_CAROL", content="prior carol"),
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="<@U_BOT> hi"),
        ],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB", "U_CAROL"}

    dispatched = await harness._should_continue_threaded(_channel_reply("U_ALICE", "101.000000"))
    assert dispatched is False


# -----------------------------------------------------------------------------
# Engagement-pending window
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engagement_pending_window_drops_originator(harness: _TestHarness):
    """During the sync window (Site C wrote humans, Site A has NOT run), even the originator's own non-mention
    drops. After Site A fires, the same sender dispatches."""
    await harness._on_dispatch_accept(_mention("U_ALICE"))
    # Engagement is pending — Alice's follow-up should drop.
    assert await harness._should_continue_threaded(_channel_reply("U_ALICE", "101.000000")) is False

    # Now Site A fires. Subsequent non-mention from Alice dispatches.
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="<@U_BOT> hi")],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")
    assert await harness._should_continue_threaded(_channel_reply("U_ALICE", "102.000000")) is True


@pytest.mark.asyncio
async def test_engagement_pending_window_poisons_for_second_human(harness: _TestHarness):
    """During the sync window, Bob's non-mention drops but Bob is recorded into humans. When Site A later
    fires with a single-human conversation, Bob's poison survives (SADD is union-only) and Alice's subsequent
    non-mention drops."""
    await harness._on_dispatch_accept(_mention("U_ALICE"))
    # Bob's race during the pending window.
    assert await harness._should_continue_threaded(_channel_reply("U_BOB", "101.000000")) is False
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}

    # Site A then runs with a single-human conversation (Bob's post hadn't been reconciled, or whatever).
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="<@U_BOT> hi")],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")
    # Bob's poison survived.
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}

    # Alice's later non-mention drops (multi-human sticky).
    assert await harness._should_continue_threaded(_channel_reply("U_ALICE", "102.000000")) is False


# -----------------------------------------------------------------------------
# First-turn race — second human chimes in during the in-flight turn
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_turn_race_second_human_recorded_before_handle_event_runs():
    """Site C must write the humans seed *inside* `_dispatch_event`, before `background_and_forget`. An
    implementation that placed the eager write inside `handle_event` would let Bob's `_dispatch_event` run
    before the humans key existed and lose the second-human signal."""
    harness = _TestHarness()

    # Block `handle_event` indefinitely so the background task does not race the assertion.
    block = asyncio.Event()
    handled_events: list[dict] = []

    async def blocking_handle(*, event: dict) -> None:
        handled_events.append(event)
        await block.wait()

    harness.handle_event = blocking_handle  # type: ignore[method-assign]

    # Alice's @-mention through the full dispatch path.
    await harness._dispatch_event(envelope=envelope(_mention("U_ALICE")))
    # After _dispatch_event returns, Site C has written humans (regardless of whether handle_event has started).
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE"}
    assert await harness._redis_client.exists(ENGAGED_KEY) == 0  # Site A hasn't run inside handle_event.

    # Bob's non-mention through dispatch. _should_continue_threaded fires; humans key exists, Bob is added,
    # engaged is absent → drop.
    await harness._dispatch_event(
        envelope=envelope(_channel_reply("U_BOB", "101.000000"), event_id="Ev_BOB")
    )
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}

    # Now Site A fires (we simulate the post-sync step directly).
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="<@U_BOT> hi")],
    )
    await harness._write_thread_state_post_sync(conv, channel_type="channel")
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}

    # Yield so the dispatched Alice @-mention's background task records its event before we measure baseline.
    await asyncio.sleep(0)
    handled_before = len(handled_events)
    # Alice's later non-mention drops (sticky), proving the race was closed.
    await harness._dispatch_event(
        envelope=envelope(_channel_reply("U_ALICE", "102.000000"), event_id="Ev_ALICE2")
    )
    await asyncio.sleep(0)  # let any newly-scheduled bg task run too
    # Alice's non-mention dropped before reaching handle_event — handled_events did not grow.
    assert len(handled_events) == handled_before
    assert handled_events == [_mention("U_ALICE")]

    # Release the blocked handler so the test exits cleanly.
    block.set()
    if harness.background_tasks:
        await asyncio.gather(*harness.background_tasks, return_exceptions=True)


# -----------------------------------------------------------------------------
# Cross-site race — Site A vs Site B SADD on the same key
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_site_race_sadd_union_atomic(harness: _TestHarness):
    """Schedule Site A (single-human reconciled view) and Site B (drop for U_BOB) concurrently; force them to
    interleave at their SADDs via `asyncio.sleep(0)`. The final SMEMBERS must be {U_ALICE, U_BOB} regardless
    of which SADD lands first — SADD is union-only."""
    # Pre-seed humans `{U_ALICE}` + engaged so Site B sees a populated cache and would fire its predicate
    # branch under the dangerous interleaving.
    await harness._redis_client.sadd(HUMANS_KEY, "U_ALICE")
    await harness._redis_client.set(ENGAGED_KEY, "1", ex=90 * 24 * 60 * 60)

    # Site A is going to SADD {U_ALICE} (a single-human reconciled view) — the dangerous case because under
    # any read-modify-write design this could clobber Bob.
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="solo")],
    )

    async def site_a():
        await asyncio.sleep(0)  # yield so site B can interleave
        await harness._write_thread_state_post_sync(conv, channel_type="channel")

    async def site_b():
        await asyncio.sleep(0)  # yield so site A can interleave
        await harness._should_continue_threaded(_channel_reply("U_BOB", "101.000000"))

    await asyncio.gather(site_a(), site_b())

    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}


# -----------------------------------------------------------------------------
# Cross-envelope race — dispatch lock serializes Site C against Site B
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_envelope_race_serialized_by_dispatch_lock():
    """Two `_dispatch_event` calls for the same `conversation_uuid` interleave under the dispatch lock —
    Site C's SADD completes before Site B starts its EXISTS check, so Bob is recorded into humans. Without
    the lock Bob's EXISTS would race ahead and his signal would be lost."""
    harness = _TestHarness()

    block = asyncio.Event()
    handled_events: list[dict] = []

    async def blocking_handle(*, event: dict) -> None:
        handled_events.append(event)
        await block.wait()

    harness.handle_event = blocking_handle  # type: ignore[method-assign]

    # Inject a delay into Alice's Site C SADD so we can observe whether Bob blocks correctly.
    alice_sadd_started = asyncio.Event()
    alice_sadd_release = asyncio.Event()

    async def sadd_hook(key: str, members: tuple) -> None:
        # Only delay the first SADD on the humans key (Alice's Site C). Subsequent SADDs (Bob's Site B,
        # Site A's later writes if any) should pass through. `members` is the original tuple before
        # FakeRedis encodes it to bytes — match against the str form the call site uses.
        if key == HUMANS_KEY and members == ("U_ALICE",) and not alice_sadd_started.is_set():
            alice_sadd_started.set()
            await alice_sadd_release.wait()

    harness._redis_client.sadd_hook = sadd_hook

    async def alice_dispatch():
        await harness._dispatch_event(envelope=envelope(_mention("U_ALICE"), event_id="Ev_ALICE"))

    async def bob_dispatch():
        # Wait until Alice's Site C SADD is in flight, then attempt Bob's dispatch.
        await alice_sadd_started.wait()
        await harness._dispatch_event(
            envelope=envelope(_channel_reply("U_BOB", "101.000000"), event_id="Ev_BOB")
        )

    alice_task = asyncio.create_task(alice_dispatch())
    bob_task = asyncio.create_task(bob_dispatch())

    # Give Bob's task a chance to reach the dispatch lock. It should be blocked because Alice holds it.
    await alice_sadd_started.wait()
    # At this point Alice's SADD is mid-flight inside the dispatch lock; Bob is queued.
    # If the lock were missing, Bob's _should_continue_threaded could observe humans absent and drop Bob
    # without recording. With the lock, Bob waits until Alice's Site C completes.

    # Release Alice's SADD so the dispatch lock unwinds.
    alice_sadd_release.set()
    await asyncio.gather(alice_task, bob_task)

    # After both dispatches: Alice is in humans (Site C), Bob is in humans (Site B's write-through).
    assert await _smembers_str(harness, HUMANS_KEY) == {"U_ALICE", "U_BOB"}

    block.set()
    if harness.background_tasks:
        await asyncio.gather(*harness.background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_dispatch_lock_isolation_different_conversations():
    """Two `_dispatch_event` calls for *different* `conversation_uuid`s do not block each other — the per-key
    lock is the load-bearing isolation. Without it, every workspace would serialize through one lock and
    dispatch latency would skyrocket."""
    harness = _TestHarness()

    block = asyncio.Event()

    async def blocking_handle(*, event: dict) -> None:
        await block.wait()

    harness.handle_event = blocking_handle  # type: ignore[method-assign]

    # Stage a delay on Alice's @-mention into channel 1. Bob's @-mention into channel 2 should proceed without
    # waiting for Alice's lock.
    alice_sadd_started = asyncio.Event()
    alice_sadd_release = asyncio.Event()
    bob_other_uuid = slack_conversation_uuid(TEAM_ID, "C_OTHER", "200.000000")
    bob_other_humans = f"slack_thread_humans:{bob_other_uuid}"

    async def sadd_hook(key: str, members: tuple) -> None:
        if key == HUMANS_KEY and not alice_sadd_started.is_set():
            alice_sadd_started.set()
            await alice_sadd_release.wait()

    harness._redis_client.sadd_hook = sadd_hook

    alice_task = asyncio.create_task(
        harness._dispatch_event(envelope=envelope(_mention("U_ALICE"), event_id="Ev_ALICE"))
    )

    await alice_sadd_started.wait()
    # Now run Bob's dispatch for a *different* channel/thread. It must complete without waiting on
    # alice_sadd_release.
    bob_event = _mention("U_BOB", ts="200.000000")
    bob_event["channel"] = "C_OTHER"
    await asyncio.wait_for(
        harness._dispatch_event(envelope=envelope(bob_event, event_id="Ev_BOB")),
        timeout=1.0,
    )
    # Bob's Site C wrote to its own key, separate from Alice's pending one.
    assert await _smembers_str(harness, bob_other_humans) == {"U_BOB"}

    alice_sadd_release.set()
    await alice_task
    block.set()
    if harness.background_tasks:
        await asyncio.gather(*harness.background_tasks, return_exceptions=True)


# -----------------------------------------------------------------------------
# `_dispatch_conversation_uuid` defaults / overrides
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_lock_noop_for_unkeyable_event(harness: _TestHarness):
    """An event the harness can't key to a thread (missing channel/ts) makes `_dispatch_conversation_uuid`
    return `None`; `_dispatch_lock` then becomes a no-op context manager — dispatch still works."""
    # Just verify the predicate returns None and the lock context manager completes without taking a real
    # lock entry.
    assert harness._dispatch_conversation_uuid({"type": "app_mention"}) is None
    async with harness._dispatch_lock(None):
        pass


# -----------------------------------------------------------------------------
# Base class defaults — non-SlackHarness subclasses keep working
# -----------------------------------------------------------------------------


class _BaseOnly(SlackBase):
    """Minimal `SlackBase` subclass — used to verify the new hooks default to safe no-ops."""

    def __init__(self) -> None:
        super().__init__(app_token="xapp-test", bot_token="xoxb-test")
        self.team_id = TEAM_ID
        self.bot_user_id = BOT_USER
        self.bot_id = BOT_ID
        self.app_id = APP_ID
        self.handled: list[dict] = []

    async def handle_event(self, *, event: dict) -> None:
        self.handled.append(event)


@pytest.mark.asyncio
async def test_base_class_defaults_keep_non_slackharness_subclasses_working():
    """A minimal SlackBase subclass that overrides only `handle_event` still dispatches and drops the same
    set of events it always did — the new hooks default to safe no-ops."""
    base = _BaseOnly()
    base._redis_client = FakeRedis()

    # An accept-path event dispatches as before — `_on_dispatch_accept` is a no-op on the base.
    await base._dispatch_event(envelope=envelope(_mention("U_ALICE")))
    await base.drain_background_tasks()
    assert base.handled == [_mention("U_ALICE")]

    # A non-mention thread reply still drops — `_should_continue_threaded` defaults to False.
    base.handled.clear()
    await base._dispatch_event(envelope=envelope(_channel_reply("U_ALICE", "101.000000"), event_id="Ev_REPLY"))
    await base.drain_background_tasks()
    assert base.handled == []
