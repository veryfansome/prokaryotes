"""`SlackHarness` — Socket Mode worker harness for one Slack workspace.

`SlackHarness(SlackBase)` adds the LLM / tool wiring on top of `SlackBase`'s Socket Mode lifecycle and dispatch.
Each inbound `@`-mention or DM message runs one LLM turn: `handle_event` derives the deterministic
`conversation_uuid`, takes the per-thread turn lock, and runs `_locked_turn` under a shared
`SLACK_TURN_TIMEOUT_SECONDS` budget. `_locked_turn` reconciles the Slack thread into the active `Conversation`
(`sync_slack_thread`), builds the channel prelude, and drives `_run_turn` — projection, `stream_turn`, NDJSON →
Slack via `SlackStreamer`, and finalize.

Slack does not use `HarnessBase.stream_and_finalize`: bot `source_id`s are real Slack `ts`s known only after
posting, a long reply is several posts, and the failure path needs the same finalize call to make the failure
notice durable. `_finalize_slack_turn` commits one bot `ConversationMessage` per `PostedMessage`, each stamped
with `reply_to_source_id = event["ts"]`.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import uuid

from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    NormalizedMessage,
    TurnExecution,
    TurnItem,
)
from prokaryotes.conversation_v1.project import project_for_llm
from prokaryotes.conversation_v1.reconcile import reconcile
from prokaryotes.harness_v1 import build_llm_client
from prokaryotes.slack_v1 import SlackBase
from prokaryotes.slack_v1.client import SlackClientWithToken
from prokaryotes.slack_v1.replay import (
    _earliest_raw_window_ts,
    _human_user_ids_in,
    _mentioned_user_ids_in,
    _slack_author_id,
    _strip_in_flight_orphans,
    build_prelude,
    resolve_display_names,
    sanitize_mentions,
)
from prokaryotes.slack_v1.streaming import PostedMessage, SlackStreamer
from prokaryotes.tools_v1.think import ThinkTool
from prokaryotes.utils_v1 import system_message_utils
from prokaryotes.utils_v1.llm_utils import (
    ANTHROPIC_DEFAULT_MODEL,
    COMPACTION_TOKEN_THRESHOLD_PCT,
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    OPENAI_DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)

# Arbitrary, fixed namespace for deriving a `conversation_uuid` from a Slack thread anchor.
SLACK_THREAD_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000005ac4")

# Per-turn timeout wrapping the *entire* locked region (sync + prelude + run). A stalled `conversations.replies`
# fetch, paginated thread, Redis/ES call, rate-limiter backoff, tool call, or LLM call cannot then hold the
# per-thread lock and wedge every later mention in that thread indefinitely.
SLACK_TURN_TIMEOUT_SECONDS = 10 * 60


def slack_conversation_uuid(team_id: str, channel_id: str, thread_ts: str) -> str:
    """Derive the deterministic `conversation_uuid` for a Slack thread anchor.

    Namespaced by `team_id` so threads in different workspaces cannot collide; stable across restarts and Redis
    eviction, so every inbound event can recompute it locally with no stored mapping.
    """
    return str(uuid.uuid5(SLACK_THREAD_NAMESPACE, f"{team_id}:{channel_id}:{thread_ts}"))


class SlackHarness(SlackBase):
    """Provider-agnostic Slack Socket Mode harness.

    `impl` selects the LLM client (`"anthropic"` → `AnthropicClient`, `"openai"` → `OpenAIClient`); the
    instruction role is the client's concern. `llm_client` and `default_model` are read by `HarnessBase`'s
    compaction helpers (`_build_compact_fn`, `_summarize_and_compact`).
    """

    def __init__(self, *, impl: str, app_token: str, bot_token: str):
        super().__init__(app_token=app_token, bot_token=bot_token)
        self.impl = impl
        self.llm_client, _ = build_llm_client(impl)  # instruction role is the client's concern
        self.default_model = ANTHROPIC_DEFAULT_MODEL if impl == "anthropic" else OPENAI_DEFAULT_MODEL

    async def handle_event(self, *, event: dict) -> None:
        thread_ts = event.get("thread_ts") or event["ts"]
        channel_id = event["channel"]
        conversation_uuid = slack_conversation_uuid(self.team_id, channel_id, thread_ts)
        slack = SlackClientWithToken(self.slack_client, self._bot_token)

        # app_mention events do not carry channel_type; fall back to "channel".
        channel_type = event.get("channel_type", "channel")

        # Serialize turns for one thread — see "Serializing same-thread turns". The wait_for wraps the *entire*
        # locked region (sync_slack_thread + build_prelude + _run_turn) so no phase can hold the lock — and wedge
        # every later mention in this thread — indefinitely.
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

    async def on_start(self):
        self.llm_client.init_client()
        await super().on_start()

    async def on_stop(self):
        await super().on_stop()
        await self.llm_client.close()

    async def sync_slack_thread(
        self,
        *,
        channel_id: str,
        conversation_uuid: str,
        slack_client: SlackClientWithToken,
        thread_ts: str,
        triggering_ts: str,
    ) -> Conversation:
        """Fetch the current Slack thread and reconcile it into the active `Conversation` snapshot.

        Reuses `reconcile` / `_apply_result` / `_cache_and_persist_conversation` but skips the web-only steps —
        Slack supplies `source_id`s and `author_id`s directly. The bounded `oldest` fetch excludes the compacted
        prefix, so `_split_compacted_prefix` is intentionally not invoked.
        """
        stored = await self._load_slack_stored(conversation_uuid)

        # Bounded fetch: when stored has history, start at the active raw window's first source_id — the
        # compacted prefix is already in storage and ancestor_summaries. Cold-recovery on a brand-new thread (or
        # a fresh root after Case B) passes oldest=None and full-fetches. inclusive=True is required: Slack
        # excludes messages exactly at `oldest` otherwise, which would drop stored.messages[0] from incoming and
        # reconcile would false-`delete` it.
        oldest = _earliest_raw_window_ts(stored)

        # Effective cutoff: at least triggering_ts, but extend to include anything already in stored.messages
        # that's later than triggering_ts. This handles the same-thread serialization case so reconcile sees a
        # clean append rather than a false-delete of the prior turn's committed reply.
        latest_stored = max(
            (m.source_id for m in stored.messages if not m.deleted),
            default=None,
        )
        effective_cutoff = max(triggering_ts, latest_stored) if latest_stored is not None else triggering_ts

        thread = await slack_client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            oldest=oldest,
            inclusive=oldest is not None,
            include_all_metadata=True,  # surface prokaryotes_in_flight markers
            paginate_until_ts=effective_cutoff,  # client-side pagination cap
        )

        # Orphan-placeholder pre-pass: a previous turn that crashed between post_placeholder and
        # _finalize_slack_turn left a bot-authored post in Slack carrying in-flight metadata but never made it
        # into storage. The metadata + "source_id not in stored" check identifies exactly those messages.
        thread = await _strip_in_flight_orphans(
            thread,
            channel_id=channel_id,
            slack_client=slack_client,
            stored=stored,
        )

        # Resolve display names for both authors AND every <@USER> mention in human-authored text, so
        # sanitize_mentions can rewrite "<@U07ABC>" to the actual user's display name. The cache is keyed by
        # (team_id, user_id) because standard `U…` IDs aren't globally unique — see `resolve_display_names`.
        referenced_user_ids = _human_user_ids_in(thread) | _mentioned_user_ids_in(thread, self.bot_user_id)
        display_names = await resolve_display_names(
            slack_client,
            self.redis_client,
            referenced_user_ids,
            team_id=self.team_id,
        )

        stored_source_ids = {m.source_id for m in stored.messages}
        incoming: list[NormalizedMessage] = []
        for m in thread:
            # Inner filter: include messages that existed at trigger time OR are already in stored. The pagination
            # cap uses `effective_cutoff` so stored messages with ts > triggering_ts are returned; the inner
            # filter uses the *unextended* triggering_ts so unrelated chatter racing in is not this turn's
            # concern, with stored messages exempted so reconcile can match them.
            if m["ts"] > triggering_ts and m["ts"] not in stored_source_ids:
                continue  # don't break — a later stored message could still follow
            author_id = _slack_author_id(m, self.bot_user_id, self.bot_id, self.app_id)
            is_bot = author_id == self.bot_user_id
            text = m.get("text") or ""
            incoming.append(
                NormalizedMessage(
                    source_id=m["ts"],
                    author_id=author_id,
                    # Bot posts were stored verbatim as-posted by _finalize_slack_turn, so they must replay
                    # verbatim — otherwise reconcile mis-classifies an unchanged prior reply as `edit`.
                    # sanitize_mentions only ever applies to humans.
                    content=text if is_bot else sanitize_mentions(text, self.bot_user_id, display_names),
                    display_name=display_names.get(author_id) if not author_id.startswith("bot:") else None,
                )
            )

        # No call to `_split_compacted_prefix` here — the bounded `oldest` fetch already excludes the compacted
        # prefix by construction, which is exactly what reconcile expects against `stored.messages`.
        result = reconcile(stored, incoming)
        final = await self._apply_result(
            stored=stored,
            result=result,
            normalized=incoming,
            bot_author_id=self.bot_user_id,
            conversation_uuid=conversation_uuid,
        )
        await self._cache_and_persist_conversation(final)
        return final

    def _build_instruction_parts(
        self,
        *,
        channel_id: str,
        conversation: Conversation,
        originator_display_name: str,
        tool_callbacks: dict,
    ) -> list[str]:
        """Assemble the trusted instruction string — five sections, no background blocks.

        `ancestor_summary_block` and `channel_prelude` are not in the instruction message; they enter the LLM
        input as leading user-role `ProjectedItem`s via the projection seam. Positions 1–5 match
        `WebHarness._build_instruction_parts`; the role swap is `# Slack context` for `# User context`.
        """
        parts: list[str] = []
        parts.extend(system_message_utils.get_core_instruction_parts(summaries=bool(conversation.ancestor_summaries)))
        parts.append("")
        parts.extend(system_message_utils.get_runtime_context_parts())
        parts.append("")
        parts.append("# Tool usage")
        for name in sorted(tool_callbacks.keys()):
            parts.append("")
            parts.extend(tool_callbacks[name].system_message_parts)
        parts.append("")
        parts.extend(system_message_utils.get_personality_parts())
        parts.append("")
        parts.append("# Slack context")
        parts.append("")
        parts.append(f"- You are replying to {originator_display_name} in Slack")
        parts.append(f"- The Slack channel ID is {channel_id}")
        if self.team_name:
            parts.append(f"- The Slack workspace is {self.team_name}")
        parts.append(
            "- Do NOT start your reply with a `<@user>` mention — the harness prepends that mechanically;"
            " emitting one yourself would double-mention the user."
        )
        return parts

    async def _finalize_slack_turn(
        self,
        *,
        completed: bool,
        conversation: Conversation,
        posted: list[PostedMessage],
        triggering_source_id: str,
        turn_items: list[TurnItem],
    ) -> None:
        """Commit a turn's posted bot messages and (when tools ran) its `TurnExecution`, then cache and persist.

        Commits one bot `ConversationMessage` per `PostedMessage` — `content` stored verbatim (the as-posted
        text, so the next `conversations.replies` reconciles cleanly), `reply_to_source_id` set to the triggering
        user message's `source_id` so the projection's two-pass walk keeps the turn pair intact. Each is inserted
        at its `source_id`-sorted position to preserve the Slack-side storage invariant. A non-empty `turn_items`
        persists one `TurnExecution` keyed to `posted[0].source_id` with `completed` from the caller (`True`
        happy path, `False` failure-notice path).
        """
        for post in posted:
            message = ConversationMessage(
                source_id=post.source_id,
                author_id=conversation.bot_author_id,
                content=post.content,
                reply_to_source_id=triggering_source_id,
            )
            _insert_message_sorted(conversation.messages, message)

        if turn_items:
            await self.search_client.put_turn_execution(
                TurnExecution(
                    conversation_uuid=conversation.conversation_uuid,
                    bot_message_source_id=posted[0].source_id,
                    items=turn_items,
                    completed=completed,
                )
            )
        await self._cache_and_persist_conversation(conversation)
        for post in posted:
            await self.refresh_assistant_index_with(
                conversation.conversation_uuid,
                post.source_id,
                post.content,
            )

    async def _load_slack_stored(self, conversation_uuid: str) -> Conversation:
        """Cold-recovery load of the active `Conversation` snapshot for a Slack thread.

        Redis fast path on `conversation:{conversation_uuid}`; on miss, `find_latest_active_snapshot_uuid` seeds
        an exact-load / ancestor-chain rebuild; on a total miss, a fresh `Conversation` keyed to the harness's
        `bot_user_id`.
        """
        cached = await self.redis_client.get(f"conversation:{conversation_uuid}")
        if cached:
            try:
                return Conversation.model_validate_json(cached)
            except Exception:
                logger.warning("Corrupt cached conversation %s — falling back to ES", conversation_uuid)

        snapshot_uuid = await self.search_client.find_latest_active_snapshot_uuid(conversation_uuid)
        if snapshot_uuid is None:
            return Conversation(conversation_uuid=conversation_uuid, bot_author_id=self.bot_user_id, messages=[])

        return await self._load_stored(
            conversation_uuid=conversation_uuid,
            snapshot_uuid=snapshot_uuid,
            bot_author_id=self.bot_user_id,
            partial=[],
        )

    async def _locked_turn(
        self,
        *,
        channel_id: str,
        channel_type: str,
        conversation_uuid: str,
        event: dict,
        slack: SlackClientWithToken,
        thread_ts: str,
    ) -> None:
        """Run one turn under the per-thread lock: reconcile the thread, build the prelude, drive `_run_turn`."""
        conversation = await self.sync_slack_thread(
            channel_id=channel_id,
            conversation_uuid=conversation_uuid,
            slack_client=slack,
            thread_ts=thread_ts,
            triggering_ts=event["ts"],
        )
        prelude = await build_prelude(
            channel_id=channel_id,
            channel_type=channel_type,
            conversation_uuid=conversation_uuid,
            redis_client=self.redis_client,
            slack_client=slack,
            thread_ts=thread_ts,
            triggering_ts=event["ts"],
        )
        await self._run_turn(
            channel_id=channel_id,
            conversation=conversation,
            event=event,
            prelude=prelude,
            slack_client=slack,
            thread_ts=thread_ts,
        )

    async def _run_turn(
        self,
        *,
        channel_id: str,
        conversation: Conversation,
        event: dict,
        prelude: str | None,
        slack_client: SlackClientWithToken,
        thread_ts: str,
    ) -> None:
        """Project the reconciled conversation, drive `stream_turn`, route NDJSON to Slack, and finalize."""
        # TurnExecutions for the raw window's bot messages — projection interleaves their tool items.
        bot_source_ids = [
            m.source_id for m in conversation.messages if not m.deleted and m.author_id == conversation.bot_author_id
        ]
        historical_turns = await self.search_client.get_turn_executions(
            conversation.conversation_uuid,
            bot_source_ids,
        )

        think_tool = ThinkTool(self.llm_client, self.default_model)
        tool_callbacks = {think_tool.name: think_tool}
        # FileTool / ShellCommandTool intentionally omitted; see Tool Selection.

        # Tombstoned-trigger guard: if the triggering mention was deleted between the inbound event and turn
        # start (a rare message_deleted race), the projection has no live trigger to place last — the two-pass
        # walk would end on a prior assistant message — so answering would reply to stale thread context. Drop
        # the turn. No placeholder has been posted yet, so there is no Slack-side artifact to clean up.
        trigger_msg = conversation.message_by_source_id(event["ts"])
        if trigger_msg is None or trigger_msg.deleted:
            logger.error(
                "Dropping turn for conversation %s: triggering message %s is missing or tombstoned",
                conversation.conversation_uuid,
                event["ts"],
            )
            return

        # Instruction string is strictly trusted content — five sections, no background blocks. The originator's
        # display name was already resolved by sync_slack_thread and stored on the triggering ConversationMessage.
        instruction = "\n".join(
            self._build_instruction_parts(
                channel_id=channel_id,
                conversation=conversation,
                originator_display_name=trigger_msg.display_name or "Slack user",
                tool_callbacks=tool_callbacks,
            )
        )
        # The channel prelude (when present) goes through `leading_context_blocks` so the projection's final
        # `_merge_consecutive_same_role` pass coalesces it with the summary block and the first stored user
        # message. triggering_source_id is informational/defensive — the two-pass walk orders the projection.
        leading_context_blocks: list[str] = [prelude] if prelude else []
        projected_items = project_for_llm(
            conversation,
            historical_turns=historical_turns,
            leading_context_blocks=leading_context_blocks,
            triggering_source_id=event["ts"],
        )

        pending_compaction = [False]

        def on_usage(input_tokens: int, output_tokens: int) -> None:
            context_window = MODEL_CONTEXT_WINDOWS.get(self.default_model, DEFAULT_CONTEXT_WINDOW)
            ctx_pct = int(input_tokens / context_window * 100)
            if ctx_pct >= COMPACTION_TOKEN_THRESHOLD_PCT:
                pending_compaction[0] = True

        # Per-turn ID stamps every Slack post made during this turn via Slack message metadata. It is the
        # load-bearing piece of the orphan-placeholder recovery contract.
        turn_id = str(uuid.uuid4())

        streamer = SlackStreamer(
            channel_id=channel_id,
            conversation_uuid=conversation.conversation_uuid,
            slack_client=slack_client,
            thread_ts=thread_ts,
            turn_id=turn_id,
            # Prefix the reply with <@user> in channels and mpim so the trigger user is notified; DMs pass None.
            reply_to_user_id=event["user"] if event.get("channel_type") != "im" else None,
        )
        # The streamer phase — post_placeholder, the consume loop, and finish — all run inside one try so any
        # failure goes through streamer.fail(). Putting post_placeholder() or finish() outside the try would let
        # a Slack API exception or timeout-cancellation leave in-flight metadata on Slack posts that were never
        # reconciled into storage.
        committed_turn_items: list[TurnItem] = []
        posted: list[PostedMessage] = []
        try:
            await streamer.post_placeholder()
            async for line in self.llm_client.stream_turn(
                items=projected_items,
                instruction=instruction,
                model=self.default_model,
                on_committed_turn_item=committed_turn_items.append,
                on_usage=on_usage,
                stream_ndjson=True,
                tool_callbacks=tool_callbacks,
            ):
                await streamer.consume(line)
            posted = await streamer.finish()
        except (Exception, asyncio.CancelledError):
            # Exception = LLM/tool/Slack API failure; CancelledError = the SLACK_TURN_TIMEOUT_SECONDS wait_for in
            # handle_event fired. CancelledError is a BaseException, so a plain `except Exception` would skip it.
            notice: PostedMessage | None = await streamer.fail()
            # Persist the failure notice as the turn's stored bot message so the next replay classifies it as
            # `match` rather than chat.delete-ing it as an orphan. committed_turn_items holds whatever tool calls
            # completed before the failure; passing it through with completed=False keeps an aborted-mid-tools
            # turn's partial TurnExecution.
            if notice is not None:
                try:
                    # Hold the per-thread lock until finalize completes — see _await_finalize_through_cancel.
                    await self._await_finalize_through_cancel(
                        self._finalize_slack_turn(
                            completed=False,
                            conversation=conversation,
                            posted=[notice],
                            triggering_source_id=event["ts"],
                            turn_items=committed_turn_items,
                        )
                    )
                    await streamer.clear_in_flight_metadata([notice])
                except Exception:
                    # Storage failed while persisting the failure notice. Log and swallow — the original
                    # streamer-phase exception is the more diagnostically useful signal and should surface
                    # unchanged. See the "Failure-notice persist failed" row of the failure-window matrix.
                    logger.exception("failed to persist failure notice for turn_id=%s", turn_id)
            raise

        if posted:
            # Hold the per-thread lock until finalize completes — see the failure-path comment above. Metadata
            # clear runs outside the protection: it's best-effort by design.
            await self._await_finalize_through_cancel(
                self._finalize_slack_turn(
                    completed=True,
                    conversation=conversation,
                    posted=posted,
                    triggering_source_id=event["ts"],
                    turn_items=committed_turn_items,
                )
            )
            # Best-effort metadata clear runs *after* the durable persist. A crash in the window between persist
            # and clear leaves the metadata in place; the next replay's orphan pre-pass sees `source_id in
            # stored` and preserves the message.
            await streamer.clear_in_flight_metadata(posted)
            await self._maybe_compact(conversation, pending_compaction[0])


def _insert_message_sorted(messages: list[ConversationMessage], message: ConversationMessage) -> None:
    """Insert `message` into `messages` at its `source_id`-sorted position, preserving the Slack-side
    `source_id`-sorted invariant on `conversation.messages`.

    `_finalize_slack_turn`'s commit is in practice always a tail-append (a turn's own posts carry the latest
    `ts`), so the sorted insert is defensive here — but keeping one uniform rule across the syncer's append path
    and the finalize commit avoids a divergent invariant.
    """
    keys = [m.source_id for m in messages]
    index = bisect.bisect_right(keys, message.source_id)
    messages.insert(index, message)
