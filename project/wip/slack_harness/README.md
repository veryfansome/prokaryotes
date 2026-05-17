# Slack Harness Design

## Goals

A new harness that lets a Slack workspace member talk to the prokaryotes agent by `@`-mentioning the bot, then continue the exchange in the same Slack thread.

Prokaryotes is a **locally running multi-user service**, not a hosted SaaS. The Slack integration is structured to match: no public HTTPS surface, no OAuth, no app distribution.

This design depends on the [HarnessBase extraction](../harness_base/README.md) — pulling the Redis/ES/compaction lifecycle out of `WebBase` into a transport-agnostic `HarnessBase` that the Slack worker can inherit without picking up FastAPI.

**One harness instance per Slack workspace.** Each `SlackHarness` is bound at startup to exactly one workspace's tokens, opens exactly one Socket Mode connection, and serves only that workspace. Connecting more workspaces means adding more `slack-*` services to `docker-compose.yml`.

User-visible behavior:

- The operator (or an individual user of the local service) creates a Slack app in **their own workspace** from a shared manifest, installs it to that workspace via the Slack UI, and pastes the resulting app-level + bot tokens into a new `slack-<name>` docker-compose service entry. No public install link, no admin consent flow on the Slack side beyond the operator's own workspace.
- **Channels and group DMs (mpim)**: the bot replies only when explicitly `@`-mentioned. It never reacts to free-form chatter, even inside a thread it has previously replied in. A follow-up question requires another `@prokaryote`.
- A `@`-mention at channel top level produces a reply **as a thread on the user's message**. A `@`-mention inside an existing thread produces a reply in that same thread.
- The bot's reply prefixes `<@user>` (the user who triggered the mention) in channels and mpim so the trigger user is notified.
- **1:1 DMs**: the bot replies to every top-level DM message as a thread on it, and to every thread reply within that DM thread. No `@`-mention is required — a DM is unambiguous by definition.
- All previous thread messages (including chatter that did not mention the bot) are still visible to the model as context on the next turn — only the trigger rule is `@`-restricted.
- When the `@`-mention is at channel top-level, the bot can additionally include a short tail of channel messages preceding the mention as context for its first turn.
- A long-running thread continues to work indefinitely — compaction handles context-window pressure transparently, exactly as it does for the web harness.

What this is **not** about:

- Slash commands, modals, or message shortcuts.
- Granting the model file or shell access in a Slack context (see Tool Selection).

---

## Observed Repository Context

Observed in the current repository:

- `prokaryotes/harness_v1/web.py` is the only `WebBase` subclass. It builds a per-turn instruction message, registers `FileTool` / `ShellCommandTool` / `ThinkTool`, and streams NDJSON back over an HTTP `StreamingResponse`. Compaction is registered via `pending_compaction` and `compact_fn` closures handed into `stream_and_finalize()`.
- `prokaryotes/web_v1/partition_sync.py` reconciles via Redis fast path → exact ES load → ancestor-chain rebuild. `_partition_can_follow_client` accepts `partition_uuid=None` on the client, so a caller that does not echo a `partition_uuid` still gets the cached partition on the Redis fast path.
- `prokaryotes/web_v1/compaction.py` defines `_compact_partition()` (the background CAS swap) and `get_compaction_status` (the polling endpoint used by the browser UI). Compaction does not require any client-side handshake when no UI is rendering a `compaction_pending` indicator.
- `prokaryotes/search_v1/context_partitions.py` exposes `find_partition_by_tail_hash`, `get_partition`, `put_partition`, `update_partition`, and `search_partitions`. There is **no** lookup for "the latest active partition for `conversation_uuid=X`" today.
- `scripts/web.py` is a four-line entry point that instantiates `WebHarness` and calls `harness.init()`. There is no `scripts/slack.py` yet.

These observations shape two architectural decisions below: a deterministic `conversation_uuid` derivation that namespaces conversations by `team_id`, and a new ES query for cold-Redis recovery on long-idle threads.

---

## Slack ↔ Prokaryotes Mapping

| Slack concept | Prokaryotes concept |
|---|---|
| Slack workspace (`team_id`) | The whole `SlackHarness` instance — one harness, one workspace. `team_id`, `bot_user_id`, and `team_name` are resolved once at startup via `auth.test` and held as instance state; envelope `team_id` is sanity-checked against it. |
| Slack thread, identified by `(team_id, channel_id, thread_ts)` | `conversation_uuid` (derived deterministically; see below). |
| Top-level DM message (no `thread_ts`) | A fresh conversation — bot replies into a thread anchored on the user's message; `thread_ts` for routing becomes `event["ts"]` |
| Each thread message | `ChatMessage` (`assistant` if posted by the bot user, otherwise `user`) |
| Bot-posted messages within a thread | Collapsed into a single `assistant` turn when replayed (a long response split across N posts re-merges into one `ChatMessage`) |
| Channel messages preceding a top-level `@`-mention | First-turn system-message **prelude**, not persisted as `ChatMessage`s |
| In-thread messages preceding an `@`-mention that adopts an existing thread | Replayed as ordinary `ChatMessage`s by `conversations.replies` — no separate prelude (it would double-count) |
| The Slack user who first `@`-mentioned the bot | The conversation's "originator"; their display name fills the `# Slack context` section of the instruction message |
| Other thread participants in multi-human threads | Still mapped to `user` role; message bodies are prefixed `<display_name> ` (space, no colon) so the model can attribute |
| Slack edits to past messages | Surface on next replay as a divergence → `sync_from_conversation` truncates and re-appends, identical to the web client's retry path |
| Slack `event_id` | Idempotency key for inbound Socket Mode delivery (Slack redelivers on missed ack) |

### Deterministic `conversation_uuid`

`conversation_uuid` is derived once from the thread anchor and never stored separately:

```python
SLACK_THREAD_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000005ac4")  # arbitrary, fixed

def slack_conversation_uuid(team_id: str, channel_id: str, thread_ts: str) -> str:
    return str(uuid.uuid5(SLACK_THREAD_NAMESPACE, f"{team_id}:{channel_id}:{thread_ts}"))
```

Consequences:

- No new Postgres table is needed for thread → conversation mapping. Every inbound event can recompute the UUID locally.
- The mapping is stable across restarts and across Redis eviction.
- Two threads with different anchors cannot collide.

### Recovering `partition_uuid` after Redis eviction

Redis caches each conversation for `CONVERSATION_CACHE_EXPIRY_SECONDS` (default 7 days) and renews TTL on every write. A thread that goes idle longer than that loses its cached partition. To reattach, the harness needs the latest *active* (`is_compacted=false`) partition's UUID for the conversation.

Add one new method on `ContextPartitionSearcher`:

```python
async def find_latest_active_partition_uuid(self, conversation_uuid: str) -> str | None:
    """Return the partition_uuid of the most recently modified non-compacted partition.

    Used by harnesses that do not have a client echoing partition_uuid back.
    """
```

ES query: `term: conversation_uuid` AND `term: is_compacted=false` AND `compaction_state in {committed, missing}`, sort `dt_modified desc`, size 1. Cost: one ES round-trip per cold-thread reattach. The result is then attached to `ChatConversation.partition_uuid` before calling `sync_context_partition`, which seeds the chain walk correctly.

---

## Architecture

### Layout

Following the repo's `*_v1/` pattern:

```
prokaryotes/
  harness_v1/
    slack.py             # new: SlackHarness(SlackBase) — Socket Mode worker
  slack_v1/              # new module
    __init__.py          # SlackBase(HarnessBase) — Socket Mode lifecycle, dispatch
    client.py            # async wrapper around chat.postMessage / chat.update / conversations.replies
    replay.py            # build_chat_conversation, prelude, name resolution
    streaming.py         # consume stream_and_finalize NDJSON and post to Slack
scripts/
  slack.py               # entry point — asyncio.run(run()); one workspace per process
```

`HarnessBase` is introduced by the [HarnessBase extraction](../harness_base/README.md) wip. `SlackBase` inherits from it directly.

### `SlackBase` and `SlackHarness`

`SlackBase` owns the Socket Mode client and the event dispatch. `SlackHarness` adds the LLM/tool wiring. Both inherit from `HarnessBase` — no FastAPI, no Postgres, no inbound HTTP.

```python
# prokaryotes/slack_v1/__init__.py (sketch)
class SlackBase(HarnessBase, ABC):
    """Worker harness bound to a single Slack workspace."""

    def __init__(self, app_token: str, bot_token: str):
        super().__init__()
        self._app_token = app_token
        self._bot_token = bot_token
        self.slack_client = SlackClient()  # bot-API wrapper, token bound per call
        self.socket: SocketModeClient | None = None

        # Resolved at startup via auth.test:
        self.team_id: str | None = None
        self.bot_user_id: str | None = None
        self.team_name: str | None = None

    async def on_start(self):
        await super().on_start()
        info = await self.slack_client.auth_test(bot_token=self._bot_token)
        if not info.get("ok"):
            raise RuntimeError(f"auth.test failed: {info.get('error')}")
        self.team_id = info["team_id"]
        self.bot_user_id = info["user_id"]   # "user_id" is the bot user's Slack ID
        self.team_name = info["team"]
        logger.info("Slack harness bound to team_id=%s team_name=%r", self.team_id, self.team_name)

        self.socket = build_socket_mode_client(self._app_token)
        self.socket.socket_mode_request_listeners.append(self._listener)
        await self.socket.connect()

    async def on_stop(self):
        if self.socket is not None:
            await self.socket.disconnect()
            await self.socket.close()
        await self.slack_client.close()
        await super().on_stop()

    async def _listener(self, client, request) -> None:
        # Ack first so Slack doesn't redeliver while we run the LLM turn.
        await client.send_socket_mode_response(SocketModeResponse(envelope_id=request.envelope_id))
        await self._dispatch_event(envelope=request.to_dict())

    async def _dispatch_event(self, *, envelope: dict) -> None:
        if envelope.get("type") != "events_api":
            return
        payload = envelope["payload"]
        event_id = payload.get("event_id")
        if event_id and not await self._claim_event_id(event_id):
            return  # duplicate delivery — Slack redelivers if ack timed out

        event = payload["event"]
        if event.get("type") in {"tokens_revoked", "app_uninstalled"}:
            logger.error("Slack app removed or token revoked for team_id=%s; shutting down harness", self.team_id)
            await self.on_stop()  # process supervisor restarts us; if the tokens are still bad, we'll crash again on auth.test
            return

        # Sanity check team_id; mismatch means the bot token was mis-configured against
        # an app from a different workspace.
        if payload.get("team_id") and payload["team_id"] != self.team_id:
            logger.warning(
                "Dropping event with team_id=%s — harness is bound to %s",
                payload["team_id"], self.team_id,
            )
            return

        if self._should_handle(event):
            self.background_and_forget(self.handle_event(event=event))

    async def _claim_event_id(self, event_id: str) -> bool:
        return bool(await self.redis_client.set(f"slack_event_seen:{event_id}", "1", ex=600, nx=True))

    def _should_handle(self, event: dict) -> bool:
        if event.get("type") == "app_mention":
            return True
        if event.get("type") != "message":
            return False
        if event.get("user") == self.bot_user_id:
            return False
        if event.get("subtype") is not None:
            return False
        if event.get("channel_type") == "im":
            return True       # every non-bot DM message — DMs are unambiguous
        return False          # all other non-mention channel/mpim/thread chatter — ignored

    @abstractmethod
    async def handle_event(self, *, event: dict) -> None: ...
```

Trigger rules:

- `app_mention` (channel, group, or mpim — top-level or threaded; Slack fires the event for both) → handle
- DM (`channel_type: "im"`) message from a human, no subtype → handle
- Everything else (channel/mpim chatter without an `@`, threaded replies without an `@`, message edits, bot-to-bot, joins/leaves) → drop

```python
# prokaryotes/harness_v1/slack.py (sketch)
class SlackHarness(SlackBase):
    def __init__(self, *, impl: str, app_token: str, bot_token: str):
        super().__init__(app_token=app_token, bot_token=bot_token)
        self.impl = impl
        self.llm_client, self.instruction_role = build_llm_client(impl)
        self.default_model = ANTHROPIC_DEFAULT_MODEL if impl == "anthropic" else OPENAI_DEFAULT_MODEL

    async def on_start(self):
        self.llm_client.init_client()
        await super().on_start()

    async def on_stop(self):
        await super().on_stop()
        await self.llm_client.close()

    async def handle_event(self, *, event: dict) -> None:
        thread_ts = event.get("thread_ts") or event["ts"]
        channel_id = event["channel"]
        conversation_uuid = slack_conversation_uuid(self.team_id, channel_id, thread_ts)
        slack = SlackClientWithToken(self.slack_client, self._bot_token)

        # app_mention events do not carry channel_type; fall back to "channel".
        channel_type = event.get("channel_type", "channel")

        chat_conversation, prelude = await build_chat_conversation(
            slack_client=slack,
            redis_client=self.redis_client,
            search_client=self.search_client,
            bot_user_id=self.bot_user_id,
            team_id=self.team_id,
            channel_type=channel_type,
            channel_id=channel_id,
            thread_ts=thread_ts,
            triggering_ts=event["ts"],
            conversation_uuid=conversation_uuid,
        )

        await self._run_turn(
            chat_conversation=chat_conversation,
            channel_id=channel_id,
            event=event,
            prelude=prelude,
            slack_client=slack,
            thread_ts=thread_ts,
        )
```

`_run_turn` mirrors `WebHarness.post_chat` closely but consumes the NDJSON stream itself and routes events to Slack. See [Reply Streaming](#reply-streaming-to-slack).

---

## Per-Workspace Setup

Each connected Slack workspace is a separately-created Slack app owned by that workspace, and runs in its own `SlackHarness` process. The workspace owner builds the Slack app from the manifest below, installs it to their workspace via the Slack UI, and pastes the resulting tokens into a new `slack-<name>` `docker-compose` service. The only ongoing cost is that manifest changes have to be re-applied by each workspace owner.

### Shared manifest

Every workspace's Slack app is created from this manifest. It declares Socket Mode (no `request_url`), no Redirect URLs (no OAuth), and the bot scopes / event subscriptions the harness needs:

```json
{
    "display_information": {
        "name": "Prokaryote",
        "description": "Prokaryotes agent. @-mention to start a thread, or DM directly.",
        "background_color": "#1e1e1e"
    },
    "features": {
        "bot_user": {
            "display_name": "prokaryote",
            "always_online": false
        }
    },
    "oauth_config": {
        "scopes": {
            "bot": [
                "app_mentions:read",
                "channels:history",
                "chat:write",
                "groups:history",
                "im:history",
                "im:read",
                "im:write",
                "mpim:history",
                "mpim:read",
                "users:read"
            ]
        }
    },
    "settings": {
        "event_subscriptions": {
            "bot_events": [
                "app_mention",
                "app_uninstalled",
                "message.channels",
                "message.groups",
                "message.im",
                "message.mpim",
                "tokens_revoked"
            ]
        },
        "interactivity": {
            "is_enabled": false
        },
        "org_deploy_enabled": false,
        "socket_mode_enabled": true,
        "is_hosted": false,
        "token_rotation_enabled": false
    }
}
```

The app-level token (`xapp-...`) with `connections:write` is generated separately from the Slack app's Basic Information page after creation — it's not part of the manifest schema.

### Onboarding flow for a new workspace

```
1. Workspace owner visits https://api.slack.com/apps?new_app=1 → "From manifest"
   → pastes the manifest from this doc → picks their workspace as the home workspace → Create

2. In the new app's config UI:
   a. Basic Information → App-Level Tokens → Generate Token and Scopes
        • name it "socket"
        • scope: connections:write
      Copy the resulting xapp-... token.
   b. Install App → Install to Workspace → Approve.
      Copy the resulting xoxb-... bot token from OAuth & Permissions.

3. Add a new docker-compose service for this workspace:

    slack-<short-name>:
      build: .
      command: python -m scripts.slack
      environment:
        SLACK_HARNESS_IMPL: anthropic
        SLACK_APP_TOKEN: xapp-...    # the xapp- token from step 2a
        SLACK_BOT_TOKEN: xoxb-...    # the xoxb- token from step 2b
        # ...same backend env as web (DB, Redis, ES, LLM keys)
      depends_on:
        - elasticsearch
        - elasticsearch-init
        - redis
      restart: unless-stopped

4. docker compose up -d slack-<short-name>

5. Invite the bot to channels (`/invite @prokaryote`) or DM it directly. Done.
```

Removing a workspace: delete the service from `docker-compose.yml` and run `docker compose down slack-<short-name>`. Tokens were only ever in env; nothing persists outside Slack itself.

### Slack-side state

The harness resolves three pieces of workspace metadata on startup by calling `auth.test`:

| Field | Source | Used for |
|---|---|---|
| `team_id` | `auth.test` response | Salt for `conversation_uuid` derivation; sanity check on envelope `team_id` |
| `bot_user_id` | `auth.test` response | Filtering the bot's own messages out of `_should_handle` and replay |
| `team_name` | `auth.test` response | Display only — `# Slack context` section in the instruction message |

If `auth.test` fails (network error, bad token), `on_start` raises and the process exits. The container restart policy brings it back; if the token is still bad it crashes again, and you read the logs. There is no graceful "wait for tokens to become valid" mode — bad tokens are a config error, not a runtime state.

### Token revocation and app-uninstall

Slack delivers `tokens_revoked` and `app_uninstalled` over the socket if the workspace admin removes the app or revokes the token. `_dispatch_event` logs at error level and calls `on_stop`. The supervisor restarts the process, `auth.test` fails on next start, and the container enters a crash loop the operator can see in logs.

### `SlackClient` and `SlackClientWithToken`

```python
class SlackClient:
    def __init__(self):
        self._http = httpx.AsyncClient(base_url="https://slack.com/api/", timeout=20.0)

    async def auth_test(self, *, bot_token: str) -> dict: ...
    async def chat_post_message(self, *, bot_token: str, channel: str, **kwargs) -> dict: ...
    async def chat_update(self, *, bot_token: str, channel: str, ts: str, **kwargs) -> dict: ...
    async def conversations_replies(self, *, bot_token: str, channel: str, ts: str) -> list[dict]: ...
    async def conversations_history(self, *, bot_token: str, channel: str, **kwargs) -> list[dict]: ...
    async def users_info(self, *, bot_token: str, user: str) -> dict: ...
    async def close(self) -> None: await self._http.aclose()


class SlackClientWithToken:
    """Curries the harness's bot_token onto every call so the per-turn code path
    (build_chat_conversation, SlackStreamer, etc.) does not have to thread it through."""

    def __init__(self, base: SlackClient, bot_token: str):
        self._base, self._token = base, bot_token

    async def chat_post_message(self, **kwargs): return await self._base.chat_post_message(bot_token=self._token, **kwargs)
    # …mirror the rest of SlackClient's public surface…
```

---

## Inbound Event Flow

```
SlackBase holds one persistent WebSocket bound to the harness's workspace at startup:

  Slack ──WS──► SocketModeClient(app_token=xapp-...)
                     │
                     ├─► ack envelope_id back over the socket (sub-3s, before any work)
                     ├─► dedupe event_id via Redis SET NX EX 600 (handles Slack redelivery)
                     ├─► tokens_revoked / app_uninstalled → on_stop (process exits, supervisor restarts)
                     ├─► envelope team_id != self.team_id → drop with a warning
                     ├─► quick filter (app_mention / DM message; never non-@ chatter)
                     └─► background_and_forget(handle_event(event))
                                │
                                ├─► SlackClientWithToken curries self._bot_token
                                ├─► derive conversation_uuid (self.team_id + channel_id + thread_ts)
                                ├─► fetch thread via conversations.replies
                                ├─► (optional) fetch channel-tail prelude — top-level channel @ only
                                ├─► build ChatConversation
                                ├─► sync_context_partition → reconcile / rebuild
                                ├─► build instruction parts (incl. cached channel-tail prelude when present)
                                ├─► register tools (ThinkTool by default)
                                ├─► post placeholder Slack message into the thread
                                ├─► consume stream_and_finalize NDJSON:
                                │     • buffer text_delta, chat.update placeholder ~1Hz
                                │     • map tool_call / progress_message to status updates
                                │     • on long output, split into multiple thread posts
                                │     • compaction_pending → no-op (background)
                                └─► final chat.update with rendered Markdown
```

### Why the work runs in the background

Slack expects a Socket Mode `ack` within ~3 seconds of envelope delivery, otherwise it considers the event undelivered and will redeliver. An LLM turn typically takes longer than that and may run for minutes if tools are involved. The pattern is: ack the envelope immediately inside the socket listener, then fire `HarnessBase.background_and_forget(handle_event(...))`.

This is the same `background_and_forget` already used by `stream_and_finalize` for the compaction handoff, so the lifecycle management (task tracking, exception logging, shutdown drain with a 30-second wait) is reused unchanged.

### What triggers a turn

`_should_handle` is the only trigger gate. If it returns `True`, `handle_event` runs an LLM turn unconditionally. There is no separate `_should_respond` step — the trigger rule is "did the user `@` me, or is this a DM?" and that question is fully answered by the inbound event envelope, without needing to inspect Redis or ES.

This means the partition state lookup is no longer in the hot path of "should I respond at all" — it only runs inside `build_chat_conversation` to recover `partition_uuid` for an existing conversation. `find_latest_active_partition_uuid` still earns its keep for cold-Redis recovery of long-idle threads where the user comes back and `@`-mentions the bot again.

---

## Building `ChatConversation` from a Slack thread

`build_chat_conversation` is the bridge from Slack state to `ChatConversation`:

```python
async def build_chat_conversation(
    *,
    slack_client: SlackClientWithToken,
    redis_client: Redis,
    search_client: SearchClient,
    bot_user_id: str,
    team_id: str,
    channel_type: str,           # "im", "mpim", "channel", or "group" — from the inbound event
    channel_id: str,
    thread_ts: str,
    triggering_ts: str,
    conversation_uuid: str,
) -> tuple[ChatConversation, str | None]:
    thread = await slack_client.conversations_replies(channel=channel_id, ts=thread_ts)
    messages = collapse_consecutive_bot_messages(thread, bot_user_id)

    # Resolve display names for prefixing (cached per call).
    display_names = await resolve_display_names(slack_client, redis_client, user_ids_in(messages))

    chat_messages: list[ChatMessage] = []
    triggering_index = None
    for index, m in enumerate(messages):
        if m["user"] == bot_user_id:
            chat_messages.append(ChatMessage(role="assistant", content=m["text"]))
        else:
            prefix = f"<{display_names[m['user']]}> " if multi_human_thread(messages) else ""
            chat_messages.append(ChatMessage(role="user", content=prefix + sanitize_mentions(m["text"], bot_user_id)))
        if m["ts"] == triggering_ts:
            triggering_index = index

    # Discard anything after the triggering message — late events can race, and we
    # only want to respond to messages up to and including the one that woke us.
    if triggering_index is not None:
        chat_messages = chat_messages[: triggering_index + 1]

    partition_uuid = await resolve_partition_uuid(conversation_uuid, redis_client, search_client)
    prelude = await build_prelude(
        slack_client=slack_client,
        redis_client=redis_client,
        channel_type=channel_type,
        channel_id=channel_id,
        conversation_uuid=conversation_uuid,
        thread_ts=thread_ts,
        triggering_ts=triggering_ts,
    )

    return ChatConversation(
        conversation_uuid=conversation_uuid,
        partition_uuid=partition_uuid,
        messages=chat_messages,
    ), prelude
```

### Helper details

**`collapse_consecutive_bot_messages`** merges N consecutive bot-authored posts into one `assistant` message. This is needed because a long LLM response is split across multiple thread posts (Slack's per-message limit), but the conversation model expects one assistant turn per LLM call. Joins with `"\n\n"` (or a sentinel split marker if we later need to undo the merge for re-posting).

**`sanitize_mentions`** strips the `<@BOT>` prefix from user messages so the model doesn't see its own mention echoed back as content. Foreign `<@USER>` mentions are rewritten to `<display_name>` for readability.

**`resolve_partition_uuid`**:

```python
async def resolve_partition_uuid(conversation_uuid, redis_client, search_client) -> str | None:
    # Redis fast path: if the cached partition exists, sync_context_partition can use it
    # without us passing a partition_uuid (partition_can_follow_client returns True for None).
    if await redis_client.exists(f"context_partition:{conversation_uuid}"):
        return None
    # Cold thread: fetch latest non-compacted partition_uuid from ES so chain
    # reconstruction has a head to start from.
    return await search_client.find_latest_active_partition_uuid(conversation_uuid)
```

**`multi_human_thread`** returns True iff the thread has at least two distinct non-bot participants. Single-human threads skip the `<name>` prefix to keep prompts clean.

### Pre-mention prelude (`build_prelude`)

The prelude is a system-message section providing channel context the user assumes the bot can see. It only applies to top-level channel `@`-mentions; all other shapes return `None`. Once computed it is cached and reused on every turn of the same conversation (see end-of-section note):

```python
async def build_prelude(
    *,
    slack_client: SlackClientWithToken,
    redis_client: Redis,
    channel_type: str,
    channel_id: str,
    conversation_uuid: str,
    thread_ts: str,
    triggering_ts: str,
) -> str | None:
    # 1:1 DMs have no surrounding context: there is no channel tail, and the full
    # thread content (if any) is already replayed into chat_messages.
    if channel_type == "im":
        return None

    # Cache: once computed, the prelude is stable for the life of the conversation.
    cache_key = f"slack_prelude:{conversation_uuid}"
    cached = await redis_client.get(cache_key)
    if cached is not None:
        return cached.decode() or None  # empty string sentinel ⇒ explicitly no prelude

    lines: list[str] = []

    # Only one case left: top-level @-mention in a channel (triggering_ts == thread_ts).
    # Pull a short tail of channel messages preceding the mention so the bot can see
    # what was being discussed.
    if triggering_ts == thread_ts:
        history = await slack_client.conversations_history(
            channel=channel_id, latest=thread_ts, inclusive=False, limit=20,
        )
        if history:
            lines.append("# Channel context preceding this mention")
            lines.append("")
            lines.append("These messages were posted in the channel before the user mentioned you. "
                         "Treat them as background context, not as messages addressed to you.")
            lines.append("")
            for m in reversed(history):  # oldest first
                lines.append(format_message(m))

    prelude = "\n".join(lines) if lines else ""
    await redis_client.set(cache_key, prelude or "", ex=60 * 60 * 24 * 90)
    return prelude or None
```

When a `@`-mention adopts an existing channel thread (i.e., `triggering_ts != thread_ts`), no prelude is generated — the pre-mention thread content is already in `chat_messages` via the `conversations.replies` replay.

The Redis cache (`slack_prelude:{conversation_uuid}`, 90-day TTL) keeps subsequent turns from re-fetching channel history every time.

The prelude is injected via the **instruction message** (system/developer), not as a `ChatMessage`. This matches the compaction architecture's treatment of `ancestor_summaries` — background context, not attributed conversational turns.

---

## Per-turn flow (`_run_turn`)

This is the Slack analogue of `WebHarness.post_chat`. The skeleton:

```python
async def _run_turn(
    self,
    *,
    chat_conversation: ChatConversation,
    channel_id: str,
    event: dict,
    prelude: str | None,
    slack_client: SlackClientWithToken,
    thread_ts: str,
) -> None:
    context_partition = await self.sync_context_partition(chat_conversation)

    think_tool = ThinkTool(self.llm_client, self.default_model)
    tool_callbacks = {think_tool.name: think_tool}
    # FileTool / ShellCommandTool intentionally omitted; see Tool Selection.

    # event["user"] is the Slack user ID; resolve to a display name via users.info.
    # resolve_display_names is the same helper build_chat_conversation uses; it caches
    # in Redis so this typically hits cache.
    display_names = await resolve_display_names(slack_client, self.redis_client, {event["user"]})
    originator_display_name = display_names.get(event["user"]) or "Slack user"

    instruction_parts = self._build_instruction_parts(
        context_partition=context_partition,
        originator_display_name=originator_display_name,
        prelude=prelude,
        tool_callbacks=tool_callbacks,
    )
    instruction_message = ContextPartitionItem(
        role=self.instruction_role,
        content="\n".join(instruction_parts),
    )
    context_partition.items.insert(0, instruction_message)

    pending_compaction = [False]

    def on_usage(input_tokens, output_tokens):
        ctx_pct = int(input_tokens / MODEL_CONTEXT_WINDOWS.get(self.default_model, DEFAULT_CONTEXT_WINDOW) * 100)
        if ctx_pct >= COMPACTION_TOKEN_THRESHOLD_PCT:
            pending_compaction[0] = True

    async def compact(snapshot):
        return await self._summarize_and_compact(snapshot=snapshot, model=self.default_model)

    streamer = SlackStreamer(
        slack_client=slack_client,
        channel_id=channel_id,
        thread_ts=thread_ts,
        # Prefix the bot's reply with <@user> for channels and mpim, so the user who
        # triggered the @-mention gets a notification. DMs skip the prefix.
        reply_to_user_id=event["user"] if event.get("channel_type") != "im" else None,
    )
    await streamer.post_placeholder()

    try:
        async for line in self.stream_and_finalize(
            context_partition=context_partition,
            conversation_uuid=chat_conversation.conversation_uuid,
            response_generator=self.llm_client.stream_turn(
                context_partition=context_partition,
                model=self.default_model,
                on_usage=on_usage,
                stream_ndjson=True,
                tool_callbacks=tool_callbacks,
            ),
            pending_compaction=pending_compaction,
            compact_fn=compact,
        ):
            await streamer.consume(line)
    except Exception:
        await streamer.fail()  # post a short error message in the thread
        raise
    finally:
        await streamer.finish()
```

`_build_instruction_parts` reuses `system_message_utils.get_core_instruction_parts` and `get_runtime_context_parts` like the web harness, but:

- replaces `# User context` with `# Slack context` ("originator", channel name, workspace name)
- injects `prelude` (when present) as a dedicated section before `# Personality`
- injects `context_partition.ancestor_summary_block()` for OpenAI (same provider-specific handling as `WebHarness`)
- adds a short note telling the model not to start its reply with `<@user>` — the harness prepends that mechanically, and the model emitting one would double-mention the user

---

## Reply Streaming to Slack

`stream_and_finalize` was designed to feed an HTTP `StreamingResponse`. For Slack we consume its NDJSON output server-side and translate to Slack API calls. `SlackStreamer` is a small stateful helper:

```python
class SlackStreamer:
    """Consume the NDJSON event stream and post / update Slack thread messages.

    Behavior:
    - Post a placeholder thread reply on start (so the user gets immediate feedback).
      The placeholder, and every subsequent message posted by the streamer, is prefixed
      with f"<@{reply_to_user_id}> " when reply_to_user_id is set (channels and mpim).
      DMs pass reply_to_user_id=None and post unprefixed.
    - Buffer text_delta chunks. Flush via chat.update at most every FLUSH_INTERVAL_SECONDS
      (default 1.0) or when the buffer crosses FLUSH_CHARS (default 1500).
    - When the buffered text would exceed Slack's per-message limit (~3500 chars to leave
      room for Markdown overhead), seal the current thread message at the last paragraph
      break and post a fresh continuation message. The continuation posts do not repeat
      the <@user> prefix — only the first message in a reply carries it.
    - On tool_call / progress_message events, update an ephemeral status line on the
      most recent message ("Calling shell_command…") rather than appending to the body.
    - On compaction_pending, no-op (background compaction is invisible in Slack).
    - On stream end, do a final flush so the latest buffer state is committed.
    """

    FLUSH_INTERVAL_SECONDS = 1.0
    FLUSH_CHARS = 1500
    SLACK_MESSAGE_SOFT_LIMIT = 3500

    async def consume(self, ndjson_line: str) -> None:
        event = json.loads(ndjson_line)
        match event:
            case {"partition_uuid": _}: pass  # Slack ignores this
            case {"text_delta": chunk}: await self._append_text(chunk)
            case {"tool_call": payload}: await self._render_tool_call(payload)
            case {"progress_message": payload}: await self._render_progress(payload)
            case {"compaction_pending": True}: pass
            case _: pass  # forward-compat: ignore unknown events
```

Practical considerations:

1. **Rate limits.** Slack's `chat.update` is roughly 1 req/sec per channel. The 1 Hz flush interval is chosen to match. If LLM tokens arrive faster, the buffer absorbs the difference.
2. **Split points.** When a single response approaches `SLACK_MESSAGE_SOFT_LIMIT`, split at the last `\n\n` or `\n` before the limit. Each split is a separate `chat.postMessage` call into the same `thread_ts`. The harness records the resulting `ts` so subsequent token flushes target the right Slack message.
3. **Markdown.** Slack `mrkdwn` differs from GitHub-flavored Markdown. Post in `mrkdwn` mode; code fences render, tables degrade to plain text. A small `format_for_slack` translator (`**bold**` → `*bold*`, etc.) is nice-to-have.
4. **Tool-call surfacing.** `tool_call` and `progress_message` events from the web NDJSON protocol describe transient activity. In Slack we render them as an italicized status line appended to the in-progress message body, e.g. `_…running shell_command (ls -la)_`. The status line is rewritten on each new tool event and removed at end-of-stream so the final message is clean.
5. **Failure surface.** If the LLM stream raises, the streamer replaces the placeholder with a short user-facing error and re-raises so `log_async_task_exception` records it.

---

## Compaction Integration

Compaction works without changes:

- `on_usage` sets `pending_compaction[0] = True` at the configured threshold, exactly as in the web harness.
- `stream_and_finalize` acquires `compaction_lock:{conversation_uuid}` and fires `_compact_partition` in the background.
- The CAS swap commits the child partition to Redis; the next inbound Slack event fast-paths off the new partition.
- The internal NDJSON `compaction_pending` event (emitted for the browser UI) has no Slack counterpart; we drop it.

`find_latest_active_partition_uuid` is the only addition to the partition-sync surface — needed because Slack can't echo a `partition_uuid` back like the browser client does.

---

## Tool Selection

Default tool set: **`ThinkTool` only**.

`FileTool` and `ShellCommandTool` operate on `Path.cwd()` on the host running the bot. Exposing them to anyone who can `@`-mention the bot in a connected workspace is a privilege expansion the design declines. `ThinkTool` is a pure reasoning aid with no side effects.

---

## Auth and Security

**Inbound.** No HTTP surface. Socket Mode is an outbound WebSocket authenticated at connect time by the workspace's `xapp-` token (`SLACK_APP_TOKEN`).

**Outbound.** Bot token (`SLACK_BOT_TOKEN`) read once at startup and curried through `SlackClientWithToken` for each turn.

**Idempotency.** `event_id` claimed via Redis `SET NX EX 600`. Socket Mode redelivers if the envelope isn't acked within ~3s; dedup catches those.

**Scope minimum** (configured per workspace's Slack app): `app_mentions:read`, `chat:write`, `channels:history`, `groups:history`, `im:history`, `im:read`, `im:write`, `mpim:history`, `mpim:read`, `users:read`. Plus the app-level scope `connections:write` on the `xapp-` token. Event subscriptions: `app_mention`, `message.channels`, `message.groups`, `message.im`, `message.mpim`, `tokens_revoked`, `app_uninstalled`.

**Local-trust assumption.** Tokens live in env vars on the harness host. Anyone with access to `docker-compose.yml`, `docker inspect`, or the running container's `/proc/self/environ` can read them. The trust boundary is the host, not the env var.

---

## Deployment

### Environment variables (per harness instance)

| Variable | Purpose |
|---|---|
| `SLACK_APP_TOKEN` | `xapp-...` app-level token from the workspace's Slack app config |
| `SLACK_BOT_TOKEN` | `xoxb-...` bot token from "Install App → Install to Workspace" |
| `SLACK_HARNESS_IMPL` | `anthropic` (default) or `openai` |
| Existing: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, `REDIS_*`, `ELASTIC_URI` | Inherited backend env |

`POSTGRES_*` is intentionally absent — the Slack harness does not connect to Postgres. The `chat_user` table is only used by the web chat UI; Slack identifies users via Slack user IDs.

### Docker Compose

One `slack-<workspace-name>` service per workspace. Example with two workspaces:

```yaml
slack-acme:
  build: .
  command: python -m scripts.slack
  environment:
    SLACK_HARNESS_IMPL: anthropic
    SLACK_APP_TOKEN: ${ACME_SLACK_APP_TOKEN}
    SLACK_BOT_TOKEN: ${ACME_SLACK_BOT_TOKEN}
    ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    # ...Redis, ES env
  depends_on:
    - elasticsearch
    - elasticsearch-init
    - redis
  restart: unless-stopped

slack-personal:
  build: .
  command: python -m scripts.slack
  environment:
    SLACK_HARNESS_IMPL: anthropic
    SLACK_APP_TOKEN: ${PERSONAL_SLACK_APP_TOKEN}
    SLACK_BOT_TOKEN: ${PERSONAL_SLACK_BOT_TOKEN}
    ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    # ...Redis, ES env
  depends_on:
    - elasticsearch
    - elasticsearch-init
    - redis
  restart: unless-stopped
```

No `ports:` entries — the Slack harness exposes nothing. Slack reaches it via outbound Socket Mode. Outbound network from each container needs to reach `wss://wss-primary.slack.com` (Socket Mode) and `https://slack.com/api/` (Web API). No `healthcheck:` either — nothing depends on the Slack service, so failure-loop detection is a job for `restart: unless-stopped` plus log inspection.

### Slack app configuration checklist (per workspace)

Each workspace owner does this once for their workspace, then hands the two tokens to the operator (or runs the deploy themselves):

1. Create a new Slack app from the manifest in this doc: https://api.slack.com/apps?new_app=1 → "From manifest".
2. Basic Information → App-Level Tokens → Generate Token and Scopes. Scope: `connections:write`. Copy the `xapp-...` token.
3. Install App → Install to Workspace → Approve consent. Copy the `xoxb-...` bot token from OAuth & Permissions.
4. Add the `slack-<workspace-name>` service to `docker-compose.yml` with both tokens in env (or pass via a workspace-specific `.env` file).
5. `docker compose up -d slack-<workspace-name>`. Invite the bot to channels: `/invite @prokaryote`. Or DM `@prokaryote` directly.

The manifest declares `socket_mode_enabled: true`, no `request_url`, no `redirect_urls`. There is no public URL to configure on the Slack side and no signing secret to copy.

### Entry point (`scripts/slack.py`)

```python
import asyncio
import os

from dotenv import load_dotenv

from prokaryotes.harness_v1.slack import SlackHarness
from prokaryotes.utils_v1.logging_utils import setup_logging


async def run():
    harness = SlackHarness(
        impl=os.getenv("SLACK_HARNESS_IMPL", "anthropic"),
        app_token=os.environ["SLACK_APP_TOKEN"],
        bot_token=os.environ["SLACK_BOT_TOKEN"],
    )
    await harness.on_start()
    try:
        await asyncio.Event().wait()  # block forever; Socket Mode runs in background tasks
    finally:
        await harness.on_stop()


if __name__ == "__main__":
    load_dotenv()
    setup_logging()
    asyncio.run(run())
```

---

## Testing

### Unit tests

- `tests/unit_tests/test_slack_dedupe.py` — `event_id` claim races (two concurrent claims, one wins).
- `tests/unit_tests/test_slack_lifecycle.py` — `SlackBase.on_start` / `on_stop`:
  - happy path: `auth.test` succeeds → `team_id`, `bot_user_id`, `team_name` populated; socket connected.
  - `auth.test` fails (`ok: false`) → `on_start` raises; process is expected to exit.
  - `on_stop` disconnects and closes the socket; safe to call when `on_start` failed mid-way.
- `tests/unit_tests/test_slack_dispatch.py` — `SlackBase._dispatch_event` and `_should_handle`:
  - listener acks via `send_socket_mode_response` before calling `_dispatch_event`.
  - `events_api` envelope with `app_mention` → `handle_event` invoked.
  - `tokens_revoked` / `app_uninstalled` → `on_stop` called; `handle_event` not invoked.
  - duplicate `event_id` dropped.
  - envelope `team_id` mismatch with `self.team_id` → dropped with a warning.
  - `_should_handle` accepts `app_mention` (top-level and threaded), `message.im`; rejects threaded `message` events in channels and mpim that lack an `@`-mention, bot-authored messages, and subtyped messages.
- `tests/unit_tests/test_slack_replay.py` — `build_chat_conversation`:
  - Top-level channel mention → one user message (no prelude in `messages`)
  - Mention adopting an existing channel thread → pre-mention thread messages appear once, as `user` `ChatMessage`s with display-name prefixes; no prelude (pre-mention content is in `chat_messages`, not duplicated as prelude)
  - In-thread non-`@` chatter between two follow-up `@`-mentions appears as `user` `ChatMessage`s preserving the conversation flow
  - Top-level DM → one user message; `thread_ts` resolves to `event["ts"]`; no prelude
  - DM thread reply → previous DM thread messages reconstructed via `conversations.replies`; no prelude
  - Multi-human thread → display-name prefixes; single-human and DM → bare content
  - Bot messages split across N Slack posts collapse into one `assistant` `ChatMessage`
  - `triggering_ts` truncates late-arriving messages from the replay
- `tests/unit_tests/test_slack_prelude.py` — channel-history prelude caching, format, top-level channel `@`-mention case, DM short-circuit returns `None` without contacting Slack, adopted-thread `@`-mention returns `None` (no in-thread prelude — replay covers it).
- `tests/unit_tests/test_slack_streamer.py` — NDJSON consumption and reply formatting:
  - `text_delta` buffering and 1 Hz flush cadence
  - Soft-limit splitting at paragraph boundaries; only the first message carries the `<@user>` prefix, continuation posts do not
  - `reply_to_user_id` set → placeholder and final message both start with `<@user> `
  - `reply_to_user_id=None` (DM) → no prefix on placeholder or final message
  - `tool_call` / `progress_message` rendering and clearing at end-of-stream
  - `compaction_pending` is a no-op
  - failure path posts the error message and re-raises
- `tests/unit_tests/test_slack_recovery.py` — `resolve_partition_uuid`:
  - Redis cached → returns `None` (use fast path)
  - Redis cold, ES has active → returns the ES `partition_uuid`
  - Redis cold, ES empty → returns `None`
- `tests/unit_tests/test_search_v1_context_partitions.py` — extend coverage of the new `find_latest_active_partition_uuid` query, including ignoring `compaction_state=pending` and `is_compacted=true` docs.

### Integration tests (Tier B)

- `tests/integration_tests/tier_b/test_slack_flow.py` — fake `SocketModeClient` and fake `SlackClient` against a real Redis / Elasticsearch (no Postgres required — Slack harness doesn't use it). One `SlackHarness` instance bound to one synthetic workspace via env tokens:
  - Harness `on_start` resolves `team_id` / `bot_user_id` / `team_name` via fake `auth.test`; socket connects.
  - First channel `app_mention` creates a conversation, posts a placeholder prefixed with `<@triggering_user>`, finalizes the assistant message in the thread.
  - Non-mention thread reply by another human is dropped; bot does **not** post.
  - Second `@`-mention in the same thread (by either user) continues the conversation off the Redis fast path; the bot's reply is prefixed with `<@second_triggering_user>` and the prior chatter is visible to the model as context.
  - Top-level DM message creates a conversation and replies as a thread on the user's DM (no `<@user>` prefix); subsequent in-thread DM replies continue it without requiring an `@`-mention.
  - mpim top-level chatter without an `@`-mention is dropped; an mpim `@`-mention creates a conversation and replies in a thread; non-mention thread replies are dropped; another `@`-mention continues the conversation.
  - Envelope-id ack happens before any LLM work begins (asserted by examining the fake socket's response log).
  - After artificially evicting the Redis key, a fresh `@`-mention in an existing thread rebuilds from ES via `find_latest_active_partition_uuid` and chain reconstruction (or starts fresh when no active partition exists).
  - Compaction trigger: artificially push `on_usage` past the threshold and verify the background swap commits a child partition; verify the next `@`-mention uses the child.
  - Duplicate `event_id` is ignored.
  - `app_uninstalled` event over the socket → `on_stop` is invoked; subsequent events are not delivered.

### UI / smoke

A Tier A smoke against the live Slack API is **not** included by default — it requires a sandbox workspace and live tokens. Document the manual smoke procedure in `tests/README.md` instead.

---

## Open Questions

1. **Tokens at rest.** v1 keeps `SLACK_APP_TOKEN` and `SLACK_BOT_TOKEN` as plain env values. For a local multi-user service the trust boundary is the host, not the env var — but if the operator wants tokens not to be readable via `docker inspect` / `/proc/.../environ`, a secret store (Docker secrets, sops, etc.) is a normal next step.
2. **Edits and deletes.** Slack `message_changed` and `message_deleted` events are dropped by `_should_handle` today. If an edit happens after we replied, the next inbound event replays the (now-modified) thread, hits a divergence in `sync_from_conversation`, and truncates — same retry mechanic the browser uses. Acceptable. Deletes leave gaps; the model sees a shorter history. Also acceptable for v1.
3. **Simultaneous `@`-mentions in the same thread.** Much less likely with the `@`-only trigger rule, but two users could still `@prokaryote` the bot at the same instant. Two background tasks would then race on the same `conversation_uuid`. The compaction lock protects compaction itself, but the LLM-turn `finalize`s could overwrite each other. Cheap mitigation: `slack_turn_lock:{conversation_uuid}` (Redis SET NX, short TTL) around `_run_turn`; the loser posts "I'm still answering the previous question, one moment…" and exits. Worth including; not blocking for v1.
4. **Prelude size.** 20 channel messages is arbitrary. Token budget could be 1500–2000 instead of message count. Worth tuning during integration testing.
5. **Tool surface expansion.** Slack-specific tools (search, user info, channel history) are mentioned as future work. The harness should be structured so adding them is a one-line registration; nothing in the design precludes that.
6. **Posting Markdown.** Slack's `mrkdwn` differs from GitHub-flavored Markdown. A simple translation (`**bold**`→`*bold*`, strip headings) is enough for v1, but ideal rendering may want richer Block Kit formatting (sections, dividers). Defer.
7. **Assistant API.** Slack's "assistant" thread features (`assistant.threads.setStatus`, typing indicators) would improve UX but require the `assistant:write` scope and slightly different event shapes. Optional polish; design above does not depend on it.
8. **Crash-loop backstop.** Persistent token failures degrade into a tight `restart: unless-stopped` loop (see Token revocation). A future "stop trying after N consecutive failures" backstop would be nicer; not in scope for v1.

---

## Out of Scope for v1

- Slash commands and message shortcuts
- Modals and interactive components
- Slack Connect / shared channels
- Slack App Directory distribution (the design is single-instance, locally hosted)
- OAuth v2 install flow (intentionally absent — each workspace is a separately-created app)
- One harness instance serving multiple workspaces (intentionally absent — add another `slack-*` service instead)
- Token encryption at rest (see Open Questions)
- File uploads to/from Slack
- Per-user identity mapping to internal accounts (Slack-side users are not mapped to `chat_user` rows)

---

## Relevant Code Files (expected)

| File | Role |
|---|---|
| `prokaryotes/harness_v1/slack.py` (new) | `SlackHarness(SlackBase)` — composes Socket Mode + LLM client + tools (ThinkTool) |
| `prokaryotes/slack_v1/__init__.py` (new) | `SlackBase(HarnessBase)` — Socket Mode lifecycle, `_listener`, `_dispatch_event`, `_should_handle` |
| `prokaryotes/slack_v1/client.py` (new) | `SlackClient` (token-less) + `SlackClientWithToken` (per-turn currying wrapper) |
| `prokaryotes/slack_v1/replay.py` (new) | `build_chat_conversation`, `collapse_consecutive_bot_messages`, `sanitize_mentions`, `resolve_partition_uuid`, `build_prelude`, `resolve_display_names` |
| `prokaryotes/slack_v1/streaming.py` (new) | `SlackStreamer` — NDJSON consumer that posts/updates Slack thread messages |
| `prokaryotes/search_v1/context_partitions.py` | New method `find_latest_active_partition_uuid(conversation_uuid)` |
| `scripts/slack.py` (new) | Entry point — reads env, instantiates `SlackHarness`, runs `on_start`/await/`on_stop` |
| `docker-compose.yml` | One `slack-<name>` service per connected workspace |
| `tests/unit_tests/test_slack_*.py` (new) | Per the testing section above |
| `tests/integration_tests/tier_b/test_slack_flow.py` (new) | Tier B end-to-end against fake Slack |
