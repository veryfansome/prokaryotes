"""Hermetic fakes for the Slack-harness overlay unit tests.

In-memory stand-ins for Redis, the search client, the Slack `SocketModeClient`, and the Slack Web-API client.
Everything the Slack harness tests touch is faked here so the suite runs in the plain unit tier with no Docker
and no live Slack / LLM credentials.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from prokaryotes.conversation_v1.models import Conversation, TurnExecution
from prokaryotes.search_v1.conversations import turn_execution_from_doc


class FakeRedis:
    """In-memory dict-shaped fake for the subset of `redis.asyncio.Redis` the Slack harness uses.

    Stores values as bytes (what the real client returns on `get`); `set` accepts str or bytes and honors `nx`.
    `ex` is recorded but not enforced — the unit tests do not measure TTL.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self.set_calls: list[tuple[str, bytes, int | None, bool]] = []

    async def aclose(self) -> None:
        pass

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self._store:
                self._store.pop(key)
                removed += 1
        return removed

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    async def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    async def set(self, key: str, value: str | bytes, ex: int | None = None, nx: bool = False) -> bool | None:
        if nx and key in self._store:
            return None
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._store[key] = value
        self.set_calls.append((key, value, ex, nx))
        return True


class FakeSearchClient:
    """In-memory fake for the overlay `SearchClient` surface the Slack harness reaches.

    `conversations` is keyed by `snapshot_uuid`; `turn_executions` by `(conversation_uuid, bot_message_source_id)`.
    `find_latest_active_snapshot_uuid` mirrors the production query: it ignores `is_compacted=True` and any
    `compaction_state != "committed"` doc and returns the most recently modified survivor.
    """

    def __init__(self) -> None:
        self.conversations: dict[str, dict[str, Any]] = {}
        self.turn_executions: dict[tuple[str, str], dict[str, Any]] = {}
        self.put_conversation_calls: list[str] = []
        self.put_turn_execution_calls: list[TurnExecution] = []
        self.es = None
        self.put_conversation_error: Exception | None = None

    async def close(self) -> None:
        pass

    async def delete_turn_execution(self, conversation_uuid: str, bot_message_source_id: str) -> None:
        self.turn_executions.pop((conversation_uuid, bot_message_source_id), None)

    async def find_all_conversation_docs(self, conversation_uuid: str) -> list[dict[str, Any]]:
        return [d for d in self.conversations.values() if d.get("conversation_uuid") == conversation_uuid]

    async def find_latest_active_child(
        self, conversation_uuid: str, parent_snapshot_uuid: str
    ) -> dict[str, Any] | None:
        candidates = [
            d
            for d in self.conversations.values()
            if d.get("conversation_uuid") == conversation_uuid
            and d.get("parent_snapshot_uuid") == parent_snapshot_uuid
            and not d.get("is_compacted")
            and d.get("compaction_state") == "committed"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.get("dt_modified", ""))

    async def find_latest_active_snapshot_uuid(self, conversation_uuid: str) -> str | None:
        candidates = [
            d
            for d in self.conversations.values()
            if d.get("conversation_uuid") == conversation_uuid
            and not d.get("is_compacted")
            and d.get("compaction_state") == "committed"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.get("dt_modified", ""))["snapshot_uuid"]

    async def get_conversation(self, snapshot_uuid: str) -> dict[str, Any] | None:
        return self.conversations.get(snapshot_uuid)

    async def get_turn_execution(self, conversation_uuid: str, bot_message_source_id: str) -> TurnExecution | None:
        doc = self.turn_executions.get((conversation_uuid, bot_message_source_id))
        return turn_execution_from_doc(doc) if doc is not None else None

    async def get_turn_executions(
        self, conversation_uuid: str, bot_message_source_ids: list[str]
    ) -> dict[str, TurnExecution]:
        out: dict[str, TurnExecution] = {}
        for sid in bot_message_source_ids:
            doc = self.turn_executions.get((conversation_uuid, sid))
            if doc is not None:
                out[sid] = turn_execution_from_doc(doc)
        return out

    async def put_conversation(
        self,
        conversation: Conversation,
        *,
        compaction_attempt_uuid: str | None = None,
        compaction_state: str = "committed",
        refresh: str | bool = False,
    ) -> None:
        if self.put_conversation_error is not None:
            raise self.put_conversation_error
        self.put_conversation_calls.append(conversation.snapshot_uuid)
        self.conversations[conversation.snapshot_uuid] = self._doc(conversation, compaction_state)

    async def put_turn_execution(self, turn: TurnExecution) -> None:
        self.put_turn_execution_calls.append(turn)
        now = datetime.now(UTC).isoformat()
        self.turn_executions[(turn.conversation_uuid, turn.bot_message_source_id)] = {
            "bot_message_source_id": turn.bot_message_source_id,
            "conversation_uuid": turn.conversation_uuid,
            "items_json": json.dumps({"items": [i.model_dump() for i in turn.items]}),
            "completed": turn.completed,
            "dt_created": now,
            "dt_modified": now,
        }

    async def rekey_turn_execution(self, conversation_uuid: str, old_id: str, new_id: str) -> None:
        old_key = (conversation_uuid, old_id)
        if old_key not in self.turn_executions:
            return
        doc = dict(self.turn_executions[old_key])
        doc["bot_message_source_id"] = new_id
        self.turn_executions[(conversation_uuid, new_id)] = doc
        self.turn_executions.pop(old_key, None)

    def store_conversation_doc(
        self,
        conversation: Conversation,
        *,
        is_compacted: bool = False,
        compaction_state: str = "committed",
        dt_modified: str | None = None,
    ) -> None:
        """Persist a `Conversation` as a raw ES doc, bypassing `put_conversation`.

        Lets a test stage a compacted ancestor or a pending child directly so cold-recovery and the
        active-snapshot query can be exercised.
        """
        doc = self._doc(conversation, compaction_state)
        doc["is_compacted"] = is_compacted
        if dt_modified is not None:
            doc["dt_modified"] = dt_modified
        self.conversations[conversation.snapshot_uuid] = doc

    @staticmethod
    def _doc(conversation: Conversation, compaction_state: str) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        return {
            "snapshot_uuid": conversation.snapshot_uuid,
            "conversation_uuid": conversation.conversation_uuid,
            "parent_snapshot_uuid": conversation.parent_snapshot_uuid,
            "bot_author_id": conversation.bot_author_id,
            "compaction_state": compaction_state,
            "compaction_attempt_uuid": None,
            "is_compacted": False,
            "summary": None,
            "ancestor_summaries": list(conversation.ancestor_summaries),
            "working_file_windows_json": json.dumps(
                {"windows": [w.model_dump() for w in conversation.working_file_windows]}
            ),
            "messages_json": json.dumps({"messages": [m.model_dump() for m in conversation.messages]}),
            "raw_message_start_index": conversation.raw_message_start_index,
            "dt_created": now,
            "dt_modified": now,
        }


class FakeSocketModeClient:
    """Stand-in for `slack_sdk`'s `SocketModeClient`.

    Records the order of lifecycle calls (`connect` / `disconnect` / `close`) and exposes
    `socket_mode_request_listeners` so `SlackBase.on_start` can append its listener.
    """

    def __init__(self) -> None:
        self.socket_mode_request_listeners: list[Any] = []
        self.calls: list[str] = []
        self.connected = False

    async def close(self) -> None:
        self.calls.append("close")

    async def connect(self) -> None:
        self.calls.append("connect")
        self.connected = True

    async def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.connected = False


class FakeSlackClient:
    """Stand-in for `prokaryotes.slack_v1.client.SlackClient`.

    Implements only `auth_test` / `resolve_app_id` / `close` — the calls `SlackBase.on_start` / `on_stop` make.
    `auth_ok=False` makes `auth_test` report failure so the `on_start` guard can be exercised.
    """

    def __init__(self, *, auth_ok: bool = True, app_id: str | None = "A_APP") -> None:
        self._auth_ok = auth_ok
        self._app_id = app_id
        self.closed = False
        self.auth_test_calls = 0
        self.resolve_app_id_calls = 0

    async def auth_test(self, *, bot_token: str) -> dict:
        self.auth_test_calls += 1
        if not self._auth_ok:
            return {"ok": False, "error": "invalid_auth"}
        return {
            "ok": True,
            "team_id": "T_TEAM",
            "user_id": "U_BOT",
            "bot_id": "B_BOT",
            "team": "Acme",
        }

    async def close(self) -> None:
        self.closed = True

    async def resolve_app_id(self, *, bot_token: str, bot_user_id: str) -> str | None:
        self.resolve_app_id_calls += 1
        return self._app_id


class FakeSlackThreadClient:
    """In-memory `SlackClientWithToken`-shaped fake for replay / streamer tests.

    Backs `conversations_replies` / `conversations_history` with a caller-supplied thread / channel-history list,
    records every `chat_delete` / `chat_update` / `chat_post_message` call, and resolves `users_info` from a
    static `display_names` map. `users_info` raises for any ID listed in `users_info_failures`.
    """

    def __init__(
        self,
        *,
        thread: list[dict] | None = None,
        history: list[dict] | None = None,
        display_names: dict[str, str] | None = None,
        users_info_failures: set[str] | None = None,
    ) -> None:
        self.thread = thread if thread is not None else []
        self.history = history if history is not None else []
        self._display_names = display_names or {}
        self._users_info_failures = users_info_failures or set()
        self.chat_delete_calls: list[str] = []
        self.chat_update_calls: list[dict] = []
        self.chat_post_calls: list[dict] = []
        self.replies_calls: list[dict] = []
        self.users_info_calls: list[str] = []
        self.chat_delete_error: Exception | None = None
        self.chat_update_error: Exception | None = None
        self.next_post_ts: list[str] = []
        self._post_counter = 0

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
        self.replies_calls.append(
            {
                "channel": channel,
                "ts": ts,
                "oldest": oldest,
                "inclusive": inclusive,
                "include_all_metadata": include_all_metadata,
                "paginate_until_ts": paginate_until_ts,
            }
        )
        out = self.thread
        if oldest is not None:
            out = [m for m in out if (m["ts"] > oldest or (inclusive and m["ts"] == oldest))]
        return list(out)

    async def conversations_history(
        self,
        *,
        channel: str,
        latest: str | None = None,
        oldest: str | None = None,
        inclusive: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        out = self.history
        if latest is not None:
            out = [m for m in out if (m["ts"] < latest or (inclusive and m["ts"] == latest))]
        return list(out[:limit])

    async def chat_delete(self, *, channel: str, ts: str) -> dict:
        self.chat_delete_calls.append(ts)
        if self.chat_delete_error is not None:
            raise self.chat_delete_error
        return {"ok": True}

    async def chat_update(
        self,
        *,
        channel: str,
        ts: str,
        text: str | None = None,
        blocks: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        self.chat_update_calls.append({"ts": ts, "text": text, "metadata": metadata})
        if self.chat_update_error is not None:
            raise self.chat_update_error
        return {"ok": True, "ts": ts}

    async def chat_post_message(
        self,
        *,
        channel: str,
        thread_ts: str | None = None,
        text: str | None = None,
        blocks: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        if self.next_post_ts:
            ts = self.next_post_ts.pop(0)
        else:
            # Default ts is "next microsecond after the latest message in the thread" — mirrors how Slack
            # assigns chat.postMessage ts (~current time, after the trigger that fired the turn). This makes
            # bot replies sort between the user mentions that bracket them in multi-turn integration tests,
            # instead of being lumped at the tail. Tests that need exact control still stage `next_post_ts`.
            latest = max((m.get("ts", "") for m in self.thread), default="")
            if latest:
                seconds, _, micros_str = latest.partition(".")
                try:
                    next_micros = int(micros_str or "0") + 1
                    ts = f"{seconds}.{next_micros:06d}"
                except ValueError:
                    self._post_counter += 1
                    ts = f"9000.{self._post_counter:06d}"
            else:
                self._post_counter += 1
                ts = f"9000.{self._post_counter:06d}"
        self.chat_post_calls.append({"ts": ts, "text": text, "metadata": metadata, "thread_ts": thread_ts})
        return {"ok": True, "ts": ts}

    async def users_info(self, *, user: str) -> dict:
        self.users_info_calls.append(user)
        if user in self._users_info_failures:
            raise RuntimeError(f"users.info failed for {user}")
        name = self._display_names.get(user)
        return {"ok": True, "user": {"name": name, "profile": {"display_name": name}}}


def envelope(event: dict, *, event_id: str | None = "Ev1", team_id: str | None = "T_TEAM") -> dict:
    """Build an `events_api` Socket Mode envelope dict wrapping `event`."""
    payload: dict[str, Any] = {"event": event}
    if event_id is not None:
        payload["event_id"] = event_id
    if team_id is not None:
        payload["team_id"] = team_id
    return {"type": "events_api", "payload": payload}
