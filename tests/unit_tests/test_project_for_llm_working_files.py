"""`project_for_llm` working-file emission + historical-output filtering.

Exercises the `WorkingFileWindow` contract with required `line_count` / `origin_call_ids`. `project.py` itself is
unchanged; this pins that the leading `<working_files>` block and the historical working-file output drop still
behave under the window shape.
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    TurnExecution,
    TurnItem,
    WorkingFileWindow,
)
from prokaryotes.conversation_v1.project import project_for_llm

BOT_ID = "b"


def _msg(source_id: str, content: str, *, author_id: str = "u-alice") -> ConversationMessage:
    return ConversationMessage(source_id=source_id, author_id=author_id, content=content)


def _bot(source_id: str, content: str) -> ConversationMessage:
    return ConversationMessage(source_id=source_id, author_id=BOT_ID, content=content)


def _conv(*messages: ConversationMessage, working_file_windows: list[WorkingFileWindow] | None = None) -> Conversation:
    return Conversation(
        conversation_uuid="c-1",
        bot_author_id=BOT_ID,
        messages=list(messages),
        working_file_windows=working_file_windows or [],
    )


def _window(window_id: str, path: str = "/abs/a.py", view_end_line: int = 40) -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path=path,
        status="live",
        revision="r1",
        rendered_output=f"FILE path={path} revision=r1 status=live",
        view_start_line=1,
        view_end_line=view_end_line,
        requested_end_line=view_end_line,
        line_count=view_end_line,
        origin_call_ids=[window_id],
        source_kind="read_lines",
    )


class TestLeadingWorkingFilesBlock:
    def test_no_windows_no_block(self):
        items = project_for_llm(_conv(_msg("1", "hi"), _bot("2", "hello")))
        assert [(p.role, p.content) for p in items] == [
            ("user", "hi"),
            ("assistant", "hello"),
        ]

    def test_block_leads_when_first_message_is_bot(self):
        c = _conv(_bot("1", "bot-only"), working_file_windows=[_window("c-1")])
        items = project_for_llm(c)
        assert items[0].role == "user"
        assert items[0].content is not None
        assert items[0].content.startswith('<working_files trust="file-content">\n')
        assert items[1].role == "assistant"

    def test_block_merges_with_adjacent_user_first_message(self):
        c = _conv(_msg("1", "user-msg"), working_file_windows=[_window("c-1")])
        items = project_for_llm(c)
        assert items[0].role == "user"
        content = items[0].content or ""
        assert content.startswith('<working_files trust="file-content">\n')
        # The user message follows the working-files block after \n\n
        assert content.endswith("\n\nuser-msg")

    def test_block_follows_summary_block(self):
        c = Conversation(
            conversation_uuid="c-1",
            bot_author_id=BOT_ID,
            messages=[_msg("1", "user-msg")],
            ancestor_summaries=["earlier summary"],
            working_file_windows=[_window("c-1")],
        )
        items = project_for_llm(c)
        content = items[0].content or ""
        summary_idx = content.index("<compacted_summary")
        files_idx = content.index("<working_files")
        user_idx = content.index("user-msg")
        assert summary_idx < files_idx < user_idx


class TestHistoricalWorkingFileFilter:
    def test_historical_working_file_outputs_and_paired_calls_are_dropped(self):
        """A bot's TurnExecution carrying a file_tool function_call/output pair with persistence=working_file
        should be omitted from the projection — only that pair, not other items in the same TurnExecution."""
        turn = TurnExecution(
            conversation_uuid="c-1",
            bot_message_source_id="2",
            items=[
                TurnItem(type="function_call", call_id="c-a", name="file_tool", arguments="{}"),
                TurnItem(
                    type="function_call_output",
                    call_id="c-a",
                    output="<read body>",
                    prokaryotes_annotations={"file_tool.persistence": "working_file"},
                ),
                TurnItem(type="function_call", call_id="c-b", name="think", arguments="{}"),
                TurnItem(type="function_call_output", call_id="c-b", output="<thought>"),
            ],
        )
        items = project_for_llm(
            _conv(_msg("1", "ask"), _bot("2", "bot reply")),
            historical_turns={"2": turn},
        )
        # The file_tool function_call and its paired output are absent; the think pair survives.
        names_or_outputs = [
            (p.type, p.call_id, p.output) for p in items if p.type in {"function_call", "function_call_output"}
        ]
        call_ids = {entry[1] for entry in names_or_outputs}
        assert "c-a" not in call_ids
        assert "c-b" in call_ids

    def test_historical_history_persistence_outputs_are_kept(self):
        """Frozen edit records (persistence=history) ride forward in the transcript."""
        turn = TurnExecution(
            conversation_uuid="c-1",
            bot_message_source_id="2",
            items=[
                TurnItem(type="function_call", call_id="c-edit", name="file_tool", arguments="{}"),
                TurnItem(
                    type="function_call_output",
                    call_id="c-edit",
                    output="EDITED path=/abs/a.py ...",
                    prokaryotes_annotations={"file_tool.persistence": "history"},
                ),
            ],
        )
        items = project_for_llm(
            _conv(_msg("1", "ask"), _bot("2", "ok")),
            historical_turns={"2": turn},
        )
        call_ids = {p.call_id for p in items if p.type in {"function_call", "function_call_output"}}
        assert "c-edit" in call_ids
