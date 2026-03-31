import pytest
from fastapi import HTTPException
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.models_v1 import ChatMessage
from prokaryotes.web_v1 import ProkaryoteV1


@pytest.mark.parametrize("list1, list2, divergence_idx, is_mismatch, list1_is_longer", [
    (
        # Cached conversation matches payload
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        None,
        False,
        False,
    ),
    (
        # Payload retries previous prompt
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        3,
        False,
        True,
    ),
    (
        # Payload modifies previous prompt
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Nice to meet you"),
        ],
        2,
        True,
        True,
    ),
    (
        # Cached conversation is behind payload
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
            ChatMessage(role="user", content="Do one more"),
        ],
        3,
        False,
        False,
    ),
    (
        # Cached conversation is behind and payload has diverged
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Nice to meet you"),
            ChatMessage(role="assitant", content="Nice to meet you too - what's on your mind?"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        2,
        True,
        False,
    ),
])
def test_find_context_divergence(
        list1: list[ChatMessage],
        list2: list[ChatMessage],
        divergence_idx: int,
        is_mismatch: bool,
        list1_is_longer: bool,
):
    _divergence_idx, _is_mismatch, _list1_is_longer = ProkaryoteV1.find_context_divergence(list1, list2)
    assert _divergence_idx == divergence_idx
    assert _is_mismatch == is_mismatch
    assert _list1_is_longer == list1_is_longer


@pytest.mark.parametrize("cached_context_window, payload_messages, context_window_to_stream", [
    # Payload progresses the conversation
    (
        [
            ChatMessage(role="user", content="Hi"),
            FunctionCallOutput(
                call_id="123",
                output="... user preferences ...",
                type="function_call_output"
            ),
            ChatMessage(role="assitant", content="How can I help you"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            FunctionCallOutput(
                call_id="123",
                output="... user preferences ...",
                type="function_call_output"
            ),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
    ),
    # Payload retries previous prompt
    (
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            FunctionCallOutput(
                call_id="123",
                output="... joke candidates ...",
                type="function_call_output"
            ),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        # Retains function output
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            FunctionCallOutput(
                call_id="123",
                output="... joke candidates ...",
                type="function_call_output"
            ),
        ],
    ),
    # Payload modifies previous prompt
    (
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            FunctionCallOutput(
                call_id="123",
                output="... joke candidates ...",
                type="function_call_output"
            ),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Nice to meet you"),
        ],
        # Drops function output
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Nice to meet you"),
        ],
    ),
])
def test_sync_context_window(cached_context_window, payload_messages, context_window_to_stream):
    assert ProkaryoteV1.sync_context_window(cached_context_window, payload_messages) == context_window_to_stream


@pytest.mark.parametrize("cached_context_window, payload_messages", [
    (
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
        ],
    ),
])
def test_sync_context_window_exception(cached_context_window, payload_messages):
    with pytest.raises(HTTPException, match="Payload does not alter conversation state"):
        ProkaryoteV1.sync_context_window(cached_context_window, payload_messages)
