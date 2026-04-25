import pytest

from prokaryotes.api_v1.models import (
    ChatConversation,
    ChatMessage,
    ContextPartition,
    ContextPartitionItem,
    ConversationMatchesPartitionError,
    compute_boundary_hash,
    compute_tail_hash,
)


@pytest.mark.parametrize(
        "context_items, conversation_items, expected_divergence_idx, expected_is_mismatch, expected_is_longer",
        [
            (
                    # Context partition items matches payload conversation items
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                        ContextPartitionItem(
                            role="assistant",
                            content="Why don't scientists like atoms? Because they make up everything!",
                        ),
                    ],
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                        ContextPartitionItem(
                            role="assistant",
                            content="Why don't scientists like atoms? Because they make up everything!",
                        ),
                    ],
                    None,
                    False,
                    False,
            ),
            (
                    # Payload retries previous prompt
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                        ContextPartitionItem(
                            role="assistant",
                            content="Why don't scientists like atoms? Because they make up everything!",
                        ),
                    ],
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                    ],
                    3,
                    False,
                    True,
            ),
            (
                    # Payload modifies previous prompt
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                        ContextPartitionItem(
                            role="assistant",
                            content="Why don't scientists like atoms? Because they make up everything!",
                        ),
                    ],
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Nice to meet you"),
                    ],
                    2,
                    True,
                    True,
            ),
            (
                    # Cached conversation is behind payload
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                    ],
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                        ContextPartitionItem(
                            role="assistant",
                            content="Why don't scientists like atoms? Because they make up everything!",
                        ),
                        ContextPartitionItem(role="user", content="Do one more"),
                    ],
                    3,
                    False,
                    False,
            ),
            (
                    # Cached conversation is behind and payload has diverged
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                    ],
                    [
                        ContextPartitionItem(role="user", content="Hi"),
                        ContextPartitionItem(role="assistant", content="How can I help you"),
                        ContextPartitionItem(role="user", content="Nice to meet you"),
                        ContextPartitionItem(role="assistant", content="Nice to meet you too - what's on your mind?"),
                        ContextPartitionItem(role="user", content="Got any good jokes?"),
                    ],
                    2,
                    True,
                    False,
            ),
        ]
)
def test_find_context_divergence(
        context_items: list[ContextPartitionItem],
        conversation_items: list[ContextPartitionItem],
        expected_divergence_idx: int,
        expected_is_mismatch: bool,
        expected_is_longer: bool,
):
    divergence_idx, is_mismatch, is_longer = ContextPartition.find_context_divergence(
        context_items, conversation_items,
    )
    assert divergence_idx == expected_divergence_idx
    assert is_mismatch == expected_is_mismatch
    assert is_longer == expected_is_longer


@pytest.mark.parametrize("partition_items, conversation_messages, post_sync_items", [
    # Payload progresses the conversation
    (
            [
                ContextPartitionItem(role="user", content="Hi"),
                ContextPartitionItem(
                    call_id="123",
                    output="... user preferences ...",
                    type="function_call_output"
                ),
                ContextPartitionItem(role="assistant", content="How can I help you"),
            ],
            [
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="How can I help you"),
                ChatMessage(role="user", content="Got any good jokes?"),
            ],
            [
                ContextPartitionItem(role="user", content="Hi"),
                ContextPartitionItem(
                    call_id="123",
                    output="... user preferences ...",
                    type="function_call_output"
                ),
                ContextPartitionItem(role="assistant", content="How can I help you"),
                ContextPartitionItem(role="user", content="Got any good jokes?"),
            ],
    ),
    # Payload retries previous prompt
    (
            [
                ContextPartitionItem(role="user", content="Hi"),
                ContextPartitionItem(role="assistant", content="How can I help you"),
                ContextPartitionItem(role="user", content="Got any good jokes?"),
                ContextPartitionItem(
                    call_id="123",
                    output="... joke candidates ...",
                    type="function_call_output"
                ),
                ContextPartitionItem(
                    role="assistant",
                    content="Why don't scientists like atoms? Because they make up everything!"
                ),
            ],
            [
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="How can I help you"),
                ChatMessage(role="user", content="Got any good jokes?"),
            ],
            # Retains function output
            [
                ContextPartitionItem(role="user", content="Hi"),
                ContextPartitionItem(role="assistant", content="How can I help you"),
                ContextPartitionItem(role="user", content="Got any good jokes?"),
                ContextPartitionItem(
                    call_id="123",
                    output="... joke candidates ...",
                    type="function_call_output"
                ),
            ],
    ),
    # Payload modifies previous prompt
    (
            [
                ContextPartitionItem(role="user", content="Hi"),
                ContextPartitionItem(role="assistant", content="How can I help you"),
                ContextPartitionItem(role="user", content="Got any good jokes?"),
                ContextPartitionItem(
                    call_id="123",
                    output="... joke candidates ...",
                    type="function_call_output"
                ),
                ContextPartitionItem(
                    role="assistant",
                    content="Why don't scientists like atoms? Because they make up everything!",
                ),
            ],
            [
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="How can I help you"),
                ChatMessage(role="user", content="Nice to meet you"),
            ],
            # Drops function output
            [
                ContextPartitionItem(role="user", content="Hi"),
                ContextPartitionItem(role="assistant", content="How can I help you"),
                ContextPartitionItem(role="user", content="Nice to meet you"),
            ],
    ),
])
def test_sync_context_window(partition_items, conversation_messages, post_sync_items):
    context_partition = ContextPartition(conversation_uuid="test", items=partition_items)
    context_partition.sync_from_conversation(ChatConversation(conversation_uuid="abc", messages=conversation_messages))
    assert context_partition.items == post_sync_items


@pytest.mark.parametrize("partition_items, conversation_messages", [
    (
            [
                ContextPartitionItem(role="user", content="Hi"),
                ContextPartitionItem(role="assistant", content="How can I help you"),
            ],
            [
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="How can I help you"),
            ],
    ),
])
def test_sync_context_window_exception(partition_items, conversation_messages):
    with pytest.raises(ConversationMatchesPartitionError, match="Conversation does not alter partition state"):
        (ContextPartition(conversation_uuid="test", items=partition_items)
            .sync_from_conversation(ChatConversation(conversation_uuid="abc", messages=conversation_messages)))


def test_sync_context_window_with_raw_message_start_index():
    context_partition = ContextPartition(
        conversation_uuid="test",
        ancestor_summaries=["Earlier summary"],
        raw_message_start_index=2,
        items=[
            ContextPartitionItem(role="user", content="U2"),
            ContextPartitionItem(role="assistant", content="A2"),
        ],
    )

    context_partition.sync_from_conversation(ChatConversation(
        conversation_uuid="abc",
        messages=[
            ChatMessage(role="user", content="U1"),
            ChatMessage(role="assistant", content="A1"),
            ChatMessage(role="user", content="U2"),
            ChatMessage(role="assistant", content="A2"),
            ChatMessage(role="user", content="U3"),
        ],
    ))

    assert context_partition.items == [
        ContextPartitionItem(role="user", content="U2"),
        ContextPartitionItem(role="assistant", content="A2"),
        ContextPartitionItem(role="user", content="U3"),
    ]


def test_hash_helpers_are_role_and_content_sensitive():
    messages = [
        ChatMessage(role="user", content="Same text"),
        ChatMessage(role="assistant", content="Answer"),
    ]
    changed_role = [
        ChatMessage(role="assistant", content="Same text"),
        ChatMessage(role="assistant", content="Answer"),
    ]

    assert compute_boundary_hash(messages) != compute_boundary_hash(changed_role)
    assert compute_tail_hash(messages) == compute_tail_hash([
        ContextPartitionItem(role="user", content="Same text"),
        ContextPartitionItem(role="assistant", content="Answer"),
    ])


def test_to_anthropic_messages_conversion():
    context_partition = ContextPartition(conversation_uuid="test", items=[
        ContextPartitionItem(role="system", content="Be concise"),
        ContextPartitionItem(role="user", content="Who is Ada?"),
        ContextPartitionItem(role="assistant", content="Checking"),
        ContextPartitionItem(
            call_id="call_1", id="call_1", name="lookup",
            arguments='{"query":"ada"}', type="function_call",
        ),
        ContextPartitionItem(
            call_id="call_1", output='{"name":"Ada Lovelace"}',
            type="function_call_output",
        ),
    ])

    system, messages = context_partition.to_anthropic_messages()

    assert system == "Be concise"
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "Who is Ada?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking"},
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"query": "ada"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"name":"Ada Lovelace"}'}],
        },
    ]


def test_to_anthropic_messages_prepends_ancestor_summaries():
    context_partition = ContextPartition(conversation_uuid="test", ancestor_summaries=[
        "Summary 1",
        "Summary 2",
    ], items=[
        ContextPartitionItem(role="system", content="Be concise"),
        ContextPartitionItem(role="user", content="Hello"),
    ])

    system, messages = context_partition.to_anthropic_messages()

    assert system.startswith("Summary 1")
    assert "Summary 2" in system
    assert "Be concise" in system
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ]


def test_partition_uuid_round_trip():
    partition = ContextPartition(
        conversation_uuid="test",
        items=[ContextPartitionItem(role="user", content="Hello")],
    )

    restored = ContextPartition.model_validate_json(partition.model_dump_json())

    assert restored.partition_uuid == partition.partition_uuid
