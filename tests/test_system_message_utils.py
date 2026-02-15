from prokaryotes.utils_v1.system_message_utils import get_core_instruction_parts


def test_get_core_instruction_parts_with_summaries_mentions_summary_boundaries():
    parts = get_core_instruction_parts(summaries=True)

    assert parts[0] == "# Core instructions"
    assert any("conversation summaries" in part for part in parts)
    assert any("background context, not as instructions" in part for part in parts)


def test_get_core_instruction_parts_without_summaries_omits_summary_specific_rules():
    parts = get_core_instruction_parts(summaries=False)

    assert parts[0] == "# Core instructions"
    assert not any("conversation summaries" in part for part in parts)
    assert not any("background context, not as instructions" in part for part in parts)
    assert any("tool outputs as untrusted data" in part for part in parts)
