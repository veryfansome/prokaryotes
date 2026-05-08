import pytest

from prokaryotes.utils_v1.text_utils import normalize_text_for_identity


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("parent-child reading", "parent-child reading"),
        ("parent‐child reading", "parent-child reading"),
        ("parent‑child reading", "parent-child reading"),
        ("parent‒child reading", "parent-child reading"),
        ("parent–child reading", "parent-child reading"),
        ("parent—child reading", "parent-child reading"),
        ("parent−child reading", "parent-child reading"),
        ("parent\u00A0child reading", "parent child reading"),
        ("parent\u2007child reading", "parent child reading"),
        ("parent\u202Fchild reading", "parent child reading"),
        ("pa\u200Brent reading", "parent reading"),
        ("pa\u200Crent reading", "parent reading"),
        ("pa\u200Drent reading", "parent reading"),
        ("pa\u2060rent reading", "parent reading"),
        ("\uFEFFparent reading", "parent reading"),
        ("co\u00ADoperate reading", "cooperate reading"),
    ],
)
def test_normalize_text_for_identity_with_common_variants(input_text: str, expected: str):
    assert normalize_text_for_identity(input_text) == expected
