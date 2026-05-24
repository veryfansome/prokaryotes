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
        ("parent\u00a0child reading", "parent child reading"),
        ("parent\u2007child reading", "parent child reading"),
        ("parent\u202fchild reading", "parent child reading"),
        ("pa\u200brent reading", "parent reading"),
        ("pa\u200crent reading", "parent reading"),
        ("pa\u200drent reading", "parent reading"),
        ("pa\u2060rent reading", "parent reading"),
        ("\ufeffparent reading", "parent reading"),
        ("co\u00adoperate reading", "cooperate reading"),
    ],
)
def test_normalize_text_for_identity_with_common_variants(input_text: str, expected: str):
    assert normalize_text_for_identity(input_text) == expected
