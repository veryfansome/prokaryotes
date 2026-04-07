from prokaryotes.utils_v1.text_utils import normalize_text_for_identity


def test_normalize_text_for_identity_normalizes_dash_variants():
    assert normalize_text_for_identity("parent-child reading") == "parent-child reading"
    assert normalize_text_for_identity("parent–child reading") == "parent-child reading"
