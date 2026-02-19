import pytest

import prokaryotes.imap_v1 as imap_v1

@pytest.mark.parametrize("input_tuple, expected_dict", [
    # Standard case: multiple pairs
    (
        (b'CHARSET', b'UTF-8', b'NAME', b'file.txt'),
        {'charset': 'UTF-8', 'name': 'file.txt'}
    ),
    # Edge case: Empty input
    (None, {}),
    ((), {}),
])
def test_parse_params(input_tuple, expected_dict):
    assert imap_v1.parse_params(input_tuple) == expected_dict
