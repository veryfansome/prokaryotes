Implement `match(pattern: str, text: str) -> bool` in `solution.py`.

The pattern language supports three special characters:
- `?` matches exactly one character
- `*` matches any sequence of zero or more characters
- `+` matches any sequence of one or more characters
- All other characters match themselves exactly

The pattern must match the **entire** text (not a substring).

Examples:
- `match('c?t', 'cat')` → `True`
- `match('f*d', 'food')` → `True`
- `match('ab*cd', 'abcd')` → `True` (`*` can match empty)
- `match('go+d', 'good')` → `True`
- `match('go+d', 'god')` → `False` (`+` requires at least one character)
