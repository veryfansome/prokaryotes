import time

from solution import match

assert match("hello", "hello")
assert not match("hello", "world")
assert match("", "")
assert not match("", "a")
assert not match("a", "")
assert match("h?llo", "hello")
assert not match("h?llo", "hllo")
assert match("?", "a")
assert not match("?", "")
assert not match("?", "ab")
assert match("*", "")
assert match("*", "abc")
assert match("a*", "a")
assert match("a*b", "ab")
assert match("a*b", "axyzb")
assert not match("a*b", "a")
assert match("*b", "b")
assert not match("+", "")
assert match("+", "a")
assert match("+", "abc")
assert not match("a+b", "ab")
assert match("a+b", "axb")
assert match("a+b", "axyzb")
assert match("+a+", "xax")
assert not match("+a+", "ax")
assert match("*a*b*", "xaxbx")
assert not match("*a*b*", "xbxa")
assert not match("*?*", "")
assert match("*?*", "a")

text = "a" * 25
pattern = ("*a") * 12 + "*b"
start = time.time()
result = match(pattern, text)
elapsed = time.time() - start
assert not result
assert elapsed < 2.0, f"too slow: {elapsed:.2f}s"
