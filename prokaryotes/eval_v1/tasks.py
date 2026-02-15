from prokaryotes.eval_v1.models import EvalTask

TASKS: list[EvalTask] = [
    # Tier 1: Should be trivial for frontier models
    EvalTask(
        id="t1_create_file",
        tier=1,
        description="Create a file with specific content",
        prompt="Create a file named `result.txt` in the current directory containing exactly the text `42` (one line).",
        check_command='grep -qx "42" result.txt',
    ),
    EvalTask(
        id="t1_count_files",
        tier=1,
        description="Count files of a specific type and write the count",
        prompt=(
            "Count the number of `.py` files directly inside the `src/` directory "
            "(non-recursive) and write the count as a single integer to `count.txt`."
        ),
        setup_command="mkdir -p src && touch src/a.py src/b.py src/c.py src/d.py src/readme.txt",
        check_command='grep -qx "4" count.txt',
    ),
    EvalTask(
        id="t1_find_nested",
        tier=1,
        description="Find a file in a directory tree and copy its contents",
        prompt=(
            "Find the file named `hidden.txt` somewhere in the directory tree and write its contents to `output.txt`."
        ),
        setup_command="mkdir -p a/b/c && printf 'treasure' > a/b/c/hidden.txt",
        check_command='grep -qx "treasure" output.txt',
    ),
    EvalTask(
        id="t1_filter_lines",
        tier=1,
        description="Extract lines matching a prefix",
        prompt=(
            "Extract all lines from `log.txt` that start with `ERROR` and write them "
            "to `errors.txt`. Preserve original order."
        ),
        setup_files={
            "log.txt": "INFO starting\nERROR disk full\nINFO retry\nERROR timeout\nINFO done\n",
        },
        check_command=(
            'grep -Fx "ERROR disk full" errors.txt && '
            'grep -Fx "ERROR timeout" errors.txt && '
            'test "$(grep -c "." errors.txt)" = "2"'
        ),
    ),
    EvalTask(
        id="t1_sum_numbers",
        tier=1,
        description="Sum a list of integers from a file",
        prompt="Sum all integers in `numbers.txt` (one per line) and write the result to `sum.txt`.",
        setup_files={
            "numbers.txt": "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
        },
        check_command='grep -qx "55" sum.txt',
    ),
    EvalTask(
        id="t1_fix_off_by_one",
        tier=1,
        description="Fix an off-by-one bug in a Python script",
        prompt=(
            "The script `compute.py` is supposed to print the sum of integers from 1 to 10 (which is 55), "
            "but it contains a bug. Fix it so that `python compute.py` prints exactly `55`."
        ),
        setup_files={
            "compute.py": ("total = 0\nfor i in range(1, 10):\n    total += i\nprint(total)\n"),
        },
        check_command="python compute.py | grep -qx '55'",
    ),
    EvalTask(
        id="t1_sort_by_length",
        tier=1,
        description="Sort lines by character length",
        prompt=(
            "Sort the lines in `words.txt` by length (shortest first) and write the result "
            "to `sorted.txt`. Lines of equal length may appear in any order."
        ),
        setup_files={
            "words.txt": "elephant\ncat\nbee\nalligator\ndog\n",
        },
        check_command=(
            'python3 -c "'
            "lines=open('sorted.txt').read().splitlines(); "
            "assert all(len(lines[i])<=len(lines[i+1]) for i in range(len(lines)-1)), "
            "f'not sorted by length: {lines}'\""
        ),
    ),
    EvalTask(
        id="t1_largest_file",
        tier=1,
        description="Find the largest file in a directory tree",
        prompt=(
            "Find the largest file (by byte size) anywhere in the directory tree "
            "and write its basename to `largest.txt`."
        ),
        setup_command=(
            "mkdir -p x/y && "
            "python3 -c \"open('x/alpha.bin','wb').write(b'a'*100)\" && "
            "python3 -c \"open('x/y/beta.bin','wb').write(b'a'*10000)\" && "
            "python3 -c \"open('x/gamma.bin','wb').write(b'a'*500)\""
        ),
        check_command='grep -qx "beta.bin" largest.txt',
    ),
    EvalTask(
        id="t1_csv_sum",
        tier=1,
        description="Sum a column across multiple CSV files",
        prompt=(
            "Each `.csv` file in this directory has a `value` column. "
            "Compute the total sum of all `value` entries across all CSV files and write it to `total.txt`."
        ),
        setup_files={
            "a.csv": "name,value\nalpha,10\nbeta,20\n",
            "b.csv": "name,value\ngamma,30\n",
            "c.csv": "name,value\ndelta,15\nepsilon,25\n",
        },
        check_command='grep -qx "100" total.txt',
    ),
    EvalTask(
        id="t1_run_and_capture",
        tier=1,
        description="Run a script and persist its stdout",
        prompt="Run `python gen.py` and write its stdout output to `output.txt`.",
        setup_files={
            "gen.py": 'print("\\n".join(str(i ** 2) for i in range(1, 6)))\n',
        },
        check_command=(
            'python3 -c "'
            "import subprocess; "
            "e=subprocess.check_output('python3 gen.py',shell=True).decode(); "
            "a=open('output.txt').read(); "
            "assert e==a, 'output mismatch'\""
        ),
    ),
    EvalTask(
        id="t1_debug_median",
        tier=1,
        description="Debug a subtle logic error (wrong index)",
        prompt=(
            "The script `median.py` is supposed to print the median of [1,2,3,4,5] "
            "but it prints the wrong answer. Debug it, fix it, and verify it outputs the correct value."
        ),
        setup_files={
            "median.py": (
                "data = [1, 2, 3, 4, 5]\ndata.sort()\nmid = len(data) // 2\nprint(data[mid - 1])\n"
            ),
        },
        check_command="python median.py | grep -qx '3'",
    ),
    EvalTask(
        id="t1_implement_function",
        tier=1,
        description="Implement a function so a test suite passes",
        prompt=(
            "Implement the function `def reverse_words(s: str) -> str` in `solution.py` so that it "
            "returns the words of `s` in reverse order (e.g. `'hello world'` → `'world hello'`). "
            "All tests in `test_solution.py` must pass."
        ),
        setup_files={
            "test_solution.py": (
                "from solution import reverse_words\n\n"
                "def test_basic():\n"
                "    assert reverse_words('hello world') == 'world hello'\n\n"
                "def test_single_word():\n"
                "    assert reverse_words('hello') == 'hello'\n\n"
                "def test_multiple():\n"
                "    assert reverse_words('one two three') == 'three two one'\n"
            ),
        },
        check_command="python -m pytest test_solution.py -q",
    ),
    EvalTask(
        id="t1_iterative_repair",
        tier=1,
        description="Fix a script that has multiple layered errors",
        prompt=(
            "The script `pipeline.py` fails when run. Fix all errors so that "
            "`python pipeline.py` runs successfully and prints `done`."
        ),
        setup_files={
            "pipeline.py": (
                "import json\n\n"
                "def load_data(path):\n"
                "    with open(path) as f:\n"
                "        return json.load(f)\n\n"
                "def process(data):\n"
                "    return [x * 2 for x in data['values']]\n\n"
                "def main():\n"
                "    data = load_data('data.json')\n"
                "    result = process(data)\n"
                "    print('done')\n\n"
                "main()\n"
            ),
        },
        check_command="python pipeline.py | grep -qx 'done'",
    ),
    EvalTask(
        id="t1_benchmark_impls",
        tier=1,
        description="Benchmark two implementations and identify the faster one",
        prompt=(
            "Benchmark `impl_a.py` and `impl_b.py` by running each at least 3 times. "
            "Write the filename of the faster implementation to `faster.txt` (e.g. `impl_a.py`)."
        ),
        setup_files={
            "impl_a.py": (
                "import time\n"
                "start = time.time()\n"
                "result = sum(range(1_000_000))\n"
                "print(f'result={result} elapsed={time.time()-start:.4f}s')\n"
            ),
            "impl_b.py": (
                "import time\n"
                "start = time.time()\n"
                "result = 0\n"
                "for i in range(1_000_000):\n"
                "    result += i\n"
                "print(f'result={result} elapsed={time.time()-start:.4f}s')\n"
            ),
        },
        check_command='grep -qx "impl_a.py" faster.txt',
    ),
    EvalTask(
        id="t1_log_analysis",
        tier=1,
        description="Analyse a log file and produce a structured report",
        prompt=(
            "Analyse `app.log` and write a report to `report.txt`. "
            "Each log line is formatted as `<LEVEL> <message>` (e.g. `ERROR connection refused`). "
            "The report must contain a line formatted exactly as `MOST_COMMON_ERROR: <message>` "
            "where `<message>` is the text after the `ERROR ` prefix "
            "(e.g. `MOST_COMMON_ERROR: disk full`)."
        ),
        setup_files={
            "app.log": (
                "ERROR connection refused\n"
                "INFO request received\n"
                "ERROR timeout\n"
                "ERROR connection refused\n"
                "WARNING slow query\n"
                "ERROR connection refused\n"
                "ERROR timeout\n"
                "INFO shutdown\n"
            ),
        },
        check_command='grep -q "^MOST_COMMON_ERROR: connection refused" report.txt',
    ),
    EvalTask(
        id="t1_formatted_report",
        tier=1,
        description="Generate a formatted report satisfying 12 precise simultaneous constraints",
        prompt=(
            "Read `scores.csv` and `config.json`, then write `report.txt` conforming exactly to "
            "the following spec:\n\n"
            "1. Line 1: `REPORT: {date}` where `date` comes from `config.json`.\n"
            "2. Line 2: a separator — `{separator}` repeated `{width}` times (both from config).\n"
            "3. One line per **ranked** entry (score ≥ `min_score` from config), "
            "format: `  {rank}. {name} ({category}) \u2014 {score:.1f}` (two leading spaces, "
            "em-dash \\u2014).\n"
            "4. Entries sorted by score descending; ties broken alphabetically by name.\n"
            "5. **Competition ranking**: tied entries share a rank and the next rank skips "
            "(e.g. two entries tied at rank 2 means the next rank is 4, not 3).\n"
            "6. The same separator line again after the last entry.\n"
            "7. `Total entries: {N}` — count of **all** rows in the CSV including below-threshold.\n"
            "8. `Ranked entries: {M}` — count of entries meeting the threshold.\n"
            "9. `Average score: {X.X}` — mean of **ranked** entries only, rounded to 1 decimal.\n"
            "10. `Top category: {name}` — category with the highest **sum** of scores among "
            "ranked entries; ties broken alphabetically.\n"
            "11. One line per category that has ranked entries, alphabetically: "
            "`  {category}: {N} entries` (two leading spaces)."
        ),
        setup_files={
            "scores.csv": (
                "name,score,category\n"
                "alice,85,engineering\n"
                "bob,72,design\n"
                "carol,85,engineering\n"
                "dave,55,design\n"
                "eve,90,engineering\n"
                "frank,68,design\n"
                "grace,45,marketing\n"
                "henry,72,engineering\n"
            ),
            "config.json": '{"date": "2024-03-15", "min_score": 60, "separator": "=", "width": 40}\n',
        },
        check_command=(
            "python3 << 'PYEOF'\n"
            "lines = open('report.txt').read().splitlines()\n"
            "assert len(lines) == 15, f'expected 15 lines, got {len(lines)}'\n"
            "assert lines[0] == 'REPORT: 2024-03-15'\n"
            "assert lines[1] == '=' * 40\n"
            "assert lines[2] == '  1. eve (engineering) \u2014 90.0'\n"
            "assert lines[3] == '  2. alice (engineering) \u2014 85.0'\n"
            "assert lines[4] == '  2. carol (engineering) \u2014 85.0'\n"
            "assert lines[5] == '  4. bob (design) \u2014 72.0'\n"
            "assert lines[6] == '  4. henry (engineering) \u2014 72.0'\n"
            "assert lines[7] == '  6. frank (design) \u2014 68.0'\n"
            "assert lines[8] == '=' * 40\n"
            "assert lines[9] == 'Total entries: 8'\n"
            "assert lines[10] == 'Ranked entries: 6'\n"
            "assert lines[11] == 'Average score: 78.7'\n"
            "assert lines[12] == 'Top category: engineering'\n"
            "assert lines[13] == '  design: 2 entries'\n"
            "assert lines[14] == '  engineering: 4 entries'\n"
            "PYEOF"
        ),
        timeout_seconds=300,
    ),
    EvalTask(
        id="t1_wildcard_match",
        tier=1,
        description="Implement wildcard pattern matching (?, *, +); naive recursion times out on hidden perf test",
        prompt=(
            "Implement `match(pattern: str, text: str) -> bool` in `solution.py`.\n\n"
            "The pattern language supports three special characters:\n"
            "- `?` matches exactly one character\n"
            "- `*` matches any sequence of zero or more characters\n"
            "- `+` matches any sequence of one or more characters\n"
            "- All other characters match themselves exactly\n\n"
            "The pattern must match the **entire** text (not a substring).\n\n"
            "Examples:\n"
            "- `match('c?t', 'cat')` → `True`\n"
            "- `match('f*d', 'food')` → `True`\n"
            "- `match('ab*cd', 'abcd')` → `True` (`*` can match empty)\n"
            "- `match('go+d', 'good')` → `True`\n"
            "- `match('go+d', 'god')` → `False` (`+` requires at least one character)"
        ),
        setup_files={
            "solution.py": "def match(pattern: str, text: str) -> bool:\n    pass\n",
        },
        check_command=(
            "python3 << 'PYEOF'\n"
            "from solution import match\n"
            "assert match('hello', 'hello') == True\n"
            "assert match('hello', 'world') == False\n"
            "assert match('', '') == True\n"
            "assert match('', 'a') == False\n"
            "assert match('a', '') == False\n"
            "assert match('h?llo', 'hello') == True\n"
            "assert match('h?llo', 'hllo') == False\n"
            "assert match('?', 'a') == True\n"
            "assert match('?', '') == False\n"
            "assert match('?', 'ab') == False\n"
            "assert match('*', '') == True\n"
            "assert match('*', 'abc') == True\n"
            "assert match('a*', 'a') == True\n"
            "assert match('a*b', 'ab') == True\n"
            "assert match('a*b', 'axyzb') == True\n"
            "assert match('a*b', 'a') == False\n"
            "assert match('*b', 'b') == True\n"
            "assert match('+', '') == False\n"
            "assert match('+', 'a') == True\n"
            "assert match('+', 'abc') == True\n"
            "assert match('a+b', 'ab') == False\n"
            "assert match('a+b', 'axb') == True\n"
            "assert match('a+b', 'axyzb') == True\n"
            "assert match('+a+', 'xax') == True\n"
            "assert match('+a+', 'ax') == False\n"
            "assert match('*a*b*', 'xaxbx') == True\n"
            "assert match('*a*b*', 'xbxa') == False\n"
            "assert match('*?*', '') == False\n"
            "assert match('*?*', 'a') == True\n"
            "import time\n"
            "text = 'a' * 25\n"
            "pattern = ('*a') * 12 + '*b'\n"
            "start = time.time()\n"
            "result = match(pattern, text)\n"
            "elapsed = time.time() - start\n"
            "assert result == False\n"
            "assert elapsed < 2.0, f'too slow: {elapsed:.2f}s'\n"
            "PYEOF"
        ),
    ),
    EvalTask(
        id="t1_compact_rle",
        tier=1,
        description=(
            "Implement encode/decode for a custom run-length encoding scheme; correctness tested against hidden cases"
        ),
        prompt=(
            "Implement two functions in `solution.py`:\n\n"
            "- `encode(s: str) -> str`: scan left to right and replace each maximal run of "
            "**3 or more** identical characters with `(N)C`, where `N` is the run length and "
            "`C` is the character. Runs of 1 or 2 identical characters are left as-is.\n"
            "- `decode(s: str) -> str`: reverse the encoding — replace each `(N)C` token "
            "(where `N` is a positive integer and `C` is a single character) with `N` copies "
            "of `C`. All other characters pass through unchanged.\n\n"
            "The input to `encode` contains only lowercase ASCII letters. "
            "The input to `decode` is always a valid encoded string.\n\n"
            "Examples:\n"
            "- `encode('bbbcd')` → `'(3)bcd'`\n"
            "- `encode('xxyyyzz')` → `'xx(3)yzz'`\n"
            "- `decode('(4)x')` → `'xxxx'`\n"
            "- `decode('ab(3)cd')` → `'abcccd'`"
        ),
        setup_files={
            "solution.py": (
                "def encode(s: str) -> str:\n"
                "    pass\n\n\n"
                "def decode(s: str) -> str:\n"
                "    pass\n"
            ),
        },
        check_command=(
            "python3 << 'PYEOF'\n"
            "from solution import encode, decode\n"
            "assert encode('aaabbc') == '(3)abbc'\n"
            "assert encode('aab') == 'aab'\n"
            "assert encode('aa') == 'aa'\n"
            "assert encode('a' * 12) == '(12)a'\n"
            "assert encode('aabbbcccc') == 'aa(3)b(4)c'\n"
            "assert encode('') == ''\n"
            "assert encode('a') == 'a'\n"
            "assert encode('abcde') == 'abcde'\n"
            "assert decode('(3)abbc') == 'aaabbc'\n"
            "assert decode('aa(3)b(4)c') == 'aabbbcccc'\n"
            "assert decode('(12)a') == 'a' * 12\n"
            "assert decode('') == ''\n"
            "assert decode('a') == 'a'\n"
            "for s in ['hello', 'aaaaabbbcc', 'abcde', 'aaabbbccc', 'xaaay', 'z', 'aabbcc']:\n"
            "    assert decode(encode(s)) == s\n"
            "PYEOF"
        ),
    ),

    # Tier 2: Debugging, multi-file exploration, triggers think, more tool calls, etc.
    EvalTask(
        id="t2_cross_module_bug",
        tier=2,
        description="Diagnose and fix a cross-module bug in a multi-file scoring pipeline",
        prompt=(
            "The scoring pipeline in this project produces incorrect rankings. "
            "Diagnose the root cause and fix it so that `python main.py` outputs the correct ranking."
        ),
        setup_files={
            "items.json": (
                '[{"name":"alpha","feature_a":8,"feature_b":2},'
                '{"name":"beta","feature_a":5,"feature_b":6},'
                '{"name":"gamma","feature_a":3,"feature_b":9},'
                '{"name":"delta","feature_a":6,"feature_b":8}]'
            ),
            "config.py": (
                "# Feature weights\n"
                "WEIGHT_A = 0.7\n"
                "WEIGHT_B = 0.1\n"
                "TOP_N = 3\n"
            ),
            "scorer.py": (
                "from config import WEIGHT_A, WEIGHT_B\n\n"
                "def score(item):\n"
                "    return WEIGHT_A * item['feature_a'] + WEIGHT_B * item['feature_b']\n"
            ),
            "ranker.py": (
                "from config import TOP_N\n"
                "from scorer import score\n\n"
                "def rank(items):\n"
                "    scored = [(item, score(item)) for item in items]\n"
                "    scored.sort(key=lambda x: x[1], reverse=True)\n"
                "    return scored[:TOP_N]\n"
            ),
            "main.py": (
                "import json\n"
                "from ranker import rank\n\n"
                "items = json.load(open('items.json'))\n"
                "for i, (item, s) in enumerate(rank(items), 1):\n"
                "    print(f'{i}. {item[\"name\"]} (score: {s:.2f})')\n"
            ),
        },
        check_command="python main.py | head -1 | grep -q '1. delta'",
        timeout_seconds=300,
    ),
    EvalTask(
        id="t2_pipeline_diagnosis",
        tier=2,
        description="Diagnose a root-cause bug in a multi-stage pipeline where intermediate state misleads",
        prompt=(
            "Run the pipeline with `bash run_pipeline.sh` and inspect `report.txt`. "
            "The results are incorrect. Diagnose the root cause, fix it, and verify the pipeline "
            "produces the correct output."
        ),
        setup_files={
            "records.csv": (
                "id,category,amount\n"
                "1,electronics,10\n"
                "2,electronics,10\n"
                "3,electronics,10\n"
                "4,electronics,10\n"
                "5,electronics,10\n"
                "6,clothing,45\n"
                "7,books,5\n"
                "8,books,5\n"
                "9,books,5\n"
                "10,books,5\n"
            ),
            "stage1.py": (
                "import csv\n"
                "import json\n"
                "from collections import defaultdict\n\n"
                "totals = defaultdict(int)\n"
                "with open('records.csv') as f:\n"
                "    for row in csv.DictReader(f):\n"
                "        totals[row['category']] = max(totals[row['category']], int(row['amount']))\n\n"
                "with open('stage1_out.json', 'w') as f:\n"
                "    json.dump(dict(totals), f, indent=2)\n"
            ),
            "stage2.py": (
                "import json\n\n"
                "totals = json.load(open('stage1_out.json'))\n"
                "# finds the category with the highest aggregated total\n"
                "winner = max(totals, key=totals.get)\n"
                "json.dump({'winner': winner, 'scores': totals}, open('stage2_out.json', 'w'), indent=2)\n"
            ),
            "stage3.py": (
                "import json\n\n"
                "result = json.load(open('stage2_out.json'))\n"
                "with open('report.txt', 'w') as f:\n"
                "    f.write(f\"WINNER: {result['winner']}\\n\")\n"
                "    for cat, score in sorted(result['scores'].items(), key=lambda x: -x[1]):\n"
                "        f.write(f'  {cat}: {score}\\n')\n"
            ),
            "run_pipeline.sh": "python stage1.py && python stage2.py && python stage3.py\n",
        },
        check_command="bash run_pipeline.sh && grep -q '^WINNER: electronics' report.txt",
        timeout_seconds=300,
    ),
    EvalTask(
        id="t2_invoice_pipeline",
        tier=2,
        description="Fix two interacting bugs (one in data, one in logic) in an invoice pipeline",
        prompt=(
            "Run `bash run.sh` and inspect `report.txt`. "
            "The pipeline is producing incorrect results. Diagnose and fix it."
        ),
        setup_files={
            "invoices.csv": (
                "id,customer,amount,currency,discount_code\n"
                "1,alice,150,GBP,\n"
                "2,alice,150,GBP,\n"
                "3,bob,200,USD,VIP20\n"
                "4,bob,200,USD,VIP20\n"
                "5,bob,200,USD,VIP20\n"
                "6,carol,200,USD,\n"
                "7,carol,200,USD,\n"
            ),
            "rates.json": '{"USD": 1.0, "EUR": 1.2}\n',
            "converter.py": (
                "import json\n\n"
                "RATES = json.load(open('rates.json'))\n\n"
                "def to_usd(amount, currency):\n"
                "    rate = RATES.get(currency, RATES['EUR'])\n"
                "    return amount * rate\n"
            ),
            "discounts.py": (
                "DISCOUNTS = {'VIP20': 0.20}\n\n"
                "def apply_discount(amount, code):\n"
                "    if not code:\n"
                "        return amount\n"
                "    discount = DISCOUNTS.get(code, 0)\n"
                "    amount = amount * (1 - discount)\n"
                "    if code.startswith('VIP'):\n"
                "        amount = amount * (1 - discount)\n"
                "    return amount\n"
            ),
            "aggregator.py": (
                "import csv\n"
                "import json\n"
                "from collections import defaultdict\n"
                "from converter import to_usd\n"
                "from discounts import apply_discount\n\n"
                "totals = defaultdict(float)\n"
                "with open('invoices.csv') as f:\n"
                "    for row in csv.DictReader(f):\n"
                "        usd = to_usd(float(row['amount']), row['currency'])\n"
                "        net = apply_discount(usd, row['discount_code'])\n"
                "        totals[row['customer']] += net\n\n"
                "json.dump(dict(totals), open('totals.json', 'w'), indent=2)\n"
            ),
            "reporter.py": (
                "import json\n\n"
                "totals = json.load(open('totals.json'))\n"
                "winner = max(totals, key=totals.get)\n"
                "with open('report.txt', 'w') as f:\n"
                "    f.write(f'TOP_CUSTOMER: {winner}\\n')\n"
                "    for customer, total in sorted(totals.items(), key=lambda x: -x[1]):\n"
                "        f.write(f'  {customer}: {total:.2f}\\n')\n"
            ),
            "run.sh": "python aggregator.py && python reporter.py\n",
        },
        check_command="bash run.sh && grep -q '^TOP_CUSTOMER: bob' report.txt",
        timeout_seconds=300,
    ),
]
