lines = open("report.txt").read().splitlines()
assert len(lines) == 15, f"expected 15 lines, got {len(lines)}"
assert lines[0] == "REPORT: 2024-03-15"
assert lines[1] == "=" * 40
assert lines[2] == "  1. eve (engineering) — 90.0"
assert lines[3] == "  2. alice (engineering) — 85.0"
assert lines[4] == "  2. carol (engineering) — 85.0"
assert lines[5] == "  4. bob (design) — 72.0"
assert lines[6] == "  4. henry (engineering) — 72.0"
assert lines[7] == "  6. frank (design) — 68.0"
assert lines[8] == "=" * 40
assert lines[9] == "Total entries: 8"
assert lines[10] == "Ranked entries: 6"
assert lines[11] == "Average score: 78.7"
assert lines[12] == "Top category: engineering"
assert lines[13] == "  design: 2 entries"
assert lines[14] == "  engineering: 4 entries"
