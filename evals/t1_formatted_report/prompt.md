Read `scores.csv` and `config.json`, then write `report.txt` conforming exactly to the following spec:

1. Line 1: `REPORT: {date}` where `date` comes from `config.json`.
2. Line 2: a separator — `{separator}` repeated `{width}` times (both from config).
3. One line per **ranked** entry (score ≥ `min_score` from config), format: `  {rank}. {name} ({category}) — {score:.1f}` (two leading spaces, em-dash —).
4. Entries sorted by score descending; ties broken alphabetically by name.
5. **Competition ranking**: tied entries share a rank and the next rank skips (e.g. two entries tied at rank 2 means the next rank is 4, not 3).
6. The same separator line again after the last entry.
7. `Total entries: {N}` — count of **all** rows in the CSV including below-threshold.
8. `Ranked entries: {M}` — count of entries meeting the threshold.
9. `Average score: {X.X}` — mean of **ranked** entries only, rounded to 1 decimal.
10. `Top category: {name}` — category with the highest **sum** of scores among ranked entries; ties broken alphabetically.
11. One line per category that has ranked entries, alphabetically: `  {category}: {N} entries` (two leading spaces).
