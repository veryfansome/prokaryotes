import csv
import json
from collections import defaultdict

totals = defaultdict(int)
with open('records.csv') as f:
    for row in csv.DictReader(f):
        totals[row['category']] = max(totals[row['category']], int(row['amount']))

with open('stage1_out.json', 'w') as f:
    json.dump(dict(totals), f, indent=2)
