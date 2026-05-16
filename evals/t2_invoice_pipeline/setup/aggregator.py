import csv
import json
from collections import defaultdict

from converter import to_usd
from discounts import apply_discount

totals = defaultdict(float)
with open('invoices.csv') as f:
    for row in csv.DictReader(f):
        usd = to_usd(float(row['amount']), row['currency'])
        net = apply_discount(usd, row['discount_code'])
        totals[row['customer']] += net

json.dump(dict(totals), open('totals.json', 'w'), indent=2)
