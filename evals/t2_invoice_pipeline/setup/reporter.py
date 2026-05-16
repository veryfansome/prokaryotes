import json

totals = json.load(open('totals.json'))
winner = max(totals, key=totals.get)
with open('report.txt', 'w') as f:
    f.write(f'TOP_CUSTOMER: {winner}\n')
    for customer, total in sorted(totals.items(), key=lambda x: -x[1]):
        f.write(f'  {customer}: {total:.2f}\n')
