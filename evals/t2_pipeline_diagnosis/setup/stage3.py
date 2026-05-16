import json

result = json.load(open('stage2_out.json'))
with open('report.txt', 'w') as f:
    f.write(f"WINNER: {result['winner']}\n")
    for cat, score in sorted(result['scores'].items(), key=lambda x: -x[1]):
        f.write(f'  {cat}: {score}\n')
