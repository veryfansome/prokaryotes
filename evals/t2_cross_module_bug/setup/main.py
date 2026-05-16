import json

from ranker import rank

items = json.load(open('items.json'))
for i, (item, s) in enumerate(rank(items), 1):
    print(f'{i}. {item["name"]} (score: {s:.2f})')
