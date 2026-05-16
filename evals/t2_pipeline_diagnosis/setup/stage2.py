import json

totals = json.load(open('stage1_out.json'))
# finds the category with the highest aggregated total
winner = max(totals, key=totals.get)
json.dump({'winner': winner, 'scores': totals}, open('stage2_out.json', 'w'), indent=2)
