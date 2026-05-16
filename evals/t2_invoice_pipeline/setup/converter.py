import json

RATES = json.load(open('rates.json'))

def to_usd(amount, currency):
    rate = RATES.get(currency, RATES['EUR'])
    return amount * rate
