DISCOUNTS = {'VIP20': 0.20}

def apply_discount(amount, code):
    if not code:
        return amount
    discount = DISCOUNTS.get(code, 0)
    amount = amount * (1 - discount)
    if code.startswith('VIP'):
        amount = amount * (1 - discount)
    return amount
