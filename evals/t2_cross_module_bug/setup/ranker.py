from config import TOP_N
from scorer import score


def rank(items):
    scored = [(item, score(item)) for item in items]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:TOP_N]
