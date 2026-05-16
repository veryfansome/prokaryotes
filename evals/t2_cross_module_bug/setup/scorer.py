from config import WEIGHT_A, WEIGHT_B


def score(item):
    return WEIGHT_A * item['feature_a'] + WEIGHT_B * item['feature_b']
