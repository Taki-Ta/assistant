import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个非零、同维向量的余弦相似度。"""
    if len(a) != len(b):
        raise ValueError("向量长度不一致，无法计算余弦相似度")

    dot_product = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量无法计算余弦相似度")
    return dot_product / (norm_a * norm_b)
