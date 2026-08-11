"""无需模型调用、可重复计算的检索与引用指标。"""

from __future__ import annotations


def recall_at_k(expected_ids: list[str], retrieved_ids: list[str], k: int) -> float | None:
    """计算前 K 个结果覆盖的期望证据比例。"""
    if not expected_ids:
        return None
    return round(len(set(expected_ids).intersection(retrieved_ids[:k])) / len(set(expected_ids)), 4)


def reciprocal_rank(expected_ids: list[str], retrieved_ids: list[str]) -> float | None:
    """计算首个相关证据排名的倒数。"""
    if not expected_ids:
        return None
    expected = set(expected_ids)
    for index, item in enumerate(retrieved_ids, start=1):
        if item in expected:
            return round(1 / index, 4)
    return 0.0


def citation_coverage(expected_ids: list[str], citation_ids: list[str]) -> float | None:
    """计算回答引用对期望证据的覆盖比例。"""
    if not expected_ids:
        return None
    expected = set(expected_ids)
    return round(len(expected.intersection(citation_ids)) / len(expected), 4)


def empty_retrieval_rate(retrieval_lists: list[list[str]]) -> float | None:
    """计算没有返回任何证据的样本比例。"""
    if not retrieval_lists:
        return None
    return round(sum(not item for item in retrieval_lists) / len(retrieval_lists), 4)


def mean_metric(values: list[float | None]) -> float | None:
    """忽略不可计算值后求平均，并保持四位小数。"""
    valid_values = [value for value in values if value is not None]
    return round(sum(valid_values) / len(valid_values), 4) if valid_values else None
