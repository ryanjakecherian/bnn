from typing import Any

__all__ = [
    'sandwich_list',
]


def sandwich_list(a: Any, l: list, b: Any) -> list:
    return [a] + l + [b]


def pow(a: float, b: float) -> float:
    return a**b
