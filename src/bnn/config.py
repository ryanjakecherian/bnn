from typing import Any, Callable


def sandwich_list(a: Any, l: list, b: Any) -> list:
    return [a] + l + [b]


def pow(a: float, b: float) -> float:
    return a**b


def lambda_const(a: float) -> Callable[[Any], float]:
    def myfunc(*args, **kwargs):
        return a

    return myfunc


def none(*args, **kwargs) -> None:
    return None
