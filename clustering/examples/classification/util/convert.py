from typing import Iterable


def as_list(iterable: Iterable) -> list:
    if isinstance(iterable, list):
        return iterable
    return list(iterable)
