"""Call-local, repeatable lazy JSON sequences; no matrix or encoded cache."""
from itertools import zip_longest


class LazyList:
    def __init__(self, factory):
        self.factory = factory

    def __iter__(self):
        return iter(self.factory())

    def __eq__(self, other):
        if not isinstance(other, (LazyList, list, tuple)):
            return False
        missing = object()
        return all(a == b for a, b in zip_longest(self, other, fillvalue=missing))


def array_values(array):
    """Match ndarray.tolist scalar conversion and logical C ordering without it."""
    if array.ndim == 0:
        return array.item()
    return LazyList(lambda: (array_values(array[i]) for i in range(array.shape[0])))
