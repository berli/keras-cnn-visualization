from typing import Tuple, Iterable

import numpy as np
import itertools


def pairwise(iterable: Iterable) -> Iterable:
    a = iter(iterable)
    return zip(a, a)


def bbox(a: np.ndarray) -> Tuple[int, ...]:
    N = a.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(a, axis=ax)
        indices = np.where(nonzero)[0]
        if len(indices) > 0:
            # Add first (inclusive) and last (exclusive) element
            min_index, max_index = indices[[0, -1]]
            out.extend([min_index, max_index + 1])
        else:
            # Add empty boundings
            out.extend([0, 0])
    return tuple(out)