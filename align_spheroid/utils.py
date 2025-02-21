import itertools
from itertools import chain, repeat

def perms(x):
    """Python equivalent of MATLAB perms."""
    return np.vstack(list(itertools.permutations(x)))[::-1]


def replicate_items(lst, n):
    return list(chain.from_iterable(repeat(item, n) for item in lst))