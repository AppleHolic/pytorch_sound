from torch._six import container_abcs
from itertools import repeat


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


single = _ntuple(1)
double = _ntuple(2)
