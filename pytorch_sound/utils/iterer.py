from torch._six import container_abcs


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple([x] * n)
    return parse


def repeat(iterable):
    """
    Infinite loop on iterable object
    :param iterable: iterable object
    """
    while True:
        for x in iterable:
            yield x


single = _ntuple(1)
double = _ntuple(2)
