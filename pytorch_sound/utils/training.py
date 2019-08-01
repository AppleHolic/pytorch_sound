import inspect
import torch
from typing import Dict, Any


def parse_model_kwargs(model_cls: torch.nn.Module, **kwargs) -> Dict[str, Any]:
    """
    Parse matched arguments in given class
    :param model_cls: module class
    :param kwargs: raw arguments
    :return: parsed(filtered) arguments
    """
    kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(model_cls).args}
    return kwargs
