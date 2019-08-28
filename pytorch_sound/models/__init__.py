# reference on
# https://github.com/pytorch/fairseq/blob/master/fairseq/models/__init__.py
# 
import torch.nn as nn
from typing import Callable, Dict, Any
from pytorch_sound.utils.training import parse_model_kwargs


MODEL_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


def build_model(arch_name: str, extra_kwargs: Dict[str, Any] = None) -> nn.Module:
    """
    Build module using architecture name.
    *notice* Your register functions must be registered earlier than calling this function
    :param arch_name: The architecture name that registered by function.
    :param extra_kwargs: Extra arguments that you want to change
    :return: Module instance
    """
    cls = ARCH_MODEL_REGISTRY[arch_name]
    kwargs = parse_model_kwargs(cls, **ARCH_CONFIG_REGISTRY[arch_name]())
    # extra kwargs fetch
    if extra_kwargs:
        for key, val in extra_kwargs.items():
            if key in kwargs:
                kwargs[key] = val
    return cls(**kwargs)


def register_model(name: str) -> Callable:
    """
    New model types can be added to cached dict with the :func:`register_model`
    function decorator.
    For example::
        @register_model('lstm')
        class LSTM(nn.Module):
            (...)
    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_model_architecture(model_name: str, arch_name: str) -> Callable:
    """
    New model architectures can be added to cached dict with the
    :func:`register_model_architecture` function decorator. After registration,
    model can be initialized its own name.
    For example::
        @register_model_architecture('lstm', 'lstm_arch')
        def lstm_arch():
            return {
              'param1': 1,
              'param2': 2, ...
            }
    The decorated function should must return dictionary that contains parameters of the model.
    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture
    """

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError('Cannot register model architecture for unknown model type ({})'.format(model_name))
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model architecture ({})'.format(arch_name))
        if not callable(fn):
            raise ValueError('Model architecture must be callable ({})'.format(arch_name))
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn
