from functools import wraps
from typing import Callable
import torch


def cached_property(func: Callable) -> property:
    """Decorator that converts a method into a lazily-evaluated cached property"""
    # Create a secret attribute name for the cached property
    attr_name = "_cached_" + func.__name__

    @property
    @wraps(func)
    def _cached_property(self):
        # If the secret attribute doesn't exist, compute the property and set it
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))

        # Otherwise, return the cached property
        return getattr(self, attr_name)

    return _cached_property


def get_layer_list(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Get "the" list of layers from a model.

    This is operationalized as the unique `nn.ModuleList` that contains
    more than half of all the parameters in the model, if it exists.

    Args:
        model: The model to search.

    Returns:
        The nn.ModuleList.

    Raises:
        ValueError: If no such list exists.
    """
    total_params = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList):
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > total_params / 2:
                return module

    raise ValueError(
        "Could not find suitable `ModuleList`; is this an encoder-decoder model?"
    )
