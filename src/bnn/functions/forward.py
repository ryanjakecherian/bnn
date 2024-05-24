import abc

import torch

from . import functions


class ForwardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor: ...


class MatMultSign(ForwardFunc):
    def __call__(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        integer = x @ W
        out_binary = functions.binarise(x=integer, threshold=0)

        return out_binary
