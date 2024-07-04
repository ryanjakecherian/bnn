import abc

import torch

import bnn.type

from . import functions

__all__ = [
    'ForwardFunc',
    'MatMulBinarise',
    'SignBinarise',
    'LayerMeanBinarise',
    'LayerMedianBinarise',
    'OneHot',
]


class ForwardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor: ...


class MatMulBinarise(ForwardFunc):
    def __call__(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # TODO - make this configurable?
        integer = functions.int_matmul(x, W)
        out_binary = self.binarise(x=integer)

        return out_binary

    @abc.abstractmethod
    def binarise(self, x: torch.Tensor) -> torch.Tensor: ...


class SignBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        return functions.binarise(x=x, threshold=0)


class LayerMeanBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE means over layer dimension - samples stay indepenent :)
        means = torch.mean(x.to(torch.float), dim=-1)

        out = torch.ones_like(x, dtype=bnn.type.INTEGER)
        out[x < means[..., None]] = -1

        return out


class LayerMedianBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE median over layer dimension - samples stay indepenent :)
        medians_out = torch.median(x, dim=-1)
        medians = medians_out.values

        out = torch.ones_like(x, dtype=bnn.type.INTEGER)
        out[x < medians[..., None]] = -1

        return out


class MatMulMax(ForwardFunc):
    def __call__(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        integer = functions.int_matmul(x, W)
        out_binary = self.binary_max(x=integer)
        return out_binary

    @abc.abstractmethod
    def binary_max(self, x: torch.Tensor) -> torch.Tensor: ...


class OneHot(MatMulMax):
    def binary_max(self, x: torch.Tensor) -> torch.Tensor:
        return functions.one_hot_argmax(x)


class BitCountMax(MatMulMax):
    out_dims: int
    extra_dims: int

    def __init__(self, out_dims: int, extra_dims: int):
        self.out_dims = out_dims
        self.extra_dims = extra_dims

    def binary_max(self, x: torch.Tensor) -> torch.Tensor:
        # reshape and binarise
        reshaped = x.reshape(-1, self.extra_dims, self.out_dims)
        binary_reshaped = functions.binarise(reshaped)

        # count bits and argmax
        bitcounts = torch.sum(binary_reshaped, dim=-2)
        one_hot = functions.one_hot_argmax(bitcounts)

        return one_hot
