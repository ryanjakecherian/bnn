import abc

import torch

from . import functions


class ForwardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor: ...


class MatMulBinarise(ForwardFunc):
    def __call__(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        integer = x @ W
        out_binary = self.binarise(x=integer)

        return out_binary

    @abc.abstractmethod
    def binarise(self, x: torch.Tensor) -> torch.Tensor: ...


class SignBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        return functions.binarise(x=x, threshold=0)


class LayerMeanBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        means = torch.mean(x.to(torch.float), dim=-1)

        out = torch.empty_like(x)

        for i, (x_, mean) in enumerate(zip(x, means)):
            out[i] = functions.binarise(x=x_, threshold=mean)

        return out


class LayerMedianBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        medians_out = torch.median(x, dim=-1)
        medians = medians_out.values

        out = torch.empty_like(x)

        for i, (x_, median) in enumerate(zip(x, medians)):
            out[i] = functions.binarise(x=x_, threshold=median)

        return out
