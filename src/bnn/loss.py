import abc

import torch

import bnn.functions

__all___ = [
    'LossFunction',
    'l1',
]


class LossFunction(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> float: ...


class l1(LossFunction):
    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> int:
        incorrect = torch.abs(target - output)
        # TODO overflow?
        loss = incorrect.sum()

        return loss

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = output - target
        return bnn.functions.ternarise(error, threshold_lo=0, threshold_hi=1)


class CrossEntropyLoss(LossFunction):
    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> int:
        return NotImplemented

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return NotImplemented
