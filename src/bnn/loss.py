import abc

import torch

import bnn.functions


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
        loss = incorrect.sum()

        return loss

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return bnn.functions.ternarise(target - output, threshold_lo=0, threshold_hi=1)
