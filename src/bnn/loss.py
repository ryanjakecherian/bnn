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
        # HACK overflow?
        loss = torch.sum(incorrect)

        return loss

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = output - target
        return bnn.functions.ternarise(error, threshold_lo=0, threshold_hi=1)


class CrossEntropyLoss(LossFunction):
    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> int:
        # assume out is logits
        neg_log_softmax = -torch.nn.LogSoftmax(dim=-1)(output.to(float))
        loss = torch.mean(neg_log_softmax[target])
        return loss

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scaled_target = target.to(torch.int) * 2 - 1
        return torch.sign(scaled_target - output)
