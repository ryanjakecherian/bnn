import abc

import torch

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
        error = torch.abs(output - target)
        loss = torch.sum(error)

        if error.ndim > 1:
            loss = loss / len(error)

        return loss

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sign(output - target)


class l2(LossFunction):
    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> int:
        error = torch.square(output - target)
        loss = torch.sqrt(torch.sum(error))

        if error.ndim > 1:
            loss = loss / len(error)

        return loss

    # TODO implement me
    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sign(output - target)


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
