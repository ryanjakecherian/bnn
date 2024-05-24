import abc

import torch


class BackwardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        grad: torch.Tensor,
        x: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor: ...
