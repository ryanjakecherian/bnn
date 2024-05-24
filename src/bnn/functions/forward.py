import abc

import torch


class ForwardsFunc(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def __call__(ctx, x: torch.Tensor) -> torch.Tensor: ...
