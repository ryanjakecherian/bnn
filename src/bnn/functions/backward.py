import abc

import torch

from . import functions


class BackwardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        grad: torch.Tensor,
        input: torch.Tensor,
        W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class BackprojectTernarise(BackwardFunc):
    def __call__(
        self,
        grad: torch.Tensor,
        input: torch.Tensor,
        W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # FIXME check if long_int -> int conversion is safe!
        W_grad = grad.unsqueeze(-2) * input.unsqueeze(-1)

        while W_grad.dim() > 2:
            W_grad = W_grad.sum(0)

        # TODO should this be ternarised?
        W_grad_int = W_grad.to(torch.int)

        grad = grad @ W.T
        # TODO pick this threshold nicely... adaptively?
        # TODO implenent layer-normy type of thing...
        grad = self.ternarise(grad)
        grad_int = grad.to(torch.int)

        return W_grad_int, grad_int

    @abc.abstractmethod
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor: ...


class SignTernarise(BackwardFunc):
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        return functions.ternarise(grad, threshold_lo=0, threshold_hi=1)
