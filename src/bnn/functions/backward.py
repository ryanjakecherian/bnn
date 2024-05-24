import abc

import torch

from . import functions

EPS = 0.01


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
        tern_grad = self.ternarise(grad)
        tern_grad_int = tern_grad.to(torch.int)

        return W_grad_int, tern_grad_int

    @abc.abstractmethod
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor: ...


class SignTernarise(BackprojectTernarise):
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        return functions.ternarise(grad, threshold_lo=0, threshold_hi=1)


class LayerMeanStdTernarise(BackprojectTernarise):
    half_range_stds: float

    def __init__(self, half_range_stds: float = 0.5):
        self.half_range_stds = half_range_stds

    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        stds, means = torch.std_mean(grad.to(torch.float), dim=-1)

        out = torch.empty_like(grad)

        for i, (grad_, std, mean) in enumerate(zip(grad, stds, means)):
            out[i] = functions.ternarise(
                x=grad_,
                threshold_lo=mean - std * self.half_range_stds,
                # NOTE threshold_hi is inclusive, ie ternarise(thresh_hi) = 1
                # therefore, adding EPS ensures grad=zeros is stable under ternarisation
                threshold_hi=mean + std * self.half_range_stds + EPS,
            )

        return out


class LayerQuantileTernarise(BackprojectTernarise):
    lo_hi_quant: torch.Tensor

    def __init__(self, lo: float = 0.3, hi: float = 0.7):
        self.lo_hi_quant = torch.Tensor([lo, hi])

    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        lo_quants, hi_quants = torch.quantile(
            input=grad.to(torch.float),
            q=self.lo_hi_quant,
            dim=-1,
        )

        out = torch.empty_like(grad)

        for i, (grad_, lo_q, hi_q) in enumerate(zip(grad, lo_quants, hi_quants)):
            out[i] = functions.ternarise(
                x=grad_,
                threshold_lo=lo_q,
                # NOTE threshold_hi is inclusive, ie ternarise(thresh_hi) = 1
                # therefore, adding EPS ensures grad=zeros is stable under ternarisation
                threshold_hi=hi_q + EPS,
            )

        return out
