import abc

import torch

from . import functions

__all__ = [
    'BackwardFunc',
    'BackprojectTernarise',
    'SignTernarise',
    'LayerMeanStdTernarise',
    'LayerQuantileTernarise',
]

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
        W_grad = torch.einsum(
            '...j,...k->jk',
            input.to(functions.TORCH_FLOAT_TYPE),
            grad.to(functions.TORCH_FLOAT_TYPE),
        ).to(W.dtype)

        W_grad_int = W_grad.to(torch.int)

        # TODO - make this configurable?
        grad = functions.int_matmul(grad, W.T)
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
        # NOTE done over layer dimension - samples stay indepenent :)
        stds, means = torch.std_mean(grad.to(torch.float), dim=-1)

        out = torch.zeros_like(grad, dtype=torch.int)

        # calculate thresholds
        threshold_hi = torch.clamp_min(means + stds * self.half_range_stds, 0)
        threshold_lo = torch.clamp_max(means - stds * self.half_range_stds, 0)

        # apply
        out[grad > threshold_hi[..., None]] = 1
        out[grad < threshold_lo[..., None]] = -1

        return out


class LayerQuantileTernarise(BackprojectTernarise):
    lo_hi_quant: torch.Tensor

    def __init__(self, lo: float = 0.3, hi: float = 0.7):
        self.lo_hi_quant = torch.Tensor([lo, hi])

    def to(self, device: torch.device):
        self.lo_hi_quant = self.lo_hi_quant.to(device)

    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        # NOTE done over layer dimension - samples stay indepenent :)
        try:
            lo_quants, hi_quants = torch.quantile(
                input=grad.to(torch.float),
                q=self.lo_hi_quant,
                dim=-1,
            )
        except RuntimeError:
            self.to(torch.get_device(grad))
            lo_quants, hi_quants = torch.quantile(
                input=grad.to(torch.float),
                q=self.lo_hi_quant,
                dim=-1,
            )

        lo_quants = torch.clamp_max(lo_quants, max=0)
        hi_quants = torch.clamp_min(hi_quants, min=0)

        out = torch.zeros_like(grad, dtype=torch.int)

        # apply
        out[grad > hi_quants[..., None]] = 1
        out[grad < lo_quants[..., None]] = -1

        return out


class LayerQuantileSymmetricTernarise(BackprojectTernarise):
    prop_zero: float

    def __init__(self, prop_zero: float = 1 / 3):
        self.prop_zero = prop_zero

    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        abs_grad = torch.abs(grad)

        # NOTE done over layer dimension - samples stay indepenent :)
        quants = torch.quantile(
            input=abs_grad.to(torch.float),
            q=self.prop_zero,
            dim=-1,
        )

        out = torch.sign(grad)
        out[abs_grad < quants[..., None]] = 0

        return out
