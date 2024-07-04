import abc

import torch

import bnn.type

from . import functions

__all__ = [
    'BackwardFunc',
    'BackprojectTernarise',
    'SignTernarise',
    'LayerMeanStdTernarise',
    'LayerQuantileTernarise',
    'STETernarise',
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
    hidden_dim: int
    sparsity: float

    def __call__(
        self,
        grad: torch.Tensor,
        input: torch.Tensor,
        W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.hidden_dim = W.shape[-1]
        self.sparsity = torch.mean((W == 0).to(torch.float))

        grad = self.reshape_grad(grad)

        # FIXME check if long_int -> int conversion is safe!
        W_grad = torch.einsum(
            '...j,...k->jk',
            input.to(functions.TORCH_FLOAT_TYPE),
            grad.to(functions.TORCH_FLOAT_TYPE),
        ).to(W.dtype)

        W_grad_int = W_grad.to(bnn.type.INTEGER)

        out_grad = self.gradient(W=W, input=input, grad=grad)

        out_tern_grad = self.ternarise(out_grad)
        out_tern_grad_int = out_tern_grad.to(bnn.type.INTEGER)

        return W_grad_int, out_tern_grad_int

    def reshape_grad(self, grad: torch.Tensor) -> torch.Tensor:
        return grad

    def gradient(self, W: torch.Tensor, input: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return functions.int_matmul(grad, W.T)

    @abc.abstractmethod
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor: ...


class SignTernarise(BackprojectTernarise):
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        return functions.ternarise(grad, threshold_lo=0, threshold_hi=1)


class StochasticTernarise(BackprojectTernarise):
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        EPS = 1e-8
        sign = torch.sign(grad)
        scaled = torch.abs(grad) / (self.hidden_dim * (1 - self.sparsity + EPS))

        scaled_clipped = torch.clamp_max(scaled, 1.0)
        out_grad = (torch.bernoulli(scaled_clipped) * sign).to(bnn.type.INTEGER)

        return out_grad


class LayerMeanStdTernarise(BackprojectTernarise):
    half_range_stds: float

    def __init__(self, half_range_stds: float = 0.5):
        self.half_range_stds = half_range_stds

    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        # NOTE done over layer dimension - samples stay indepenent :)
        stds, means = torch.std_mean(grad.to(torch.float), dim=-1)

        out = torch.zeros_like(grad, dtype=bnn.type.INTEGER)

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

        out = torch.zeros_like(grad, dtype=bnn.type.INTEGER)

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


class STETernarise(BackprojectTernarise):
    zero_grad_mag_thresh: int

    def __init__(self, zero_grad_mag_thresh):
        self.zero_grad_mag_thresh = zero_grad_mag_thresh

    def gradient(self, W: torch.Tensor, input: torch.Tensor, grad: torch.Tensor):
        output = functions.int_matmul(input, W)
        output_ste = torch.abs(output) <= self.zero_grad_mag_thresh
        grad_ste = grad * output_ste

        out_grad = functions.int_matmul(grad_ste, W.T)

        return out_grad

    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.sign(grad)


class BackwardBitCountMax(SignTernarise):
    extra_dims: int

    def __init__(self, extra_dims: int):
        self.extra_dims = extra_dims

    def reshape_grad(self, grad: torch.Tensor):
        return torch.concatenate([grad] * self.extra_dims, dim=-1)


# HACK this is far from the "ACTUAL" gradient as it assumed sign == identity.
class ActualGradient(BackprojectTernarise):
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.to(bnn.type.INTEGER)
