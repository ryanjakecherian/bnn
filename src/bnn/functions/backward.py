import abc

import torch

import bnn.type
import numpy

from . import functions

__all__ = [
    'BackwardFunc',
    'BackprojectTernarise',
    'SignTernarise',
    'LayerMeanStdTernarise',
    'LayerQuantileTernarise',
    'STETernarise',
    'BackwardBitCountMax',
    'ActualGradient',
    'reluBackward',
    'Modal',
]

EPS = 0.01


class BackwardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        grad: torch.Tensor,
        input: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        threshold: int
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

        #W_grad
        W_grad = torch.einsum(
            '...j,...k->jk',
            input.to(functions.TORCH_FLOAT_TYPE),
            grad.to(functions.TORCH_FLOAT_TYPE),
        ).to(W.dtype)

        W_grad_int = W_grad.to(bnn.type.INTEGER)


        #out_grad
        out_grad = self.gradient(W=W, input=input, grad=grad)

        out_tern_grad = self.ternarise(out_grad)
        out_tern_grad_int = out_tern_grad.to(bnn.type.INTEGER)



        return W_grad_int, out_tern_grad_int

    def reshape_grad(self, grad: torch.Tensor) -> torch.Tensor: #he just defined this here to give us option to override in the subclasses (namely, BackwardBitCountMax)
        return grad

    def gradient(self, W: torch.Tensor, input: torch.Tensor, grad: torch.Tensor) -> torch.Tensor: #same here, just defining so we can override in subclasses
        return functions.int_matmul(grad, W.T)

    @abc.abstractmethod
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor: ...


class SignTernarise(BackprojectTernarise):
    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        return functions.ternarise(grad, threshold_lo=0, threshold_hi=1)


class StochasticTernarise(BackprojectTernarise):
    alpha: float

    def __init__(self, alpha=1):
        self.alpha = alpha

    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        EPS = 1e-8
        sign = torch.sign(grad)
        scaled = self.alpha * torch.abs(grad) / (self.hidden_dim * (1 - self.sparsity + EPS))

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
        output_ste = torch.abs(output) <= self.zero_grad_mag_thresh     #outputs a boolean tensor {0,1}
        grad_ste = grad * output_ste    #basically if output pre-activations during forward pass are zero, the gradient in backward pass is zeroed out 

        out_grad = functions.int_matmul(grad_ste, W.T)  #then continue as normal, backpropagating the gradient

        return out_grad #not ternary since matmul of ternary matrices (grad_ste and W.T) is not ternary

    def ternarise(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.sign(grad)     #ternarising the out_grad produced by gradient()


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



# my new backwards function.
class reluBackward(BackwardFunc):
    #i think these are for metrics.
    hidden_dim: int
    sparsity: float

    def __call__(
        self,
        grad: torch.Tensor,
        input: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        threshold: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        #i think these are for metrics.
        self.hidden_dim = W.shape[-1]
        self.sparsity = torch.mean((W == 0).to(torch.float))


        preactivations = input @ W + b

        if((preactivations > 0).abs().sum() == 0):
            print("None of the preacts are > zero!?!?!?!?!?!?")

        Z_grad = grad * (preactivations > 0).to(grad.dtype) #grad is in Reals, so Z_grad will be in Reals too
        out_grad = Z_grad @ (W.T)                #Z_grad is in Reals, so out_grad will be in Reals too
        W_grad = (input.T) @ Z_grad              #Z_grad is in Reals, and input is in integer, so W_grad will be in Reals too
        b_grad = Z_grad.sum(dim=0)               #sum over batch dimension  

        return W_grad, b_grad, out_grad




class Modal(BackwardFunc): #this does not train.
    hidden_dim: int
    sparsity: float

    def __call__(
        self,
        grad: torch.Tensor,
        input: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        threshold: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        #i think these are for metrics.
        self.hidden_dim = W.shape[-1]
        self.sparsity = torch.mean((W == 0).to(torch.float))

        preactivations = input @ W + b

        if((preactivations > 0).abs().sum() == 0):
            print("None of the preacts are > zero!?!?!?!?!?!?")

        Z_grad = grad * (preactivations > 0).to(grad.dtype) #grad is in Reals, so Z_grad will be in Reals too
        out_grad = Z_grad @ (W.T)                #Z_grad is in Reals, so out_grad will be in Reals too
        W_grad = (input.T) @ Z_grad              #Z_grad is in Reals, and input is in integer, so W_grad will be in Reals too
        b_grad = Z_grad.sum(dim=0)               #sum over batch dimension  

        if numpy.isnan(threshold):
            W_grad_modal = W_grad
            
        else:
            threshold = round(threshold)
            turning_points = torch.full_like(grad, threshold-1) #if our binarise threshold is set to 0, then turning point is -1. if binarise threshold is 1, turning point is 0.
            neg_mask = preactivations < threshold #if the preact is coming from the negative side, then turning point is threshold, else turning point is threshold - 1
            turning_points[neg_mask] = threshold 

            dx = turning_points - preactivations
            dx = dx.abs()
            dx, _ = torch.max(dx, dim=0)    #selecting the max dx: dx is the minimal distance (lower bound) required to change the activation. However dx is different for each datapoint, the correct lower bound is then of course the largest dx.

            W_grad_active = torch.zeros_like(W_grad)
            W_grad_modal = torch.zeros_like(W_grad)
            for col_idx in range(W_grad.shape[1]):
                n = dx[col_idx]  # the number of weights to change for this neuron (each column of W)
                if n == 0:
                    print("n is 0")

                quotient, remainder = divmod(n.item(), W_grad.shape[0])
                
                if n > W_grad.shape[0]:
                    #i dont think this should ever be possible - unless there's a bug?
                    print(f"dx larger than number of rows in W! dx: {n}, W_grad.shape[0]: {W_grad.shape[0]}")
                    print(quotient)
                    # W_grad_modal[:,col_idx] = (quotient) * torch.sign(W_grad[:,col_idx]) #not sure this is correct
                
                _, indices = torch.topk(W_grad.abs()[:, col_idx], k=int(remainder), largest=True)  # Get top-n indices
                W_grad_active[indices, col_idx] = 1  # set these indices to 1 in current col_idx of W_grad_active

            W_grad_modal += W_grad_active * torch.sign(W_grad)

        return W_grad_modal, b_grad, out_grad