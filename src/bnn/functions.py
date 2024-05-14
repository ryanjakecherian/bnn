import torch
import torch.autograd


class binarise(torch.autograd.Function):
    # HACK should be boolean...
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = torch.ones_like(x, dtype=torch.float)
        out[x < 0] = -1

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class bit_shift(torch.autograd.Function):
    # HACK is this even necessary...?
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int) -> torch.Tensor:
        ctx.bits = bits
        divided = x / (2**ctx.bits)
        divided = torch.floor(divided)

        return divided

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output / (2**ctx.bits), None
