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


class tern_bin_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W: torch.Tensor, x: torch.Tensor):
        ctx.x = x
        ctx.W = W
        return x @ W

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = grad_output @ ctx.W.T
        grad_W = grad_output.unsqueeze(-2) * ctx.x.unsqueeze(-1)

        return grad_W, grad_x


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
        divided = grad_output / (2**ctx.bits)
        divided = torch.floor(divided)

        return divided, None
