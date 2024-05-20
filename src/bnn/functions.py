import torch
import torch.autograd


class binarise(torch.autograd.Function):
    # HACK should be boolean...
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: int = 0) -> torch.Tensor:
        out = torch.ones_like(x, dtype=torch.int)
        out[x < 0] = -1

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class tern_bin_matmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        W: torch.Tensor,
        x: torch.Tensor,
        project: bool = False,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, W)
        ctx.project = project
        return x @ W

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, W = ctx.saved_tensors
        grad_x = grad_output @ W.T
        grad_W = grad_output.unsqueeze(-2) * x.unsqueeze(-1)

        if ctx.project:
            grad_x = binarise.apply(grad_x)

        return grad_W, grad_x, None


class bit_shift(torch.autograd.Function):
    # HACK is this even necessary...?
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int) -> torch.Tensor:
        ctx.bits = bits
        divided = x / (2**ctx.bits)
        divided = torch.floor(divided)

        return divided

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        divided = grad_output / (2**ctx.bits)
        divided = torch.floor(divided)

        return divided, None
