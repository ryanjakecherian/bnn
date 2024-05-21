import torch

import bnn.functions
import bnn.random


class TernBinLayer(torch.nn.Module):
    input_dim: int
    output_dim: int

    W: torch.nn.Parameter

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self._create_W()
        self._initialise_W()

    def _create_W(self):
        self.W = torch.nn.Parameter(
            data=torch.zeros(self.input_dim, self.output_dim, dtype=torch.int),
            requires_grad=False,
        )

    def _initialise_W(self, var: float | None = None, mean: float | None = None):
        if mean is None:
            mean = 0
        if var is None:
            var = 0.5

        distribution = bnn.random.get_ternary_distribution_from_mean_and_var(mean, var)
        self.W.data = bnn.random.sample_iid_tensor_from_discrete_distribution(
            self.W.shape,
            distribution=distribution,
        )
        # reset grad
        self.W.grad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        integer = x @ self.W
        out_binary = bnn.functions.binarise(x=integer, threshold=0)

        return out_binary

    def backward(self, grad: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        """Backproject gradient signal and update W_grad."""
        # HACK check if long_int -> int conversion is safe!
        W_grad = (grad.unsqueeze(-2) * activation.unsqueeze(-1)).sum(0)
        W_grad_int = W_grad.to(torch.int)
        self.W.grad = W_grad_int

        grad = grad @ self.W.T
        # TODO pick these threshold nicely
        grad = bnn.functions.ternarise(grad, threshold_lo=0, threshold_hi=1)

        return grad.to(torch.int)

    def __repr__(self):
        return f'W: {self.W}'
