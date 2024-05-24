import torch

import bnn.functions
import bnn.random


class TernBinLayer(torch.nn.Module):
    input_dim: int
    output_dim: int
    forward_func: bnn.functions.ForwardFunc
    backward_func: bnn.functions.BackwardFunc

    W: torch.nn.Parameter

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        forward_func: bnn.functions.ForwardFunc,
        backward_func: bnn.functions.BackwardFunc,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self._create_W()
        self._initialise_W()

        self.forward_func = forward_func
        self.backward_func = backward_func

    def _create_W(self):
        self.W = torch.nn.Parameter(
            data=torch.zeros(self.input_dim, self.output_dim, dtype=torch.int),
            requires_grad=False,
        )

    def _initialise_W(
        self,
        var: float | None = None,
        mean: float | None = None,
        zero_prob: float | None = None,
    ):
        if (var is not None) and (zero_prob is not None):
            raise ValueError("Can't specify both var and nonzero prob")

        if mean is None:
            mean = 0
        if var is None and zero_prob is None:
            zero_prob = 0.5

        if zero_prob is None:
            distribution = bnn.random.get_ternary_distribution_from_mean_and_var(mean=mean, var=var)
        else:
            distribution = bnn.random.get_ternary_distribution_from_mean_and_zero_prob(mean=mean, zero_prob=zero_prob)

        self.W.data = bnn.random.sample_iid_tensor_from_discrete_distribution(
            self.W.shape,
            distribution=distribution,
        )
        # reset grad
        self.W.grad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_func(x=x, W=self.W)

    def backward(self, grad: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        """Backproject gradient signal and update W_grad."""
        # FIXME check if long_int -> int conversion is safe!
        W_grad = grad.unsqueeze(-2) * activation.unsqueeze(-1)

        while W_grad.dim() > 2:
            W_grad = W_grad.sum(0)

        # TODO should this be ternarised?
        W_grad_int = W_grad.to(torch.int)
        self.W.grad = W_grad_int

        grad = grad @ self.W.T
        # TODO pick this threshold nicely... adaptively?
        # TODO implenent layer-normy type of thing...
        grad = bnn.functions.ternarise(grad, threshold_lo=0, threshold_hi=1)

        return grad.to(torch.int)

    def __repr__(self):
        return f'W: {self.W}'
