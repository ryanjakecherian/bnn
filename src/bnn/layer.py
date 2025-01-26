import torch

import bnn.functions
import bnn.random
import bnn.type

__all__ = [
    'TernBinLayer',
]


class TernBinLayer(torch.nn.Module):
    input_dim: int
    output_dim: int
    forward_func: bnn.functions.ForwardFunc
    backward_func: bnn.functions.BackwardFunc
    backward_actual_func: bnn.functions.BackwardFunc = bnn.functions.backward.ActualGradient()

    W: torch.nn.Parameter
    b: torch.nn.Parameter

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

        self._create_W_and_b()
        self._initialise_W_and_b()

        self.forward_func = forward_func
        self.backward_func = backward_func

    def to(self, device: torch.device):
        try:
            self.backward_func.to(device)
        except AttributeError:
            pass
        super().to(device)

    def _create_W_and_b(self):        #create_params()
        self.W = torch.nn.Parameter(
            data=torch.zeros(self.input_dim, self.output_dim, dtype=torch.float),   #FIXME HAVE TO MAKE IT A FLOAT OTHERWISE CANT GET FLOAT GRADS 
            requires_grad=False,    #tells autograd not to track gradients with respect to this tensor
        )
        self.b = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, dtype=torch.float),    #FIXME HAVE TO MAKE IT A FLOAT OTHERWISE CANT GET FLOAT GRADS
            requires_grad=False,    #tells autograd not to track gradients with respect to this tensor
        )


    def _initialise_W_and_b(          #initialise_params() (where we remember that W and b have different initialisations) (normally b initialised to 0 ?) b \in mathbb{R}^{d_out} default autograd. bias updates dont need to be quantised. lr_b and lr_w may need to be different.
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

        #no need to init bias, already initialised to 0
        # actually: try init to large positive value, e.g. 10
        self.b.data = torch.full_like(self.b.data, fill_value=512)

        # reset grad
        self.W.grad = None
        self.b.grad = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x,z = self.forward_func(x=x, W=self.W, b=self.b)
        return x,z

    def backward(self, grad: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        """Backproject gradient signal and update W_grad."""
        W_grad, b_grad, out_grad = self.backward_func(grad=grad, input=activation, W=self.W, b=self.b) 
        
        self.W.grad = W_grad
        self.b.grad = b_grad

        return out_grad

    def backward_actual(self, grad: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        """Backproject gradient signal and update W_grad."""
        W_grad, out_grad = self.backward_actual_func(grad=grad, input=activation, W=self.W)

        self.W.grad = W_grad

        return out_grad

    def __repr__(self):
        return f'W: {self.W}, b: {self.b}'
