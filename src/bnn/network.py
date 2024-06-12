import torch

import bnn.functions
import bnn.layer

__all__ = [
    'TernBinNetwork',
]


class TernBinNetwork(torch.nn.Module):
    layers: torch.nn.ModuleDict

    input: torch.nn.ParameterDict
    grad: torch.nn.ParameterDict

    dims: list[int]

    def __init__(
        self,
        dims: list[int],
        forward_func: bnn.functions.ForwardFunc = bnn.functions.forward.SignBinarise(),
        backward_func: bnn.functions.BackwardFunc = bnn.functions.backward.SignTernarise(),
    ):
        super().__init__()

        self.layers = torch.nn.ModuleDict()
        self.input = torch.nn.ParameterDict()
        self.grad = torch.nn.ParameterDict()

        self.dims = dims

        # init layers
        for i, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layer = bnn.layer.TernBinLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                forward_func=forward_func,
                backward_func=backward_func,
            )

            layer_name = f'TernBinLayer{i}'
            self.layers[layer_name] = layer

        self._clear_input_and_grad()
        return

    def _clear_input_and_grad(self) -> None:
        for layer_name, layer in self.layers.items():
            # TODO check - should this be in-place or create a new parameter?
            # init activations and grads
            self.input[layer_name] = torch.nn.Parameter(
                data=torch.zeros(layer.input_dim, layer.output_dim, dtype=torch.int),
                requires_grad=False,
            )
            self.grad[layer_name] = torch.nn.Parameter(
                data=torch.zeros_like(self.input[layer_name]),
                requires_grad=False,
            )

        return

    def _initialise(
        self,
        W_mean: float | None | list[float | None] = None,
        W_var: float | None | list[float | None] = None,
        W_zero_prob: float | None | list[float | None] = None,
    ) -> None:
        # input sanity checking
        if (W_zero_prob is not None) and (W_var is not None):
            raise ValueError('Cannot spacify both var and nonzero prob!')

        if isinstance(W_mean, list):
            if len(W_mean) != len(self.layers):
                raise IndexError(
                    'Number of means does not match number of layers! '
                    f'Means: {len(W_mean)} Layers: {len(self.layers)}'
                )
        else:
            W_mean = [W_mean] * len(self.layers)

        if isinstance(W_var, list):
            if len(W_var) != len(self.layers):
                raise IndexError(
                    'Number of vars does not match number of layers! ' f'Vars: {len(W_var)} Layers: {len(self.layers)}'
                )
        else:
            W_var = [W_var] * len(self.layers)

        if isinstance(W_zero_prob, list):
            if len(W_zero_prob) != len(self.layers):
                raise IndexError(
                    'Number of zero probs does not match number of layers! '
                    f'zero probs: {len(W_zero_prob)} Layers: {len(self.layers)}'
                )
        else:
            W_zero_prob = [W_zero_prob] * len(self.layers)

        # reset all weight vars
        zipped = zip(self.layers.values(), W_var, W_mean, W_zero_prob)
        for layer, var, mean, zero_prob in zipped:
            layer: bnn.layer.TernBinLayer
            layer._initialise_W(mean=mean, var=var, zero_prob=zero_prob)

        # reset grads and activations
        self._clear_input_and_grad()

        return

    def forward_no_proj(self, x: torch.Tensor) -> torch.Tensor:
        last_layer = list(self.layers.keys())[-1]

        for layer_name, layer in self.layers.items():
            if layer_name != last_layer:
                self.input[layer_name].data = x
                x = layer(x)

        x = bnn.functions.int_matmul(x, layer.W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_name, layer in self.layers.items():
            self.input[layer_name].data = x
            x = layer(x)

        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Propagate gradients back through network and output grad wrt input."""
        for layer_name, layer in reversed(self.layers.items()):
            layer: bnn.layer.TernBinLayer

            grad = layer.backward(grad, self.input[layer_name])
            self.grad[layer_name].data = grad

        return grad


def network_params_al_ternary(Net: torch.nn.Module) -> bool:
    for parameter in Net.parameters():
        is_one = parameter == 1
        is_neg_one = parameter == -1
        is_zero = parameter == 0

        return torch.all(is_one | is_neg_one | is_zero)
