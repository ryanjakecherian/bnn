import torch

import bnn.layer


class TernBinNetwork(torch.nn.Module):
    layers: torch.nn.ModuleDict

    input: torch.nn.ParameterDict
    grad: torch.nn.ParameterDict

    def __init__(self, *dims: list[int]):
        super().__init__()

        self.layers = torch.nn.ModuleDict()
        self.input = torch.nn.ParameterDict()
        self.grad = torch.nn.ParameterDict()

        # init layers
        for i, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layer = bnn.layer.TernBinLayer(
                input_dim=input_dim,
                output_dim=output_dim,
            )

            layer_name = f'TernBinLayer{i}'
            self.layers[layer_name] = layer

            # init activations and grads
            self.input[layer_name] = torch.nn.Parameter(
                data=torch.empty(input_dim, output_dim, dtype=torch.int),
                requires_grad=False,
            )
            self.grad[layer_name] = torch.nn.Parameter(
                data=torch.empty(input_dim, output_dim, dtype=torch.int),
                requires_grad=False,
            )

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
