import torch

import bnn.layer


class TernBinNetwork(torch.nn.Module):
    layers: torch.nn.ModuleDict

    activation: torch.nn.ParameterDict
    activation_grad: torch.nn.ParameterDict

    project: bool
    # TODO add activations as parameters...?

    def __init__(self, *dims: list[int], bit_shift: int, project: bool):
        super().__init__()

        self.project = project
        self.layers = torch.nn.ModuleDict()
        self.activation = torch.nn.ParameterDict()
        self.activation_grad = torch.nn.ParameterDict()

        # init layers
        for i, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layer = bnn.layer.TernBinLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                bit_shift=bit_shift,
                project=project,
            )

            layer_name = f'TernBinLayer{i}'
            self.layers[layer_name] = layer

            # init activations and grads
            self.activation[layer_name] = torch.nn.Parameter(
                data=torch.empty(input_dim, output_dim, dtype=torch.int),
                requires_grad=False,
            )
            self.activation_grad[layer_name] = torch.nn.Parameter(
                data=torch.empty(input_dim, output_dim, dtype=torch.int),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_name, layer in self.layers.items():
            x = layer(x)
            self.activation[layer_name].data = x

        return x

    def backward(self, grad: torch.Tensor) -> None:
        raise NotImplementedError
