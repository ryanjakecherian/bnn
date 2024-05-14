import torch

import bnn.layer


class TernBinNetwork(torch.nn.Module):
    layers: torch.nn.ModuleDict

    def __init__(self, *dims: list[int], bit_shift: int):
        super().__init__()

        self.layers = torch.nn.ModuleDict()
        for i, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layer = bnn.layer.TernBinLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                bit_shift=bit_shift,
            )
            self.layers[f'TernBinLayer{i}'] = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            x = layer(x)

        return x
