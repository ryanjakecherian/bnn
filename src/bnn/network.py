import torch

import bnn.layer


class TernBinNetwork(torch.nn.Module):
    layers: torch.nn.ModuleList

    def __init__(self, *dims: list[int], bit_shift: int):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for input_dim, output_dim in zip(dims, dims[1:]):
            layer = bnn.layer.TernBinLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                bit_shift=bit_shift,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x
