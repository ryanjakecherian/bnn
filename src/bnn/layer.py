import torch

import bnn.functions
import bnn.utils


class TernBinLayer(torch.nn.Module):
    input_dim: int
    output_dim: int
    bit_shift: int

    W: torch.nn.Parameter

    def __init__(self, input_dim: int, output_dim: int, bit_shift: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bit_shift = bit_shift

        self._create_W()
        self._initialise_W()

    def _create_W(self):
        self.W = torch.nn.Parameter(
            torch.zeros(self.input_dim, self.output_dim, dtype=torch.float),
        )

    def _initialise_W(self, desired_var: None | float = None):
        if desired_var is None:
            desired_var = bnn.utils.calc_desired_var(
                dim=self.output_dim,
                bit_shift=self.bit_shift,
            )
        if desired_var < 0:
            raise ValueError(f"desired_var {desired_var} is not a valid probability!")
        elif desired_var > 1:
            desired_var = 1
            #raise RuntimeWarning(f"desired_var {desired_var}>1! Setting desired_var=1")

        self.W.requires_grad = False

        self.W[:] = (torch.rand_like(self.W) < desired_var).float()
        half = torch.rand_like(self.W) < 0.5
        self.W[half] = -self.W[half]

        self.W.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        integer = x @ self.W
        bitshifted = bnn.functions.bit_shift.apply(integer, self.bit_shift)
        out_binary = bnn.functions.binarise.apply(bitshifted)

        return out_binary

    def __repr__(self):
        return f"W: {self.W}"
