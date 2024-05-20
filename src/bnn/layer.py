import torch

import bnn.functions
import bnn.random


class TernBinLayer(torch.nn.Module):
    input_dim: int
    output_dim: int
    bit_shift: int

    project: bool

    W: torch.nn.Parameter
    W_grad: torch.nn.Parameter

    def __init__(self, input_dim: int, output_dim: int, bit_shift: int, project: bool):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bit_shift = bit_shift

        self.project = project

        self._create_W()
        self._initialise_W()

    def _create_W(self):
        self.W = torch.nn.Parameter(
            data=torch.zeros(self.input_dim, self.output_dim, dtype=torch.int),
            requires_grad=False,
        )
        self.W_grad = torch.nn.Parameter(
            data=torch.empty_like(self.W),
            requires_grad=False,
        )

    def _initialise_W(self, desired_var: None | float = None):
        if desired_var is None:
            desired_var = bnn.random.calc_desired_var(
                dim=self.output_dim,
                bit_shift=self.bit_shift,
            )
        if desired_var < 0:
            raise ValueError(f'desired_var {desired_var} is not a valid probability!')
        elif desired_var > 1:
            desired_var = 1
            # raise RuntimeWarning(f"desired_var {desired_var}>1! Setting desired_var=1")

        self.W[:] = bnn.random.generate_random_ternary_tensor(
            shape=self.W.shape,
            desired_var=desired_var,
            dtype=torch.int,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO make this a custom ternary multiplication, with custom backwards?
        integer = x @ self.W
        out_binary = bnn.functions.binarise.apply(integer)

        return out_binary

    def backward(self, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __repr__(self):
        return f'W: {self.W}'
