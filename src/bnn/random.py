import torch


def calc_desired_var(
    dim: int,
    bit_shift: int,
) -> float:
    bit_shift_var = 2 ** (bit_shift * 2)
    return 1 * bit_shift_var / dim


def generate_random_ternary_tensor(
    shape: list[int],
    desired_var: float,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    random = (torch.rand(shape) < desired_var).to(dtype)
    half = torch.rand_like(random, dtype=float) < 0.5
    random[half] = -random[half].to(dtype)

    return random
