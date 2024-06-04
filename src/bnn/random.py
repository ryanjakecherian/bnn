import typing

import torch

__all__ = [
    'calc_desired_var',
    'generate_random_ternary_tensor',
    'get_ternary_distribution_from_mean_and_zero_prob',
    'sample_iid_tensor_from_discrete_distribution',
    'discrete_mean',
    'discrete_var',
    'DISCRETE_DIST',
    'TERNARY_DIST',
    'check_is_valid_probability',
    'check_is_valid_distribution',
]


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


class VALUE_PROB_PAIR(typing.NamedTuple):
    value: float
    probability: float


DISCRETE_DIST = list[VALUE_PROB_PAIR]
TERNARY_DIST = tuple[VALUE_PROB_PAIR, VALUE_PROB_PAIR, VALUE_PROB_PAIR]


def get_ternary_distribution_from_mean_and_var(
    mean: float,
    var: float,
) -> TERNARY_DIST:
    if var < 0:
        raise ValueError(f'{var} < 0, is not a valid variance.')

    p_PLUS = 0.5 * (var + mean**2 + mean)
    p_MINUS = p_PLUS - mean
    p_ZERO = 1 - p_PLUS - p_MINUS

    check_is_valid_probability(p_PLUS, p_MINUS, p_ZERO)

    MINUS = VALUE_PROB_PAIR(value=-1, probability=p_MINUS)
    ZERO = VALUE_PROB_PAIR(value=0, probability=p_ZERO)
    PLUS = VALUE_PROB_PAIR(value=1, probability=p_PLUS)

    return MINUS, ZERO, PLUS


def get_ternary_distribution_from_mean_and_zero_prob(
    mean: float,
    zero_prob: float,
) -> TERNARY_DIST:
    check_is_valid_probability(zero_prob)

    if abs(mean) > 1 - zero_prob:
        raise ValueError(f"mean {mean} and zero_prob {zero_prob} can't be achieved")

    p_ZERO = zero_prob
    p_PLUS = (mean + 1 - zero_prob) / 2
    p_MINUS = 1 - p_ZERO - p_PLUS

    check_is_valid_probability(p_PLUS, p_MINUS, p_ZERO)

    MINUS = VALUE_PROB_PAIR(value=-1, probability=p_MINUS)
    ZERO = VALUE_PROB_PAIR(value=0, probability=p_ZERO)
    PLUS = VALUE_PROB_PAIR(value=1, probability=p_PLUS)

    return MINUS, ZERO, PLUS


def sample_iid_tensor_from_discrete_distribution(
    shape: list[int],
    distribution: DISCRETE_DIST,
) -> torch.Tensor:
    check_is_valid_distribution(distribution)

    out = torch.empty(shape, dtype=torch.int)

    uniform = torch.rand_like(out, dtype=torch.float)

    for value_prob in distribution:
        value, prob = value_prob.value, value_prob.probability

        uniform -= prob
        out[uniform < 0] = value
        uniform[uniform < 0] += 1

    return out


def discrete_mean(dist: DISCRETE_DIST) -> float:
    return sum(pair.probability * pair.value for pair in dist)


def discrete_var(dist: DISCRETE_DIST) -> float:
    mean = discrete_mean(dist)
    return sum(pair.value**2 * pair.probability for pair in dist) - mean**2


def check_is_valid_probability(*xs: list[float]) -> None:
    for x in xs:
        error = None
        if x > 1:
            error = f'{x} > 1'
        elif x < 0:
            error = f'{x} < 0'

        if error:
            raise ValueError('Invalid probability! ' + error)

    return


def check_is_valid_distribution(distribution: DISCRETE_DIST) -> None:
    probs = [pair.probability for pair in distribution]

    check_is_valid_probability(*probs)

    TOL = 1e-3
    if abs(sum(probs) - 1) > TOL:
        raise ValueError("Invalid distribution doesn't sum to 1.")
