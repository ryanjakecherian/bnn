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


VALUE_PROB_PAIR = tuple[int, float]
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

    return (-1, p_MINUS), (0, p_ZERO), (1, p_PLUS)


def sample_iid_tensor_from_discrete_distribution(
    shape: list[int],
    distribution: DISCRETE_DIST,
) -> torch.Tensor:
    check_is_valid_distribution(distribution)

    out = torch.empty(shape, dtype=torch.int)

    uniform = torch.rand_like(out, dtype=torch.float)

    for value, prob in distribution:
        uniform -= prob
        out[uniform < 0] = value
        uniform[uniform < 0] += 1

    return out


def discrete_mean(dist: DISCRETE_DIST) -> float:
    return sum(prob * value for value, prob in dist)


def discrete_var(dist: DISCRETE_DIST) -> float:
    mean = discrete_mean(dist)
    return sum(value**2 * prob for value, prob in dist) - mean**2


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
    _, probs = zip(*distribution)

    check_is_valid_probability(*probs)

    TOL = 1e-3
    if abs(sum(probs) - 1) > TOL:
        raise ValueError("Invalid distribution doesn't sum to 1.")
