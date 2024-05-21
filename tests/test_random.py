import bnn.random
import pytest
import torch

test_generate_random_ternary_tensor_cases = [0.1 * i for i in range(11)]


@pytest.mark.parametrize('desired_var', test_generate_random_ternary_tensor_cases)
def test_generate_random_ternary_tensor(desired_var):
    torch.manual_seed(42)

    for DTYPE in (torch.int, torch.float):
        SHAPE = [3000, 3000]

        random = bnn.random.generate_random_ternary_tensor(
            shape=SHAPE,
            desired_var=desired_var,
            dtype=DTYPE,
        )

        assert random.to(torch.float).var() == pytest.approx(desired_var, abs=0.01)
        assert random.to(torch.float).mean() == pytest.approx(0, abs=0.01)


test_discrete_mean_cases = [
    (
        [(1, 1)],
        1,
    ),
    (
        [(0, 0.5), (1, 0.5)],
        0.5,
    ),
    (
        [(-1, 0), (0, 0.5), (1, 0.5)],
        0.5,
    ),
    (
        [(-1, 0.4), (0, 0.2), (1, 0.4)],
        0,
    ),
]


@pytest.mark.parametrize('distr, expected_mean', test_discrete_mean_cases)
def test_discrete_mean(distr, expected_mean):
    mean = bnn.random.discrete_mean(distr)
    assert mean == pytest.approx(expected_mean)


test_discrete_var_cases = [
    (
        [(1, 1)],
        0,
    ),
    (
        [(0, 0.5), (1, 0.5)],
        0.25,
    ),
    (
        [(-1, 0), (0, 0.5), (1, 0.5)],
        0.25,
    ),
    (
        [(-1, 0.4), (0, 0.2), (1, 0.4)],
        0.8,
    ),
]


@pytest.mark.parametrize('distr, expected_var', test_discrete_var_cases)
def test_discrete_var(distr, expected_var):
    var = bnn.random.discrete_var(distr)
    assert var == pytest.approx(expected_var)


test_get_ternary_distribution_from_mean_and_var_cases = [
    (0.5, 0.5),
    (0.1, 0.5),
    (-0.5, 0.5),
    (-0.1, 0.5),
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 0.1),
    (0, 0.01),
]


@pytest.mark.parametrize(
    'mean, var',
    test_get_ternary_distribution_from_mean_and_var_cases,
)
def test_get_ternary_distribution_from_mean_and_var(mean, var):
    distribution = bnn.random.get_ternary_distribution_from_mean_and_var(mean, var)

    assert set(value for value, _ in distribution) == {-1, 0, 1}
    assert sum(prob for _, prob in distribution) == pytest.approx(1)

    assert bnn.random.discrete_mean(distribution) == pytest.approx(mean)
    assert bnn.random.discrete_var(distribution) == pytest.approx(var)
