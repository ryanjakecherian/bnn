import bnn.random
import bnn.type
import pytest
import torch
from bnn.random import VALUE_PROB_PAIR as VPP

test_generate_random_ternary_tensor_cases = [0.1 * i for i in range(11)]


@pytest.mark.parametrize('desired_var', test_generate_random_ternary_tensor_cases)
def test_generate_random_ternary_tensor(desired_var):
    torch.manual_seed(42)

    for DTYPE in (bnn.type.INTEGER, torch.float):
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
        [VPP(1, 1)],
        1,
    ),
    (
        [VPP(0, 0.5), VPP(1, 0.5)],
        0.5,
    ),
    (
        [VPP(-1, 0), VPP(0, 0.5), VPP(1, 0.5)],
        0.5,
    ),
    (
        [VPP(-1, 0.4), VPP(0, 0.2), VPP(1, 0.4)],
        0,
    ),
]


@pytest.mark.parametrize('distr, expected_mean', test_discrete_mean_cases)
def test_discrete_mean(distr, expected_mean):
    mean = bnn.random.discrete_mean(distr)
    assert mean == pytest.approx(expected_mean)


test_discrete_var_cases = [
    (
        [VPP(1, 1)],
        0,
    ),
    (
        [VPP(0, 0.5), VPP(1, 0.5)],
        0.25,
    ),
    (
        [VPP(-1, 0), VPP(0, 0.5), VPP(1, 0.5)],
        0.25,
    ),
    (
        [VPP(-1, 0.4), VPP(0, 0.2), VPP(1, 0.4)],
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

    assert set(pair.value for pair in distribution) == {-1, 0, 1}
    assert sum(pair.probability for pair in distribution) == pytest.approx(1)

    assert bnn.random.discrete_mean(distribution) == pytest.approx(mean)
    assert bnn.random.discrete_var(distribution) == pytest.approx(var)


test_get_ternary_distribution_from_mean_and_zero_prob_cases = [
    (0, 0),
    (0, 0.1),
    (0, 0.4),
    (0, 0.9),
    (1, 0),
    (-1, 0),
    (0.5, 0.5),
]


@pytest.mark.parametrize('mean, zero_prob', test_get_ternary_distribution_from_mean_and_zero_prob_cases)
def test_get_ternary_distribution_from_mean_and_zero_prob(mean, zero_prob):
    distribution = bnn.random.get_ternary_distribution_from_mean_and_zero_prob(
        mean=mean,
        zero_prob=zero_prob,
    )

    assert set(pair.value for pair in distribution) == {-1, 0, 1}
    assert sum(pair.probability for pair in distribution) == pytest.approx(1)

    approx_zero = pytest.approx(zero_prob)
    assert bnn.random.discrete_mean(distribution) == pytest.approx(mean)
    assert sum(pair.probability for pair in distribution if pair.value == 0) == approx_zero


test_sample_iid_tensor_from_discrete_distribution_cases = [
    (
        (1000, 1000),
        [VPP(-1, 0.5), VPP(1, 0.5)],
    ),
    (
        (1000, 1000, 1, 1),
        [VPP(-1, 0.5), VPP(1, 0.5)],
    ),
    (
        (1000, 1000),
        [VPP(-1, 0.2), VPP(0, 0.3), VPP(1, 0.5)],
    ),
]


@pytest.mark.parametrize(
    'shape, distribution',
    test_sample_iid_tensor_from_discrete_distribution_cases,
)
def test_sample_iid_tensor_from_discrete_distribution(shape, distribution):
    tensor = bnn.random.sample_iid_tensor_from_discrete_distribution(
        shape,
        distribution=distribution,
    )
    assert tensor.shape == shape

    tensor = tensor.flatten()
    for pair in distribution:
        value, prob = pair.value, pair.probability
        empirical_prob = torch.sum(tensor == value) / len(tensor)
        assert empirical_prob == pytest.approx(prob, abs=0.05)


test_check_is_valid_distribution_cases = [
    ([VPP(1, 1), VPP(1, 1), VPP(1, 1)], ValueError),
    ([VPP(1, 1), VPP(1, 1), VPP(1, 0)], ValueError),
    ([VPP(-1, 0.5), VPP(0, 0.5), VPP(1, 0.5)], ValueError),
    ([VPP(-1, 0.5), VPP(0, 0.1), VPP(1, 0.1)], ValueError),
    ([VPP(-1, 2), VPP(1, -1)], ValueError),
]


@pytest.mark.parametrize('distribution, error', test_check_is_valid_distribution_cases)
def test_check_is_valid_distribution(distribution, error):
    with pytest.raises(error):
        bnn.random.check_is_valid_distribution(distribution)
