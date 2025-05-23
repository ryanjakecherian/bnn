import bnn.functions
import bnn.type
import pytest
import torch

test_ternarise_cases = [
    # sign(x)
    (
        {
            'x': torch.Tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
            'threshold_hi': 1,
            'threshold_lo': 0,
        },
        torch.Tensor([-1, -1, -1, -1, 0, 1, 1, 1, 1]),
    ),
    # sign(x) no 0
    (
        {
            'x': torch.Tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
            'threshold_hi': 0,
            'threshold_lo': 0,
        },
        torch.Tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1]),
    ),
    # 0 if in [-1, 1] (inclusive)
    (
        {
            'x': torch.Tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
            'threshold_hi': 2,
            'threshold_lo': -1,
        },
        torch.Tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]),
    ),
]


@pytest.mark.parametrize('inputs, expected_output', test_ternarise_cases)
def test_ternarise(inputs, expected_output):
    output = bnn.functions.ternarise(**inputs)
    torch.testing.assert_close(output, expected_output)


test_binarise_cases = [
    # sign(x)
    (
        {
            'x': torch.Tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4]).to(dtype=bnn.type.INTEGER),
            'threshold': 0,
        },
        torch.Tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1]).to(dtype=bnn.type.INTEGER),
    ),
    # sign(x+1)
    (
        {
            'x': torch.Tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4]).to(dtype=bnn.type.INTEGER),
            'threshold': -1,
        },
        torch.Tensor([-1, -1, -1, 1, 1, 1, 1, 1, 1]).to(dtype=bnn.type.INTEGER),
    ),
    # sign(x-3)
    (
        {
            'x': torch.Tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4]).to(dtype=bnn.type.INTEGER),
            'threshold': 3,
        },
        torch.Tensor([-1, -1, -1, -1, -1, -1, -1, 1, 1]).to(bnn.type.INTEGER),
    ),
]


@pytest.mark.parametrize('inputs, expected_output', test_binarise_cases)
def test_binarise(inputs, expected_output):
    output = bnn.functions.binarise(**inputs)
    torch.testing.assert_close(output, expected_output)


test_one_hot_max_cases = [
    (
        torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(dtype=bnn.type.INTEGER),
        torch.Tensor([-1, -1, -1, -1, -1, -1, -1, 1]).to(dtype=bnn.type.INTEGER),
    ),
    (
        torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).to(dtype=bnn.type.INTEGER),
        torch.Tensor([[-1, -1, -1, -1, -1, -1, -1, 1]]).to(dtype=bnn.type.INTEGER),
    ),
    (
        -torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(dtype=bnn.type.INTEGER),
        torch.Tensor([1, -1, -1, -1, -1, -1, -1, -1]).to(dtype=bnn.type.INTEGER),
    ),
    (
        torch.Tensor([[1, -1], [-1, 1]]).to(dtype=bnn.type.INTEGER),
        torch.Tensor([[1, -1], [-1, 1]]).to(dtype=bnn.type.INTEGER),
    ),
    (
        torch.Tensor([[1, 1], [-1, 1]]).to(dtype=bnn.type.INTEGER),
        torch.Tensor([[1, -1], [-1, 1]]).to(dtype=bnn.type.INTEGER),
    ),
]


@pytest.mark.parametrize('inputs, expected_output', test_one_hot_max_cases)
def test_one_hot_max(inputs, expected_output):
    output = bnn.functions.one_hot_argmax(inputs)
    torch.testing.assert_close(output, expected_output)
