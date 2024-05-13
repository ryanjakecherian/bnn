import pytest
import torch

import bnn.functions

test_bit_shift_examples_cases = [
    (
        # input
        torch.Tensor([1]).float(),
        # bit_shift
        1,
        # expected_output
        torch.Tensor([0]).float(),
    ),
    (
        # input
        torch.Tensor([1]).float(),
        # bit_shift
        2,
        # expected_output
        torch.Tensor([0]).float(),
    ),
    (
        # input
        torch.Tensor([1]).float(),
        # bit_shift
        3,
        # expected_output
        torch.Tensor([0]).float(),
    ),
    (
        # input
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]).float(),
        # bit_shift
        1,
        # expected_output
        torch.Tensor([0, 0, 1, 1, 2, 2, 3, 3, 4]).float(),
    ),
    (
        # input
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]).float(),
        # bit_shift
        2,
        # expected_output
        torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2]).float(),
    ),
]


@pytest.mark.parametrize(
    "input, bit_shift, expected_output",
    test_bit_shift_examples_cases
)
def test_bit_shift_examples(input, bit_shift, expected_output):
    output = bnn.functions.bit_shift.apply(input, bit_shift)
    torch.testing.assert_close(actual=output, expected=expected_output, atol=0, rtol=0)
    

def test_bit_shift_random():
    torch.random.manual_seed(42)

    random = torch.randn(10000, dtype=torch.float) * 100 + 200
    random_int = random.to(torch.int)
    random = random_int.float()

    for i in range(3):
        expected_rshift = torch.bitwise_right_shift(random_int, i).float()
        actual_rshift = bnn.functions.bit_shift.apply(random, i)
        torch.testing.assert_close(expected_rshift, actual_rshift, atol=0, rtol=0)

        expected_lshift = torch.bitwise_left_shift(random_int, i).float()
        actual_lshift = bnn.functions.bit_shift.apply(random, -i)
        torch.testing.assert_close(expected_lshift, actual_lshift, atol=0, rtol=0)
