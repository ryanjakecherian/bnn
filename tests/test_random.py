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
