import bnn.loss
import bnn.type
import pytest
import torch

test_l1_forward_cases = [
    (
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        0,
    ),
    (
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        -torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        12,
    ),
    (
        torch.Tensor([1, 1, -1, 1, 1, 1]).to(bnn.type.INTEGER),
        torch.Tensor([1, -1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        4,
    ),
    (
        torch.Tensor([-1, -1, -1, 1, 1, 1]).to(bnn.type.INTEGER),
        torch.Tensor([1, -1, 1, 1, 1, 0]).to(bnn.type.INTEGER),
        5,
    ),
]


@pytest.mark.parametrize('output, target, expected_loss', test_l1_forward_cases)
def test_l1_forward(output, target, expected_loss):
    loss = bnn.loss.l1.forward(output=output, target=target)
    assert loss == expected_loss


test_l1_backward_cases = [
    (
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        torch.Tensor([0, 0, 0, 0, 0, 0]).to(bnn.type.INTEGER),
    ),
    (
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        -torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
    ),
    (
        torch.Tensor([1, 1, -1, 1, 1, 1]).to(bnn.type.INTEGER),
        torch.Tensor([1, -1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        -torch.Tensor([0, -1, 1, 0, 0, 0]).to(bnn.type.INTEGER),
    ),
    (
        torch.Tensor([-1, -1, -1, 1, 1, 1]).to(bnn.type.INTEGER),
        torch.Tensor([1, -1, 1, 1, 1, 1]).to(bnn.type.INTEGER),
        -torch.Tensor([1, 0, 1, 0, 0, 0]).to(bnn.type.INTEGER),
    ),
]


@pytest.mark.parametrize('output, target, expected_grad', test_l1_backward_cases)
def test_l1_backward(output, target, expected_grad):
    grad = bnn.loss.l1.backward(output=output, target=target)
    torch.testing.assert_close(grad, expected_grad)
