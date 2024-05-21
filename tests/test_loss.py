import bnn.loss
import pytest
import torch

test_l1_forward_cases = [
    (
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
        0,
    ),
    (
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
        -torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
        12,
    ),
    (
        torch.Tensor([1, 1, -1, 1, 1, 1]).to(torch.int),
        torch.Tensor([1, -1, 1, 1, 1, 1]).to(torch.int),
        4,
    ),
    (
        torch.Tensor([-1, -1, -1, 1, 1, 1]).to(torch.int),
        torch.Tensor([1, -1, 1, 1, 1, 0]).to(torch.int),
        5,
    ),
]


@pytest.mark.parametrize('output, target, expected_loss', test_l1_forward_cases)
def test_l1_forward(output, target, expected_loss):
    loss = bnn.loss.l1.forward(output=output, target=target)
    assert loss == expected_loss


test_l1_backward_cases = [
    (
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
        torch.Tensor([0, 0, 0, 0, 0, 0]).to(torch.int),
    ),
    (
        torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
        -torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
        -torch.Tensor([1, 1, 1, 1, 1, 1]).to(torch.int),
    ),
    (
        torch.Tensor([1, 1, -1, 1, 1, 1]).to(torch.int),
        torch.Tensor([1, -1, 1, 1, 1, 1]).to(torch.int),
        torch.Tensor([0, -1, 1, 0, 0, 0]).to(torch.int),
    ),
    (
        torch.Tensor([-1, -1, -1, 1, 1, 1]).to(torch.int),
        torch.Tensor([1, -1, 1, 1, 1, 1]).to(torch.int),
        torch.Tensor([1, 0, 1, 0, 0, 0]).to(torch.int),
    ),
]


@pytest.mark.parametrize('output, target, expected_grad', test_l1_backward_cases)
def test_l1_backward(output, target, expected_grad):
    grad = bnn.loss.l1.backward(output=output, target=target)
    torch.testing.assert_close(grad, expected_grad)
