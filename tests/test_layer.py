import bnn.layer
import pytest
import torch


@pytest.fixture
def get_layer_with_W():
    def get_layer_with_W_(W: torch.Tensor) -> bnn.layer.TernBinLayer:
        input_dim, output_dim = W.shape

        layer = bnn.layer.TernBinLayer(input_dim=input_dim, output_dim=output_dim)
        layer.W.data = W.clone()
        layer.W.grad = None

        return layer

    return get_layer_with_W_


test_get_layer_with_W_cases = [
    torch.ones(100, 100, dtype=torch.int),
    torch.ones(200, 100, dtype=torch.int),
    torch.zeros(100, 200, dtype=torch.int),
    (torch.rand(100, 200) * 10).to(torch.int),
]


@pytest.mark.parametrize('W', test_get_layer_with_W_cases)
def test_get_layer_with_W(W: torch.Tensor, get_layer_with_W):
    layer: bnn.layer.TernBinLayer = get_layer_with_W(W)

    assert torch.allclose(layer.W.data, W)
    assert layer.W.grad is None


test_forward_cases = [
    (
        torch.ones(100, 100, dtype=torch.int),
        torch.ones(100, dtype=torch.int),
        torch.ones(100, dtype=torch.int),
    ),
    (
        torch.ones(200, 100, dtype=torch.int),
        torch.ones(200, dtype=torch.int),
        torch.ones(100, dtype=torch.int),
    ),
    (
        torch.zeros(200, 100, dtype=torch.int),
        torch.ones(200, dtype=torch.int),
        torch.ones(100, dtype=torch.int),
    ),
    (
        torch.ones(200, 100, dtype=torch.int),
        torch.ones(10, 200, dtype=torch.int),
        torch.ones(10, 100, dtype=torch.int),
    ),
    (
        torch.Tensor([[1, 0, -1]] * 5).to(torch.int).T,
        torch.Tensor([1, -1, 1]).to(torch.int),
        torch.ones(5, dtype=torch.int),
    ),
    (
        torch.Tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]]).to(torch.int).T,
        torch.Tensor([[1, 1], [-1, -1]]).to(torch.int),
        torch.Tensor([[1, 1, 1, -1], [-1, 1, 1, 1]]).to(torch.int),
    ),
]


@pytest.mark.parametrize('W, x, expected_out', test_forward_cases)
def test_forward(W, x, expected_out, get_layer_with_W):
    layer: bnn.layer.TernBinLayer = get_layer_with_W(W)
    out = layer.forward(x)

    assert torch.allclose(out, expected_out)


test_backward_cases = []


@pytest.mark.parametrize('W, grad, activation, expected_W_grad', test_backward_cases)
def test_backward(W, grad, activation, expected_W_grad): ...
