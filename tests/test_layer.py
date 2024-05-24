import bnn.functions
import bnn.layer
import pytest
import torch


@pytest.fixture
def get_layer_with_W():
    def get_layer_with_W_(W: torch.Tensor) -> bnn.layer.TernBinLayer:
        input_dim, output_dim = W.shape

        layer = bnn.layer.TernBinLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            forward_func=bnn.functions.forward.SignBinarise(),
            backward_func=bnn.functions.backward.SignTernarise(),
        )
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

    torch.testing.assert_close(layer.W.data, W)
    assert layer.W.grad is None


test_forward_cases = [
    # all ones, square
    (
        torch.ones(100, 100, dtype=torch.int),
        torch.ones(100, dtype=torch.int),
        torch.ones(100, dtype=torch.int),
    ),
    # all ones, rectangle
    (
        torch.ones(200, 100, dtype=torch.int),
        torch.ones(200, dtype=torch.int),
        torch.ones(100, dtype=torch.int),
    ),
    # zero weights * ones
    (
        torch.zeros(200, 100, dtype=torch.int),
        torch.ones(200, dtype=torch.int),
        torch.ones(100, dtype=torch.int),
    ),
    # multiple samples, all ones
    (
        torch.ones(200, 100, dtype=torch.int),
        torch.ones(10, 200, dtype=torch.int),
        torch.ones(10, 100, dtype=torch.int),
    ),
    # random example
    (
        torch.Tensor([[1, 0, -1]] * 5).to(torch.int).T,
        torch.Tensor([1, -1, 1]).to(torch.int),
        torch.ones(5, dtype=torch.int),
    ),
    # multiple different samples, different outputs in each node
    (
        torch.Tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]]).to(torch.int).T,
        torch.Tensor([[1, 1], [-1, -1]]).to(torch.int),
        torch.Tensor([[1, 1, 1, -1], [-1, 1, 1, 1]]).to(torch.int),
    ),
]


# TODO refactor test to match code refactor
@pytest.mark.parametrize('W, x, expected_out', test_forward_cases)
def test_forward(W, x, expected_out, get_layer_with_W):
    layer: bnn.layer.TernBinLayer = get_layer_with_W(W)
    out = layer.forward(x)

    torch.testing.assert_close(out, expected_out)


test_backward_cases = [
    # W, g=0, a=rand
    (
        # W
        torch.zeros(2, 3, dtype=torch.int),
        # grad
        torch.zeros(3, dtype=torch.int),
        # activation
        torch.Tensor([-1, 1]).to(torch.int),
        # expected_out_grad
        torch.zeros(2, dtype=torch.int),
        # expected_W_grad
        torch.zeros(2, 3, dtype=torch.int),
    ),
    # W=0, g=1, a=1
    (
        # W
        torch.zeros(2, 3, dtype=torch.int),
        # grad
        torch.ones(3, dtype=torch.int),
        # activation
        torch.ones(2, dtype=torch.int),
        # expected_out_grad
        torch.zeros(2, dtype=torch.int),
        # expected_W_grad
        torch.ones(2, 3, dtype=torch.int),
    ),
    # W=0, g=-1, a=1
    (
        # W
        torch.zeros(2, 3, dtype=torch.int),
        # grad
        -torch.ones(3, dtype=torch.int),
        # activation
        torch.ones(2, dtype=torch.int),
        # expected_out_grad
        torch.zeros(2, dtype=torch.int),
        # expected_W_grad
        -torch.ones(2, 3, dtype=torch.int),
    ),
    # W=0, g=-1, a=-1
    (
        # W
        torch.zeros(2, 3, dtype=torch.int),
        # grad
        -torch.ones(3, dtype=torch.int),
        # activation
        -torch.ones(2, dtype=torch.int),
        # expected_out_grad
        torch.zeros(2, dtype=torch.int),
        # expected_W_grad
        torch.ones(2, 3, dtype=torch.int),
    ),
    # W=0, g=1, a=-1
    (
        # W
        torch.zeros(2, 3, dtype=torch.int),
        # grad
        torch.ones(3, dtype=torch.int),
        # activation
        -torch.ones(2, dtype=torch.int),
        # expected_out_grad
        torch.zeros(2, dtype=torch.int),
        # expected_W_grad
        -torch.ones(2, 3, dtype=torch.int),
    ),
    # W=1, g=1, a=1
    (
        # W
        torch.ones(2, 3, dtype=torch.int),
        # grad
        torch.ones(3, dtype=torch.int),
        # activation
        torch.ones(2, dtype=torch.int),
        # expected_out_grad
        torch.ones(2, dtype=torch.int),
        # expected_W_grad
        torch.ones(2, 3, dtype=torch.int),
    ),
    # W=1, g=-1, a=1
    (
        # W
        torch.ones(2, 3, dtype=torch.int),
        # grad
        -torch.ones(3, dtype=torch.int),
        # activation
        torch.ones(2, dtype=torch.int),
        # expected_out_grad
        -torch.ones(2, dtype=torch.int),
        # expected_W_grad
        -torch.ones(2, 3, dtype=torch.int),
    ),
    # W=all, g=all, a=all
    (
        # W
        torch.Tensor([[1, 0, -1], [-1, 0, 1]]).to(torch.int),
        # grad
        torch.Tensor([-1, 0, 1]).to(torch.int),
        # activation
        torch.Tensor([1, -1]).to(torch.int),
        # expected_out_grad
        torch.Tensor([-1, 1]).to(torch.int),
        # expected_W_grad
        torch.Tensor([[-1, 0, 1], [1, 0, -1]]).to(torch.int),
    ),
    # multi-sample...!
    (
        # W
        torch.Tensor([[1, 0, -1], [-1, 0, 1]]).to(torch.int),
        # grad
        torch.Tensor([[-1, 0, 1]] * 4).to(torch.int),
        # activation
        torch.Tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]]).to(torch.int),
        # expected_out_grad
        torch.Tensor([[-1, 1]] * 4).to(torch.int),
        # expected_W_grad
        torch.Tensor([[0, 0, 0], [0, 0, 0]]).to(torch.int),
    ),
    # multi-sample...!
    (
        # W
        torch.Tensor([[1, 0, -1], [-1, 0, 1]]).to(torch.int),
        # grad
        torch.Tensor([[-1, 1, 1], [1, 0, -1], [1, 1, 1], [-1, -1, -1]]).to(torch.int),
        # activation
        torch.Tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]]).to(torch.int),
        # expected_out_grad
        torch.Tensor([[-1, 1], [1, -1], [0, 0], [0, 0]]).to(torch.int),
        # expected_W_grad
        torch.Tensor([[0, 1, 0], [-4, -1, 0]]).to(torch.int),
    ),
]


# TODO refactor test to match code refactor
@pytest.mark.parametrize('W, grad, activation, expected_out_grad, expected_W_grad', test_backward_cases)
def test_backward(W, grad, activation, expected_out_grad, expected_W_grad, get_layer_with_W):
    layer: bnn.layer.TernBinLayer = get_layer_with_W(W)

    out_grad = layer.backward(grad, activation)

    torch.testing.assert_close(out_grad, expected_out_grad)
    torch.testing.assert_close(layer.W.grad, expected_W_grad)
