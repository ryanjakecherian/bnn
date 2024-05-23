import dataclasses

import bnn.functions
import bnn.layer
import bnn.loss
import bnn.network
import pytest
import torch


class RandomTestData:
    num_samples: int = 100
    input_dim: int
    output_dim: int
    input: torch.Tensor
    target: torch.Tensor

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = bnn.functions.binarise(torch.randn(self.num_samples, input_dim))
        self.target = bnn.functions.binarise(torch.randn(self.num_samples, output_dim))

        return


random_input_output_sizes = [
    {
        'input_dim': 16,
        'output_dim': 16,
    },
    {
        'input_dim': 32,
        'output_dim': 32,
    },
    {
        'input_dim': 16,
        'output_dim': 32,
    },
]


@pytest.fixture(params=random_input_output_sizes)
def random_data(request):
    return RandomTestData(**request.param)


def test_random_data(random_data: RandomTestData):
    assert random_data.input.dtype is torch.int
    assert random_data.target.dtype is torch.int

    assert random_data.input_dim == random_data.input.shape[-1]
    assert random_data.output_dim == random_data.target.shape[-1]

    assert random_data.num_samples == len(random_data.input)
    assert random_data.num_samples == len(random_data.target)


@pytest.fixture
def get_network():
    def get_network_(*dims: list[int]) -> bnn.network.TernBinNetwork:
        return bnn.network.TernBinNetwork(*dims)

    return get_network_


def test_get_network(get_network):
    network = get_network(16, 32, 64, 32, 16)
    assert isinstance(network, bnn.network.TernBinNetwork)
    return


random_hidden_layer_dims = [
    [],
    [16, 16],
    [16, 32, 32, 16],
    [16, 32, 64, 32, 16],
]


@pytest.fixture(params=random_hidden_layer_dims)
def get_random_network(request, get_network, random_data):
    random_dims = request.param

    def get_random_network_(input_dim, output_dim):
        return get_network(random_data.input_dim, *random_dims, random_data.output_dim)

    return get_random_network_


@dataclasses.dataclass
class RandomDataAndNetwork:
    random_data: RandomTestData
    network: bnn.network.TernBinNetwork


@pytest.fixture
def random_data_and_network(random_data, get_random_network) -> RandomDataAndNetwork:
    network = get_random_network(random_data.input_dim, random_data.output_dim)
    return RandomDataAndNetwork(random_data=random_data, network=network)


def test_integration_forwards_and_backwards_random_data(
    random_data_and_network: RandomDataAndNetwork,
):
    random_data = random_data_and_network.random_data
    network = random_data_and_network.network

    # forward pass
    output = network.forward(random_data.input)
    assert output.shape == random_data.target.shape

    # loss forward pass
    loss = bnn.loss.l1.forward(
        output=output,
        target=random_data.target,
    )
    loss_grad = bnn.loss.l1.backward(
        output=output,
        target=random_data.target,
    )
    assert loss.nelement() == 1
    assert loss_grad.shape == random_data.target.shape

    # loss backwards pass
    input_grad = network.backward(grad=loss_grad)
    assert input_grad.shape == random_data.input.shape


test_integration_forwards_and_backwards_set_data_cases = [
    # all ones
    (
        # Ws
        [
            torch.ones(10, 20, dtype=torch.int),
            torch.ones(20, 5, dtype=torch.int),
        ],
        # x
        torch.ones(10, dtype=torch.int),
        # expected_input
        [
            torch.ones(10, dtype=torch.int),
            torch.ones(20, dtype=torch.int),
        ],
        # expected_out
        torch.ones(5, dtype=torch.int),
        # grad
        torch.ones(5, dtype=torch.int),
        # expected_grad_out
        torch.ones(10, dtype=torch.int),
        # expected_grad_Ws
        [
            torch.ones(10, 20, dtype=torch.int),
            torch.ones(20, 5, dtype=torch.int),
        ],
    ),
    # all ones, some negative
    (
        # Ws
        [
            torch.ones(10, 20, dtype=torch.int),
            torch.ones(20, 5, dtype=torch.int),
        ],
        # x
        torch.ones(10, dtype=torch.int),
        # expected_input
        [
            torch.ones(10, dtype=torch.int),
            torch.ones(20, dtype=torch.int),
        ],
        # expected_out
        torch.ones(5, dtype=torch.int),
        # grad
        -torch.ones(5, dtype=torch.int),
        # expected_grad_out
        -torch.ones(10, dtype=torch.int),
        # expected_grad_Ws
        [
            -torch.ones(10, 20, dtype=torch.int),
            -torch.ones(20, 5, dtype=torch.int),
        ],
    ),
    # all ones, some negative
    (
        # Ws
        [
            -torch.ones(10, 20, dtype=torch.int),
            torch.ones(20, 5, dtype=torch.int),
        ],
        # x
        torch.ones(10, dtype=torch.int),
        # expected_input
        [
            torch.ones(10, dtype=torch.int),
            -torch.ones(20, dtype=torch.int),
        ],
        # expected_out
        -torch.ones(5, dtype=torch.int),
        # grad
        torch.ones(5, dtype=torch.int),
        # expected_grad_out
        -torch.ones(10, dtype=torch.int),
        # expected_grad_Ws
        [
            torch.ones(10, 20, dtype=torch.int),
            -torch.ones(20, 5, dtype=torch.int),
        ],
    ),
]


@pytest.mark.parametrize(
    'Ws, x, expected_input, expected_out, grad, expected_grad_out, expected_grad_Ws',
    test_integration_forwards_and_backwards_set_data_cases,
)
def test_integration_forwards_and_backwards_set_data(
    Ws,
    x,
    expected_input,
    expected_out,
    grad,
    expected_grad_out,
    expected_grad_Ws,
    get_network,
):
    # assemble network
    dims = [len(W) for W in Ws] + [len(expected_out)]
    network: bnn.network.TernBinNetwork = get_network(*dims)

    for W, layer in zip(Ws, network.layers.values()):
        layer: bnn.layer.TernBinLayer
        layer.W.data = W

    # forward
    out = network.forward(x)

    # backward
    grad_out = network.backward(grad)

    # assert forward
    torch.testing.assert_close(out, expected_out)
    for inp, expected_inp in zip(network.input.values(), expected_input):
        torch.testing.assert_close(inp, expected_inp)

    # assert backward
    torch.testing.assert_close(grad_out, expected_grad_out)
    for layer, expected_grad_W in zip(network.layers.values(), expected_grad_Ws):
        torch.testing.assert_close(layer.W.grad, expected_grad_W)
