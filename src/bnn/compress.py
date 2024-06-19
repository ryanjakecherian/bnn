from typing import NamedTuple

import numpy as np

import bnn.functions
import bnn.network

__all__ = ('compress_network',)


class TBNNSchema(NamedTuple):
    dims: list[int]
    forward_func: list[bnn.functions.ForwardFunc]
    backward_func: list[bnn.functions.BackwardFunc]


def get_schema(network: bnn.network.TernBinNetwork) -> TBNNSchema:
    dims = network.dims
    forward_func = [layer.forward_func for layer in network.layers.values()]
    backward_func = [layer.backward_func for layer in network.layers.values()]

    return TBNNSchema(
        dims=dims,
        forward_func=forward_func,
        backward_func=backward_func,
    )


def compress_network(network: bnn.network.TernBinNetwork) -> dict[str, np.ndarray]:
    Ws = {}
    for name, layer in network.layers.items():
        Ws[name] = layer.W.cpu().numpy()

    bWs = {}
    for name, W in Ws.items():
        pos, neg = _split_ternary_into_two_binary(W)
        bWs[name + '_pos'] = pos
        bWs[name + '_neg'] = neg

    return bWs


def make_network_from_schema_and_bWs(
    schema: TBNNSchema,
    bWs: dict[str, np.ndarray],
) -> bnn.network.TernBinNetwork:
    network = bnn.network.TernBinNetwork(
        dims=schema.dims,
        forward_func=schema.forward_func,
        backward_func=schema.backward_func,
    )

    for name, layer in network.layers.items():
        layer.W[...] = 0
        layer.W[bWs[name + '_pos']] = 1
        layer.W[bWs[name + '_neg']] = -1

    return network


def _split_ternary_into_two_binary(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _assert_ternary(array)

    pos = array == 1
    neg = array == -1

    return pos, neg


def _assert_ternary(array: np.ndarray):
    is_one = array == 1
    is_zero = array == 0
    is_neg_one = array == -1

    assert np.all(is_one | is_zero | is_neg_one)
