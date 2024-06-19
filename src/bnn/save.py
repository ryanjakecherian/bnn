import copy
import os
import pathlib
import pickle

from bnn.network import TernBinNetwork

__all__ = (
    'load_network',
    'save_network',
)


def save_network(network: TernBinNetwork, filename: pathlib.Path):
    if os.path.exists(filename):
        raise FileExistsError(f'{filename} already exists!')

    _make_dir_if_doesnt_exist(filename.parent)

    # reset network
    network = copy.deepcopy(network)
    network.zero_grad()
    for key in network.input.keys():
        network.input[key] = None
    for key in network.grad.keys():
        network.grad[key] = None

    with open(filename, 'wb') as f:
        pickle.dump(network.to('cpu'), f)

    return


def load_network(filename: pathlib.Path):
    with open(filename, 'rb') as f:
        network = pickle.load(f)  # noqa: S301

    if not isinstance(network, TernBinNetwork):
        raise TypeError('File does not contain a TernBinNetwork!')

    return network


def _make_dir_if_doesnt_exist(dir: pathlib.Path):
    if os.path.exists(dir):
        if not os.path.isdir(dir):
            raise FileExistsError(f'{dir} exists and is a file!')

    else:
        dir.mkdir(parents=True)

    return
