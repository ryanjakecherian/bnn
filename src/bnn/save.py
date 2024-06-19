import pathlib
import pickle

from bnn.network import TernBinNetwork


def save_network(network: TernBinNetwork, filename: pathlib.Path):
    with open(filename, 'w') as f:
        pickle.dump(network, f)


def load_network(filename: pathlib.Path):
    with open(filename, 'r') as f:
        network = pickle.load(f)  # noqa: S301

    if not isinstance(network, TernBinNetwork):
        raise TypeError('File does not contain a TernBinNetwork!')

    return network
