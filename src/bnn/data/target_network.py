import typing

import torch

import bnn.functions
import bnn.network

from .data_loader import DataLoader, LabelledDatum

__all__ = [
    'target_network_factory',
]


class TargetNetwork(DataLoader):
    _target_network: bnn.network.TernBinNetwork

    def __init__(self, target_network: bnn.network.TernBinNetwork, *args, **kwargs):
        self._target_network = target_network
        super().__init__(*args, **kwargs)

    def _healthcheck(self):
        if self._datapoints <= 0:
            raise ValueError(f'Datapoints must be > 0, received {self._datapoints}')

        if self._batch_size <= 0:
            raise ValueError(f'Batch size must be > 0, received {self._batch_size}')

        if (not self._include_last_if_uneven) and (self._datapoints < self._batch_size):
            raise ValueError('Batch size < Datapoints and excluding last batch, so no data.')

        if self._datapoints < self._batch_size:
            raise Warning(f'Batch size ({self._batch_size}) > Datapoints ({self._datapoints}).')

    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size
        self._healthcheck()

    def __len__(self) -> int:
        return self._datapoints

    def __next__(self) -> LabelledDatum:
        # calculate next batch size
        size = self._get_next_batch_size()

        # random input
        input = bnn.functions.binarise(torch.randn(size, self._target_network.dims[0])).to(torch.int)
        # feed forward
        output = self._target_network.forward(input)
        # count iterations
        self._count_its(size)

        return LabelledDatum(target=output, input=input)

    def __iter__(self) -> typing.Generator[LabelledDatum, None, None]:
        self._reset_its()
        return self


def target_network_factory(
    datapoints: int,
    batch_size: int,
    target_network: bnn.network.TernBinNetwork,
    include_last_if_uneven: bool = False,
) -> TargetNetwork:
    TNDL = TargetNetwork(
        datapoints=datapoints,
        batch_size=batch_size,
        target_network=target_network,
        include_last_if_uneven=include_last_if_uneven,
    )
    return TNDL
