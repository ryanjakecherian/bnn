import typing

import torch

import bnn.functions
import bnn.network

from .data_loader import DataLoader, LabelledDatum


class TargetNetwork(DataLoader):
    _target_network: bnn.network.TernBinNetwork
    _include_last_if_uneven: bool
    _datapoints: int
    _iteration: int

    def __init__(
        self,
        datapoints: int,
        batch_size: int,
        target_network: bnn.network.TernBinNetwork,
        include_last_if_uneven: bool = False,
    ):
        self._target_network = target_network
        self.batch_size = batch_size
        self._include_last_if_uneven = include_last_if_uneven
        self._datapoints = datapoints

        self._reset_its()
        self._healthcheck()

    def _healthcheck(self):
        if self._datapoints <= 0:
            raise ValueError(f'Datapoints must be > 0, received {self._datapoints}')

        if self.batch_size <= 0:
            raise ValueError(f'Batch size must be > 0, received {self.batch_size}')

        if (not self._include_last_if_uneven) and (self._datapoints < self.batch_size):
            raise ValueError('Batch size < Datapoints and excluding last batch, so no data.')

        if self._datapoints < self.batch_size:
            raise Warning(f'Batch size ({self.batch_size}) > Datapoints ({self._datapoints}).')

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        self._healthcheck()

    def _reset_its(self):
        self._iteration = 0

    def _count_its(self, count: int):
        self._iteration += count

    def __len__(self) -> int:
        return self._datapoints

    def __next__(self) -> LabelledDatum:
        if self._iteration >= self._datapoints:
            raise StopIteration

        # calculate size
        size = min(self.batch_size, self._datapoints - self._iteration)

        # stop if uneven batch size and don't include last
        if (not self._include_last_if_uneven) and (size != self.batch_size):
            raise StopIteration

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
    return TargetNetwork(
        datapoints=datapoints,
        batch_size=batch_size,
        target_network=target_network,
        include_last_if_uneven=include_last_if_uneven,
    )
