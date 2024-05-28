import torch

import bnn.functions
import bnn.network

from .data_loader import DataLoader, LabelledDatum

__all__ = [
    'TargetNetworkDataLoader',
]


class TargetNetworkDataLoader(DataLoader):
    _target_network: bnn.network.TernBinNetwork

    def __init__(
        self,
        target_network: bnn.network.TernBinNetwork,
        datapoints: int,
        batch_size: int,
        W_mean: float = 0,
        W_zero_prob: float = 0.8,
        include_last_if_uneven: bool = False,
    ):
        self._target_network = target_network
        self._target_network._initialise(W_mean=W_mean, W_zero_prob=W_zero_prob)
        super().__init__(
            datapoints=datapoints,
            batch_size=batch_size,
            include_last_if_uneven=include_last_if_uneven,
        )

    def _healthcheck(self):
        if self._datapoints <= 0:
            raise ValueError(f'Datapoints must be > 0, received {self._datapoints}')

        if self._batch_size <= 0:
            raise ValueError(f'Batch size must be > 0, received {self._batch_size}')

        if (not self._include_last_if_uneven) and (self._datapoints < self._batch_size):
            raise ValueError('Batch size < Datapoints and excluding last batch, so no data.')

        if self._datapoints < self._batch_size:
            raise Warning(f'Batch size ({self._batch_size}) > Datapoints ({self._datapoints}).')

        assert bnn.network.network_params_al_ternary(self._target_network)

    def _next(self, size: int) -> LabelledDatum:
        # random input
        input = bnn.functions.binarise(torch.randn(size, self._target_network.dims[0])).to(torch.int)
        # feed forward
        output = self._target_network.forward(input)

        return LabelledDatum(target=output, input=input)
