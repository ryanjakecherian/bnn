import abc
import typing

import torch


class LabelledDatum(typing.NamedTuple):
    input: torch.Tensor
    target: torch.Tensor


class DataLoader(abc.ABC):
    _batch_size: int
    _include_last_if_uneven: int
    _datapoints: int
    _iteration: int

    def __init__(
        self,
        datapoints: int,
        batch_size: int,
        include_last_if_uneven: bool = False,
    ):
        # save
        self._datapoints = datapoints
        self._batch_size = batch_size
        self._include_last_if_uneven = include_last_if_uneven
        # init and healthcheck
        self._reset_its()
        self._healthcheck()

    @abc.abstractmethod
    def _healthcheck(self): ...

    @abc.abstractmethod
    def set_batch_size(self, batch_size: int): ...

    @abc.abstractmethod
    def __next__(self) -> LabelledDatum: ...

    @abc.abstractmethod
    def __iter__(self) -> typing.Generator[LabelledDatum, None, None]: ...

    def __len__(self) -> int:
        return self._datapoints

    def _reset_its(self):
        self._iteration = 0

    def _count_its(self, count: int):
        self._iteration += count

    def _get_next_batch_size(self) -> int:
        """Calculate next batch size from self._batch_size.

        Raises:
            StopIteration: Data exhausted.

        Returns:
            Int: Next batch size.
        """
        if self._iteration >= self._datapoints:
            raise StopIteration

        next_batch_size = min(self._batch_size, self._datapoints - self._iteration)

        # stop if uneven batch size and don't include last
        if (not self._include_last_if_uneven) and (next_batch_size != self.batch_size):
            raise StopIteration

        return next_batch_size
