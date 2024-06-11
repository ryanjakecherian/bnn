import abc
import typing

import torch

__all__ = [
    'LabelledDatum',
    'DataLoader',
]


class LabelledDatum(typing.NamedTuple):
    input: torch.Tensor
    target: torch.Tensor


def _place_labelled_datum_on_device(
    labelled_datum: LabelledDatum,
    device: torch.device,
):
    return LabelledDatum(
        input=labelled_datum.input.to(device),
        target=labelled_datum.target.to(device),
    )


class DataLoader(abc.ABC):
    _batch_size: int
    _include_last_if_uneven: int
    _datapoints: int
    _iteration: int
    _device: torch.device | None = None

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
    def _next(self, size: int) -> LabelledDatum: ...

    def to(self, device: torch.device):
        self._device = device

    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def __len__(self) -> int:
        whole_batches = self._datapoints // self._batch_size

        last_uneven = whole_batches * self._batch_size != self._datapoints
        if self._include_last_if_uneven and last_uneven:
            num_batches = whole_batches + 1
        else:
            num_batches = whole_batches

        return num_batches

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
        if (not self._include_last_if_uneven) and (next_batch_size != self._batch_size):
            raise StopIteration

        return next_batch_size

    def __next__(self) -> LabelledDatum:
        size = self._get_next_batch_size()
        self._count_its(size)
        datum_on_device = _place_labelled_datum_on_device(labelled_datum=self._next(size), device=self._device)
        return datum_on_device

    def __iter__(self) -> typing.Generator[LabelledDatum, None, None]:
        self._reset_its()
        return self
