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
    _its: int

    def __init__(self, shuffle: bool, batch_size: int):
        self.shuffle = shuffle
        self.batch_size = batch_size

    @abc.abstractmethod
    def set_batch_size(self, batch_size: int): ...

    @abc.abstractmethod
    def __next__(self) -> LabelledDatum: ...

    @abc.abstractmethod
    def __iter__(self) -> typing.Generator[LabelledDatum, None, None]: ...

    def __len__(self) -> int:
        return self._datapoints

    def _get_next_batch_size(self) -> int:
        """Calculate next batch size from self._batch_size.

        Raises:
            StopIteration: Data exhausted.

        Returns:
            Int: Next batch size.
        """
        if self._its >= self._datapoints:
            raise StopIteration

        next_batch_size = min(self._batch_size, self._datapoints - self._its)

        # stop if uneven batch size and don't include last
        if (not self._include_last_if_uneven) and (next_batch_size != self.batch_size):
            raise StopIteration

        return next_batch_size
