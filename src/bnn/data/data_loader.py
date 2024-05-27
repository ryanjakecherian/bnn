import abc
import typing

import torch


class TrainingDatum(typing.NamedTuple):
    input: torch.Tensor
    target: torch.Tensor


class DataLoader(abc.ABC):
    batch_size: int

    def __init__(self, shuffle: bool, batch_size: int):
        self.shuffle = shuffle
        self.batch_size = batch_size

    @abc.abstractmethod
    def set_batch_size(self, batch_size: int): ...

    @abc.abstractmethod
    def __next__(self) -> TrainingDatum: ...

    @abc.abstractmethod
    def __iter__(self) -> typing.Generator[TrainingDatum, None, None]: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...
