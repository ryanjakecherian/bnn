import abc
import typing

import torch


@typing.NamedTuple
class TrainingDatum:
    input: torch.Tensor
    target: torch.Tensor


class DataLoader(abc.ABC):
    shuffle: bool
    batch_size: int

    def __init__(self, shuffle: bool, batch_size: int, name: str):
        self.shuffle = shuffle
        self.batch_size = batch_size

    @abc.abstractmethod
    def __next__(self) -> TrainingDatum: ...

    @abc.abstractmethod
    def __iter__(self) -> typing.Generator[TrainingDatum]: ...
