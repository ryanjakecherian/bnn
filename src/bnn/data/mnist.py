import pathlib
import typing

import torch
import torch.utils.data
import torchvision

import bnn.functions

from .data_loader import DataLoader, LabelledDatum

__all__ = [
    'MNISTDataLoader',
]


class MNISTDataLoader(DataLoader):
    _loader: torch.utils.data.DataLoader
    _iter: typing.Generator
    binarise_thresh: float

    def __init__(
        self,
        root: str | pathlib.Path,
        download: bool,
        train: bool,
        batch_size: int,
        binarise_thesh: float = 0.5,
        shuffle: bool = True,
        include_last_if_uneven: bool = False,
    ):
        dataset = torchvision.datasets.MNIST(
            root=str(root),
            train=train,
            download=download,
            transform=torchvision.transforms.ToTensor(),
        )

        self._loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.binarise_thresh = binarise_thesh

        super().__init__(
            datapoints=len(dataset),
            batch_size=batch_size,
            include_last_if_uneven=include_last_if_uneven,
        )

    def _healthcheck(self):
        return

    def __iter__(self) -> typing.Generator[LabelledDatum, None, None]:
        self._iter = iter(self._loader)
        return super().__iter__()

    def _next(self, size: int) -> LabelledDatum:
        image, label = next(self._iter)
        int_image = bnn.functions.binarise(image, threshold=self.binarise_thresh)
        return LabelledDatum(input=int_image, target=label)
