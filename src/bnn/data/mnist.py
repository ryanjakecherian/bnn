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
    _binarise_thresh: float
    input_size: int = 28 * 28
    output_size: int = 10

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
        self._binarise_thresh = binarise_thesh

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

        # convert input
        int_image = bnn.functions.binarise(image, threshold=self._binarise_thresh)
        int_image = int_image.reshape(-1, self.input_size)
        # convert label
        one_hot_label = torch.nn.functional.one_hot(label, num_classes=self.output_size).to(bool)

        return LabelledDatum(input=int_image, target=one_hot_label)
