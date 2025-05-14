import pathlib
import typing

import torch
import torch.utils.data
import torchvision

import bnn.functions

from .data_loader import DataLoader, LabelledDatum

__all__ = [
    'MNIST01DataLoader',
]


class MNIST01DataLoader(DataLoader):
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
        binarise_thresh: float = 0.5,
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
        self._binarise_thresh = binarise_thresh

        super().__init__(
            datapoints=len(dataset),
            batch_size=batch_size,
            include_last_if_uneven=include_last_if_uneven,
        )

    def _healthcheck(self):
        return

    def __iter__(self) -> typing.Generator[LabelledDatum, None, None]:
        self._iter = iter(self._loader) #sets the _iter attribute to the iterator of the DataLoader object (torch.utils.data.DataLoader)
        return super().__iter__()   #calls the superclass DataLoader's __iter__ method, which if you recall, sets this object's _iteration to 0 and returns self.

    def _next(self, size: int) -> LabelledDatum:
        image, label = next(self._iter) #calls the next() method on the iterator of the DataLoader object (which returns the next item in the torch.utils.data.DataLoader object)
        # image has size (batchsize, 1, 28, 28) and label has size (batchsize)


        # convert input
        int_image = bnn.functions.binarise(image, threshold=self._binarise_thresh)
        int_image = int_image.reshape(-1, self.input_size) #reshapes the image tensor from (batchsize, 1, 28, 28) to (batchsize, 28*28)
        # convert label
        one_hot_label = torch.nn.functional.one_hot(label, num_classes=self.output_size) #converts the label tensor to a one-hot tensor in {0,1}
        
        #if we want {-1,1} network, then uncomment the following line
        # one_hot_label_rescaled = one_hot_label.to(torch.int) * 2 - 1

        return LabelledDatum(input=int_image.float(), target=one_hot_label.float()) #FIXME  TORCH FORCES ME TO MAKE IT A FLOAT OTHERWISE CANT GET FLOAT GRADS