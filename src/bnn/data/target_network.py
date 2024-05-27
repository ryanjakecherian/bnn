import typing

import torch

import bnn.functions
import bnn.network

from .data_loader import DataLoader, TrainingDatum


class TargetNetwork(DataLoader):
    target_network: bnn.network.TernBinNetwork

    def __init__(
        self,
        batch_size: int,
        target_network: bnn.network.TernBinNetwork,
    ):
        self.target_network = target_network
        self.batch_size = batch_size

    def __next__(self) -> tuple[TrainingDatum]:
        # random input
        input = torch.randn(self.batch_size, self.target_network.dims[0])
        # feed forward
        output = self.target_network.forward(input)
        return output, input

    def __iter__(self) -> typing.Generator[TrainingDatum]:
        return self
