import torch

import bnn.functions

from .data_loader import DataLoader, LabelledDatum


class AllUnaryFunctionsDataLoader(DataLoader):
    _input_dim: int
    _datum: LabelledDatum

    def __init__(
        self,
        input_dim: int,
        datapoints: int,
        batch_size: int,
        include_last_if_uneven: bool = False,
    ):
        self._input_dim = input_dim
        super().__init__(
            datapoints=datapoints,
            batch_size=batch_size,
            include_last_if_uneven=include_last_if_uneven,
        )
        self._generate_datum()

    def _generate_datum(self):
        input = bnn.functions.binarise(torch.randn(self._datapoints, self._input_dim)).to(torch.int)
        ones_like_input = torch.ones_like(input)
        target = torch.concat(
            tensors=[input, -input, ones_like_input, -ones_like_input],
            dim=-1,
        )

        self._datum = LabelledDatum(input=input, target=target)

    def _healthcheck(self):
        pass

    def _next(self, size: int) -> LabelledDatum:
        input = self._datum.input[:size]
        target = self._datum.target[:size]
        return LabelledDatum(input=input, target=target)
