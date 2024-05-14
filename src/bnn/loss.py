import torch


def number_incorrect(
    output: torch.Tensor,
    label: torch.Tensor,
) -> int:
    incorrect = torch.abs(output - label)
    loss = incorrect.sum()

    return loss
