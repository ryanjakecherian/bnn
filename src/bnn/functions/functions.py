import torch
import torch.autograd

__all__ = [
    'binarise',
    'ternarise',
]


def binarise(x: torch.Tensor, threshold: int = 0) -> torch.Tensor:
    out = torch.ones_like(x)
    out[x < threshold] = -1

    return out.to(torch.int)


def ternarise(
    x: torch.Tensor,
    threshold_lo: int = 0,
    threshold_hi: int = 0,
) -> torch.Tensor:
    """Ternarise Tensor, numbers on the threshold round up"""
    if threshold_hi < threshold_lo:
        raise ValueError('lo thresh cannot be larger than hi thresh!')

    out = torch.zeros_like(x)
    out[x >= threshold_hi] = 1
    out[x < threshold_lo] = -1

    return out
