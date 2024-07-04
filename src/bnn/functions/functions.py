import torch
import torch.autograd

import bnn.type

__all__ = [
    'binarise',
    'ternarise',
    'int_matmul',
    'one_hot_argmax',
]


def binarise(x: torch.Tensor, threshold: int = 0) -> torch.Tensor:
    out = torch.ones_like(x)
    out[x < threshold] = -1

    return out.to(bnn.type.INTEGER)


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


TORCH_FLOAT_TYPE = torch.float16


# HACK - this is technically "unsafe", but should be fine for reasonable layer sizes!
def int_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    AB_float = torch.matmul(A.to(TORCH_FLOAT_TYPE), B.to(TORCH_FLOAT_TYPE))
    AB_int = torch.round(AB_float).to(bnn.type.INTEGER)
    return AB_int


def one_hot_argmax(x: torch.Tensor) -> torch.Tensor:
    argmax = torch.argmax(x, dim=-1, keepdim=True)

    # empty array
    out = torch.full_like(x, fill_value=-1)
    # add 1s at argmax
    out.scatter_(dim=-1, index=argmax, value=1)
    return out
