import torch

import bnn.type

__all__ = [
    'DeterministicSGD',
]


class DeterministicSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        if lr < 0:
            raise ValueError(f'Invalid lr: {lr}')

        defaults = dict(lr=lr)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self) -> float:
        # for metrics
        total_flips = 0
        total_parameters = 0

        for group in self.param_groups:
            lr = group['lr']

            for param in group['params']:
                if param.grad is None:
                    continue

                # aggregate number of flips
                num_flips = _deterministic_sgd(param=param, lr=lr)
                num_parameters = torch.numel(param.data)

                total_flips += num_flips
                total_parameters += num_parameters

        # calc number of flips
        proportion_flipped = total_flips / total_parameters

        return proportion_flipped


def _deterministic_sgd(
    param: torch.Tensor,
    lr: float,
) -> int:
    # FIXME - currently going to assume symbols are {-1, 0, 1}...
    grad_sign = torch.sign(param.grad).to(bnn.type.INTEGER)
    grad_abs = torch.abs(param.grad).to(bnn.type.INTEGER)

    # lr = 0 nothing is trained
    # lr = 1 everything is towards the sign of its grad
    # lr in between - higher grad is more likely to be nudged
    lr_scaled_grad = grad_abs * lr

    # flip deterministically if flip prob > 0.5
    unsigned_flips = lr_scaled_grad >= 0.5
    signed_flips = unsigned_flips * grad_sign

    # only flip if it isn't saturated
    un_saturated = (signed_flips * param.data) == 1
    num_flipped = torch.sum(un_saturated)

    # torch.sign makes sure you can't nudge outside of {-1, 0, 1}
    param.data = torch.sign((param.data - signed_flips)).to(bnn.type.INTEGER)

    return num_flipped
