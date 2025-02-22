import torch

import bnn.type

__all__ = [
    'ExpectationSGD',
]


class ExpectationSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        if lr < 0:
            raise ValueError(f'Invalid lr: {lr}')

        defaults = dict(lr=lr)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self) -> tuple[float, list[int], list[int]]:
        # for metrics
        all_num_flips = []
        all_num_parameters = []

        for group in self.param_groups:
            lr = group['lr']

            for param in group['params']:
                if param.grad is None:
                    continue

                # aggregate number of flips
                num_flips = modal_sgd(param=param, lr=lr)
                num_parameters = torch.numel(param.data)

                all_num_flips.append(num_flips)
                all_num_parameters.append(num_parameters)

        # total prop flips
        prop_flipped = sum(all_num_flips) / sum(all_num_parameters)

        return prop_flipped, all_num_flips, all_num_parameters


def _expectation_sgd(
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
    lr_clipped_scaled_grad = torch.clamp_max(lr_scaled_grad, 1)

    # sign
    unsigned_flips = torch.bernoulli(lr_clipped_scaled_grad)
    signed_flips = unsigned_flips * grad_sign

    # only flip if it isn't saturated
    un_saturated = (signed_flips * param.data) == 1
    num_flipped = torch.sum(un_saturated)

    # torch.sign makes sure you can't nudge outside of {-1, 0, 1}
    param.data = torch.sign((param.data - signed_flips)).to(bnn.type.INTEGER)

    return num_flipped


def modal_sgd(
    param: torch.Tensor,
    lr: float,
) -> int:
    
    #the grad is actually made modal in the backward function, because thats where the k for top-k is available.

    num_flipped = torch.sum(param.grad.abs())

    # torch.sign makes sure you can't nudge outside of {-1, 0, 1}
    param.data = torch.sign((param.data - param.grad)).to(bnn.type.INTEGER)

    return num_flipped
