import torch


class ExpectationSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        if lr < 0:
            raise ValueError(f'Invalid lr: {lr}')

        defaults = dict(lr=lr)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group['lr']

            for param in group['params']:
                if param.grad is None:
                    continue

                expectation_sgd(param=param, lr=lr)

        return


def expectation_sgd(
    param: torch.Tensor,
    lr: float,
) -> None:
    # FIXME - currently going to assume symbols are {-1, 0, 1}...
    grad_sign = torch.sign(param.grad).to(torch.int)
    grad_abs = torch.abs(param.grad).to(torch.int)

    # lr = 0 nothing is trained
    # lr = 1 everything is towards the sign of its grad
    # lr in between - higher grad is more likely to be nudged
    lr_scaled_grad = grad_abs * lr
    lr_clipped_scaled_grad = torch.clamp_max(lr_scaled_grad, 1)

    # sign
    unsigned_flips = torch.bernoulli(lr_clipped_scaled_grad)
    signed_flips = unsigned_flips * grad_sign

    # torch.sign makes sure you can't nudge outside of {-1, 0, 1}
    param.data = torch.sign(param.data - signed_flips).to(torch.int)

    return
