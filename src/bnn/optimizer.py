import torch


class ExpectationSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        if lr < 0:
            raise ValueError(f'Invalid lr: {lr}')

        defaults = dict(lr=lr)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                expectation_sgd(group)

        return loss


def expectation_sgd(params: list[torch.Tensor]) -> None:
    raise NotImplementedError
