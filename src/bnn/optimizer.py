import torch


class ExpectationSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        if lr < 0:
            raise ValueError(f'Invalid lr: {lr}')

        defaults = dict(lr=lr)

        super().__init__(params, defaults)

    def step(self, closure=None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # TODO implement me!
        ...

        raise NotImplementedError

        return loss
