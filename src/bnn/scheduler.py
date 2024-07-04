import torch


def MultiplicativeLRFactory(alpha: float, *args, **kwargs) -> torch.optim.lr_scheduler.MultiplicativeLR:
    def lr_lambda(*args, **kwargs):
        return alpha

    return torch.optim.lr_scheduler.MultiplicativeLR(lr_lambda=lr_lambda, *args, **kwargs)
