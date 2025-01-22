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
        # print(len(self.param_groups)) #debug

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self) -> tuple[float, list[int], list[int]]:
        # for metrics
        all_num_flips = []
        all_num_parameters = []

        for group in self.param_groups:
            lr = group['lr']
            
            i = 0                                                # debug
            for param in group['params']:
                if param.grad is None:
                    continue
                print(f'optimising layer: {i}')                                    #debug
                # aggregate number of flips
                num_flips = _expectation_sgd_ryan(param=param, lr=lr)
                print(f'number of weight flips: {num_flips.sum()}')                                  #debug
                num_parameters = torch.numel(param.data)

                all_num_flips.append(num_flips)
                all_num_parameters.append(num_parameters)
                i += 1                                              #debug

            

        # total prop flips
        prop_flipped = sum(all_num_flips) / sum(all_num_parameters)

        return prop_flipped, all_num_flips, all_num_parameters


def _expectation_sgd_bernardo(
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
    un_saturated = (signed_flips * param.data) == 1 #because if param.data = 1 then it will only change if signed_flips = 1. also if param.data = -1, then it will only change if signed_flips = -1. Note that in both cases, the product is 1. therefore only these weights are unsaturated.
    num_flipped = torch.sum(un_saturated)

    # torch.sign makes sure you can't nudge outside of {-1, 0, 1}
    param.data = torch.sign((param.data - signed_flips)).to(bnn.type.INTEGER)

    return num_flipped



def _expectation_sgd_ryan(
    param: torch.Tensor,
    lr: float,
) -> int:
    
    #debug print statements:
    if param.grad.abs().sum() == 0:
        print("param gradient all zeros")  #this is printing?? only after a few batch updates. so the network is training to output zero...
    else:
        print(f'param gradient not all zeros')

    
    
    
    param_grad = param.grad

    grad_sign = torch.sign(param_grad)
    grad_max = torch.max(torch.abs(param_grad))
    if grad_max != 0:
        grad_importance = lr*torch.abs(param_grad)/grad_max  #have temporarily removed safety mechanism (  torch.clamp_max ( , 1)  )
    else:
        grad_importance = torch.zeros_like(param_grad)

    update = grad_sign*torch.bernoulli(grad_importance)
    

                                                                                                # wait what if we let the weights update out of -1,0,1 and then re-quantise the weights after (not with sign)??
                                                                                                # this would allow network to increase the magnitude of one weight by decreasing the magnitude of other weights...?


    
    unsaturated = (param.data * update) == 1    #because if param.data = 1 then it will only change if update = 1. also if param.data = -1, then it will only change if update = -1. Note that in both cases, the product is 1. therefore only these weights are unsaturated.
    num_flipped = torch.sum(unsaturated)    

    
    temp = param.data - update
    temp = torch.round(temp)
    param.data = torch.clamp(temp, max=1, min=-1).to(torch.float)  # torch.sign makes sure you can't nudge outside of {-1, 0, 1}
    

    #note this cannot be fixed without refactoring the code, whilst we have floating point gradients
    #for future: FIXME IM ANNOYED becuase the .to(float) is is a hack to make sure that W.grad is float, even though making W int sohuldnt affect W.grad!!!!! fuckign pytorch.

    return num_flipped