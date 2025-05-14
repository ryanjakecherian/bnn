import torch

import bnn.type

__all__ = [
    'ExpectationSGD',
]



class ExpectationSGD(torch.optim.Optimizer):
    
    def __init__(self, params, log_lr: float, log_decrease_rate: float = 0, last_layer_float: bool = False):
        
        #error catching
        if 10**log_lr < 0:
            raise ValueError(f'Invalid lr: {10**log_lr}')

        #create lr dict
        if log_decrease_rate != 0:
            #log_decrease_rate is just such that the lr = 10^(log_lr - rate*i) where i is reversed layer number  (e.g. layer L would be i=0, and L-1 would have i=1, etc...)
            lrs = {}
            for i in range(len(params)):
                key = 'layers.TernBinLayer' + str(i) + '.W'
                lrs[key] = 10**(log_lr - ((len(params)-1 - i) * log_decrease_rate ) )   #s.t. the first layer has the lowest lr and the last layer has the highest lr
        
        else:
            lrs = {'layers.TernBinLayer' + str(i) + '.W': 10**log_lr for i in range(len(params))}

        #create dict of param_groups with their lrs
        param_groups = [dict(params=[p], lr=lrs[name]) for (name, p) in params.items() if name in lrs]

        #init optimizer with params and their lrs
        super().__init__(param_groups, defaults={'lr':10**log_lr})

        #set last layer float
        self.last_layer_float = last_layer_float




    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self) -> tuple[float, list[int], list[int]]:
        # for metrics
        all_num_flips = []
        all_num_parameters = []
        num_layers = len(self.param_groups) - 1

        for idx, group in enumerate(self.param_groups):
            lr = group['lr']

            for param in group['params']:
                if param.grad is None:
                    num_flips = 0
                    continue
                
                if self.last_layer_float & (idx == num_layers):       #last layer updates are unconstrained (floating point last layer)
                    # this is vanilla SGD.
                    # no implementation of adam yet
                    param.data = param.data - lr* param.grad       #used set 10*lr because lr should be greater in the floating layer. binary layers need more fine grained updates.
                    num_flips = torch.numel(param.data)

                else:                           #not last layer updates are ternary (expectationSGD)
                    num_flips = _expectation_sgd_ryan(param=param, lr=lr)

                    # if you want whole network to be floats (Not sure why i did this? perhaps for modal updates)
                    # param.data = param.data - 10*lr * param.grad #lr should be greater in the floating layer. binary layers need more fine grained updates.
                    # num_flips = torch.numel(param.data)
                

                num_parameters = torch.numel(param.data)
                all_num_flips.append(num_flips)
                all_num_parameters.append(num_parameters)    

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
    
    #debug print statement:
    if param.grad.abs().sum() == 0:
        print("param gradient all zeros") 

    old_param = param.data
    param_grad = param.grad


    clamp_before_expectation = False
    
    if clamp_before_expectation:
        #think i can simplify expression to grad*param > 0
        mask = ((param_grad < 0) & (param.data == -1)) | ((param_grad > 0) & (param.data == 1))
        param_grad[mask] = 0

    grad_sign = torch.sign(param_grad)
    grad_max = torch.max(torch.abs(param_grad))
    if grad_max != 0:
        grad_importance = lr*torch.abs(param_grad)/grad_max  #have temporarily removed safety mechanism (  torch.clamp_max ( , 1)  )
    else:
        grad_importance = torch.zeros_like(param_grad)

    update = grad_sign*torch.bernoulli(grad_importance)
                                                                                                # wait what if we let the weights update out of -1,0,1 and then re-quantise the weights after (not with sign)??
                                                                                                # this would allow network to increase the magnitude of one weight by decreasing the magnitude of other weights...?
    temp = param.data - update
    temp = torch.round(temp)    #why is this line necessary? 
    
    if clamp_before_expectation == False:
        param.data = torch.clamp(temp, max=1, min=-1).to(torch.float)  # torch.clamp makes sure you can't nudge outside of {-1, 0, 1}
    


    #for future: FIXME IM ANNOYED becuase the .to(float) is is a hack to make sure that W.grad is float, even though making W int sohuldnt affect W.grad!!!!! fuckign pytorch.
    #note this cannot be fixed without refactoring the code, whilst we have floating point gradients


    #metrics - no longer accurate when clamping before expectation
    unsaturated = (old_param * update) == 1    #because if param.data = 1 then it will only change if update = 1. also if param.data = -1, then it will only change if update = -1. Note that in both cases, the product is 1. therefore only these weights are unsaturated.
    num_flipped = torch.sum(unsaturated)

    return num_flipped