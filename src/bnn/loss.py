import abc

import torch

import bnn.type

__all___ = [
    'LossFunction',
    'l1',
    'l2',
    'CE',
]


class LossFunction(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> float: ...


class l1(LossFunction):
    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> int:
        error = torch.abs(output - target)
        loss = torch.sum(error)

        # mean
        if error.ndim > 1:
            loss = loss / len(error)

        return loss

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sign(output - target)


class l2(LossFunction):
    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> int:
        error = torch.square(output - target)
        loss = error

        return loss #no averaging since we take a running average in the train_epoch loop

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad = (output - target) #bernardo originally ternarised this, but we are not ternarising the backprop gradients.  
        return grad / grad.shape[0] 

# class CrossEntropyLoss(LossFunction):
#     @staticmethod
#     def forward(output: torch.Tensor, target: torch.Tensor) -> int:
#         # assume out is logits
#         neg_log_softmax = -torch.nn.LogSoftmax(dim=-1)(output.to(float))
#         loss = torch.mean(neg_log_softmax[target])
#         return loss

#     @staticmethod
#     def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         scaled_target = target.to(bnn.type.INTEGER) * 2 - 1
#         return torch.sign(scaled_target - output)
    

class CE(LossFunction): #cross entropy
    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # output is raw logits (softmax not done yet)
        # target is one-hot encoded
        softmax_output = torch.softmax(output, dim=-1)
        log_softmax_output = torch.log(softmax_output)
        loss = -torch.sum(target * log_softmax_output)  # cross-entropy loss definition
        
        # wait why is the below wrong?
        # loss = -log_softmax_output[torch.argmax(target, dim=-1)] / output.size(0)
        
        return loss

    @staticmethod
    def backward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad = torch.softmax(output, dim=-1) - target
        return grad / grad.shape[0]