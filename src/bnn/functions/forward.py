import abc

import torch

import bnn.type

from . import functions

__all__ = [
    'ForwardFunc',
    'MatMulBinarise',
    'SignBinarise',
    'LayerMeanBinarise',
    'LayerMedianBinarise',
    'OneHot',
]


class ForwardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor: ...




class MatMulBinarise(ForwardFunc):
    def __call__(self, x: torch.Tensor, W: torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        # TODO - make this configurable?
        integer = x @ W + b  #FIXME  I HAVE CHANGED THIS TO FLOAT MATMULS    integer = functions.int_matmul(x, W)
        out_binary = self.binarise(x=integer)

        return out_binary, integer

    @abc.abstractmethod
    def binarise(self, x: torch.Tensor) -> torch.Tensor: ...


class SignBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        return functions.binarise(x=x, threshold=0)


class LayerMeanBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE means over layer dimension - samples stay independent :)
        means = torch.mean(x.to(torch.float), dim=-1)

        out = torch.ones_like(x, dtype=bnn.type.INTEGER)
        out[x < means[..., None]] = -1

        return out


class LayerMedianBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE median over layer dimension - samples stay indepenent :)
        medians_out = torch.median(x, dim=-1)
        medians = medians_out.values

        out = torch.ones_like(x, dtype=bnn.type.INTEGER)
        out[x < medians[..., None]] = -1

        return out


class reluBinarise(MatMulBinarise):
    def binarise(self, x: torch.Tensor) -> torch.Tensor:   #maps x from {-1, 0, 1} to {0, 1}, unlike the others which map from {-1, 0, 1} to {-1,1}
        x = torch.nn.functional.relu(x)     # ReLU
        
        # OLD CODE WHICH FOUND NON-ITERATIVE MEANS:
            # means = torch.mean(x, dim=-1)       # NOTE means over layer dimension - samples stay indepenent :)

            # iter_mean = torch.mean(x[x>0])
            # while iter_mean != torch.mean(x[x>(iter_mean/2)]):
            #     iter_mean = torch.mean(x[x>(iter_mean/2)])
            # means = iter_mean
        

        iter_mean = torch.mean(x[x > 0])
            # while iter_mean != torch.mean(row[row > (iter_mean / 2)]):
            #     iter_mean = torch.mean(row[row > (iter_mean / 2)])

        for i in range(4):
            iter_mean = torch.mean(x[x > (iter_mean / 2)])  #mean is found for the whole batch, not just for the layer.
                
        out = torch.ones_like(x, dtype=torch.float) #FIXME THIS IS BECAUSE OF THE WHOLE PYTORCH PARAMETER AND GRAD HAVE TO BE SAME TYPE
        out[x < iter_mean] = 0        
        return out.float()





class MatMulMax(ForwardFunc):
    def __call__(self, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        integer = x @ W + b #FIXME I HAVE CHANGED THIS TO FLOAT MATMULS  integer = functions.int_matmul(x, W)
        
        # print(f'pre-activations: {integer}')    #debug
        out_binary = self.binary_max(x=integer)
        # print(f'post-activations: {out_binary}') #debug

        return out_binary, integer

    @abc.abstractmethod
    def binary_max(self, x: torch.Tensor) -> torch.Tensor: ...

# this layer class is sometimes used as the last layer in the network
class OneHot(MatMulMax):
    def binary_max(self, x: torch.Tensor) -> torch.Tensor:
        return functions.one_hot_argmax0(x)

# dont think this has been used in the experiments
class BitCountMax(MatMulMax):
    out_dims: int
    extra_dims: int

    def __init__(self, out_dims: int, extra_dims: int):
        self.out_dims = out_dims
        self.extra_dims = extra_dims

    def binary_max(self, x: torch.Tensor) -> torch.Tensor:
        # reshape and binarise
        reshaped = x.reshape(-1, self.extra_dims, self.out_dims)
        binary_reshaped = functions.binarise(reshaped)      #threshold is default value, so 0

        # count bits and argmax
        bitcounts = torch.sum(binary_reshaped, dim=-2)
        one_hot = functions.one_hot_argmax(bitcounts)

        return one_hot
