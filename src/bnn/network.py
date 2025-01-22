import torch

import bnn.functions
import bnn.layer
import bnn.type

__all__ = [
    'TernBinNetwork',
]


class TernBinNetwork(torch.nn.Module):
    layers: torch.nn.ModuleDict

    input: torch.nn.ParameterDict
    grad: torch.nn.ParameterDict

    dims: list[int]

    def __init__(
        self,
        dims: list[int],
        forward_func: bnn.functions.ForwardFunc | list[bnn.functions.ForwardFunc],
        backward_func: bnn.functions.BackwardFunc | list[bnn.functions.BackwardFunc],
    ):
        super().__init__()

        self.layers = torch.nn.ModuleDict()
        self.input = torch.nn.ParameterDict()
        self.grad = torch.nn.ParameterDict()

        self.dims = dims

        n_layers = len(dims) - 1
        # check forward/back func list length
        try:
            if len(forward_func) != n_layers:
                if len(forward_func) == 2:
                    # HACK needs to be this way otherwise OmegaConf complains
                    forward_func = [forward_func[0]] * (n_layers - 1) + [forward_func[-1]]
                else:
                    raise ValueError(f'{len(forward_func)}s forward funcs but {len(dims)} layers')
        except TypeError:
            forward_func = [forward_func] * n_layers

        try:
            if len(backward_func) != n_layers:
                if len(backward_func) == 2:
                    backward_func = [backward_func[0]] * (n_layers - 1) + [backward_func[-1]]
                else:
                    raise ValueError(f'{len(backward_func)}s backward funcs but {n_layers} layers')
        except TypeError:
            backward_func = [backward_func] * n_layers

        # init layers
        layers_zip = zip(dims, dims[1:], forward_func, backward_func)
        for i, (input_dim, output_dim, f_func, b_func) in enumerate(layers_zip):
            layer = bnn.layer.TernBinLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                forward_func=f_func,
                backward_func=b_func,
            )

            layer_name = f'TernBinLayer{i}'
            self.layers[layer_name] = layer

        self._clear_input_and_grad()
        return

    def _clear_input_and_grad(self) -> None:
        for layer_name, layer in self.layers.items():
            # TODO check - should this be in-place or create a new parameter?
            # init activations and grads

            #okay ive changed this to no longer instantiate paramaters, but just tensors.
            #this is because i dont know why these would be parameters anyway, and also because it means that when network.parameters() is called, these were being included and i think this was affecting optimizer loop over param_groups?
            self.input[layer_name] = torch.zeros(layer.input_dim, layer.output_dim, dtype=torch.float) #FIXME  TORCH FORCES ME TO MAKE IT A FLOAT OTHERWISE CANT GET FLOAT GRADS
            self.grad[layer_name] = torch.zeros_like(self.input[layer_name], dtype=torch.float) #FIXME  TORCH FORCES ME TO MAKE IT A FLOAT OTHERWISE CANT GET FLOAT GRADS
               
        return

    def _initialise(
        self,
        W_mean: float | None | list[float | None] = None,
        W_var: float | None | list[float | None] = None,
        W_zero_prob: float | None | list[float | None] = None,
    ) -> None:
        # input sanity checking
        if (W_zero_prob is not None) and (W_var is not None):
            raise ValueError('Cannot spacify both var and nonzero prob!')

        if isinstance(W_mean, list):
            if len(W_mean) != len(self.layers):
                raise IndexError(
                    'Number of means does not match number of layers! '
                    f'Means: {len(W_mean)} Layers: {len(self.layers)}'
                )
        else:
            W_mean = [W_mean] * len(self.layers)

        if isinstance(W_var, list):
            if len(W_var) != len(self.layers):
                raise IndexError(
                    'Number of vars does not match number of layers! ' f'Vars: {len(W_var)} Layers: {len(self.layers)}'
                )
        else:
            W_var = [W_var] * len(self.layers)

        if isinstance(W_zero_prob, list):
            if len(W_zero_prob) != len(self.layers):
                raise IndexError(
                    'Number of zero probs does not match number of layers! '
                    f'zero probs: {len(W_zero_prob)} Layers: {len(self.layers)}'
                )
        else:
            W_zero_prob = [W_zero_prob] * len(self.layers)

        # reset all weight vars
        zipped = zip(self.layers.values(), W_var, W_mean, W_zero_prob)
        for layer, var, mean, zero_prob in zipped:
            layer: bnn.layer.TernBinLayer
            layer._initialise_W(mean=mean, var=var, zero_prob=zero_prob)

        # reset grads and activations
        self._clear_input_and_grad()

        return

    # HACK
    def forward_no_proj(self, x: torch.Tensor) -> torch.Tensor:
        last_layer = list(self.layers.keys())[-1]

        for layer_name, layer in self.layers.items():
            if layer_name != last_layer:
                self.input[layer_name] = x
                x = layer(x)

        x = bnn.functions.int_matmul(x, layer.W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_name, layer in self.layers.items():
            self.input[layer_name] = x     #where x is the input (binarised!) activations which is the output from previous layer
            x = layer(x)    #calls forward of layer (because layer is a nn.module), which in turn calls forward_func of the layer, which matmuls(x,W) and binarises.

        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Propagate gradients back through network and output grad wrt input."""
        for layer_name, layer in reversed(self.layers.items()):
            layer: bnn.layer.TernBinLayer

            grad = layer.backward(grad, self.input[layer_name])    #calls backward, providing the arguments: grad and the input activations (which were the output of the forward pass)
            self.grad[layer_name] = grad

        return grad

    def backward_actual(self, grad: torch.Tensor) -> torch.Tensor:
        """Propagate gradients back through network and output grad wrt input."""
        for layer_name, layer in reversed(self.layers.items()):
            layer: bnn.layer.TernBinLayer

            grad = layer.backward_actual(grad, self.input[layer_name])
            self.grad[layer_name] = grad

        return grad


def network_params_al_ternary(Net: torch.nn.Module) -> bool:
    for parameter in Net.parameters():
        is_one = parameter == 1
        is_neg_one = parameter == -1
        is_zero = parameter == 0

        return torch.all(is_one | is_neg_one | is_zero)
