import torch.nn as nn


class MLP(nn.Module):
    """ Container for a multi-layer perceptron (MLP) network. """

    def __init__(self, feautres, net_shape, activation, problem,
                 fixed_init=False, init_weight=0.0, init_bias=0.0):
        super().__init__()
        modules = []
        # input layer and hidden layers
        for idx in range(len(net_shape)):
            if idx == 0:
                modules.append(nn.Linear(len(feautres), net_shape[idx]))
                if activation:
                    modules.append(activation)
            else:
                modules.append(nn.Linear(net_shape[idx - 1], net_shape[idx]))
                if activation:
                    modules.append(activation)

        # output layer
        modules.append(nn.Linear(net_shape[-1], 1))
        if problem == "REGRESSION":
            pass
        elif problem == "CLASSIFICATION":
            modules.append(nn.Tanh())

        self.network = nn.Sequential(*modules)

        # initialize the weights to a specified, constant value
        if fixed_init:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.constant_(module.weight, init_weight)
                    nn.init.constant_(module.bias, init_bias)
        else:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, -0.5, 0.5)
                    nn.init.uniform_(module.bias, -0.5, 0.5)

    def forward(self, x):
        x = self.network(x)
        return x
