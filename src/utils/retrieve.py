import numpy as np


def get_net_hidden_info(model, mesh_grid, problem_type):
    ''' Get net hidden info. Returns list of dictionaries. '''

    seq = list(model.children())[0]
    x_range, y_range, mesh_shape, mesh = mesh_grid

    net_hidden_info = []

    # the last layer is the output
    for layer in seq:
        layer_dict = {}
        # layer_input = mesh.detach().numpy()
        # in_dim = mesh.shape[-1]
        mesh = layer(mesh)
        out_dim = mesh.shape[-1]
        # print("In:", in_dim, "Out:", out_dim)

        if problem_type == "CLASSIFICATION":
            decision_boundary = mesh.tanh().detach().numpy()
        elif problem_type == "REGRESSION":
            decision_boundary = mesh.detach().numpy()
        new_shape = [out_dim, mesh_shape[0], mesh_shape[1]]

        confidence = np.ones(new_shape)
        decisions = np.ones(new_shape)

        for neuron in range(out_dim):
            confidence[neuron] = decision_boundary[:, neuron].reshape(mesh_shape)
            decisions_neuron = decision_boundary[:, neuron]
            decisions[neuron] = np.where(decisions_neuron < 0, -1, 1).reshape(mesh_shape)

        layer_type = type(layer)
        layer_params = []
        for param in layer.parameters():
            layer_params.append(param)

        layer_dict["out_dim"] = out_dim
        layer_dict["confidence"] = confidence
        layer_dict["decisions"] = decisions
        layer_dict["layer_type"] = layer_type
        layer_dict["layer_params"] = layer_params

        net_hidden_info.append(layer_dict)

    return net_hidden_info
