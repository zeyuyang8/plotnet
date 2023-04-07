import numpy as np
import torch
import math

MAPPING = {
    "x": {"f": lambda x: x[0], "label": "X_1"},
    "y": {"f": lambda x: x[1], "label": "X_2"},
    "x squared": {"f": lambda x: x[0] * x[0], "label": "X_1^2"},
    "y squared": {"f": lambda x: x[1] * x[1], "label": "X_2^2"},
    "x times y": {"f": lambda x: x[0] * x[1], "label": "X_1X_2"},
    "sin x": {"f": lambda x: math.sin(x[0]), "label": "sin(X_1)"},
    "sin y": {"f": lambda x: math.sin(x[1]), "label": "sin(X_2)"}
}


def create_mesh(mesh_size, margin, features):
    """ Create mesh """

    # Create a mesh grid on which we will run our model
    x_min, x_max, y_min, y_max = margin
    x_range = np.arange(x_min, x_max, mesh_size)
    y_range = np.arange(y_min, y_max, mesh_size)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Flatten
    data = np.c_[x_grid.ravel(), y_grid.ravel()]

    # Transform
    transformed_data = []
    for feature in features:
        func = MAPPING[feature]["f"]
        temp = np.apply_along_axis(func, 1, data)
        transformed_data.append(temp)

    transformed_data = np.vstack(transformed_data).T

    transformed_data_tensor = torch.tensor(transformed_data, dtype=torch.float32)

    return x_range, y_range, x_grid.shape, transformed_data_tensor
