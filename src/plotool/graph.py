import numpy as np
import matplotlib as mpl
import math
from .constants import LABEL_RANGE, COLORSCALE, RELATIVE_SIZE
from .constants import OPACITY, THICKNESS, X_RANGE, Y_RANGE, LINE_WIDTH, X_SHIFT
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_marker_color(color_map):
    '''
    Input: Predefined color_map from matplotlib
    Output: Function that takes risk and returns color for use in plt.scatter()
    '''

    # Note that this code is tricky and worth examining
    risk_norm = mpl.colors.Normalize(vmin=LABEL_RANGE[0], vmax=LABEL_RANGE[1])
    color_mapper = mpl.cm.ScalarMappable(norm=risk_norm, cmap=color_map)

    # Return a function via a lambda expression
    return lambda risk: color_mapper.to_rgba(math.tanh(risk))


WEIGHT_COLOR = create_marker_color(COLORSCALE)


def edge_calc(net_info, num_row, num_col, row_spacing, col_spacing):
    ''' Compute edge coordinates. '''
    row_width = (1 - row_spacing) / num_row
    col_width = (1 - col_spacing) / num_col

    row_interval = row_spacing / (num_row - 1)
    col_interval = col_spacing / (num_col - 1)

    node_net = {}
    for idx_col in range(num_col):
        for idx_row in range(num_row):
            node_id = (idx_col, idx_row)
            node_net[node_id] = {}

            node_verical = (1 - (idx_row * (row_width + row_interval)),
                            1 - ((idx_row + 1) * row_width + idx_row * row_interval))
            node_horizon = idx_col * (col_width + col_interval), (idx_col + 1) * col_width + idx_col * col_interval
            node_mean_vertical = np.mean(node_verical)
            node_left = node_horizon[0], node_mean_vertical
            node_right = node_horizon[1], node_mean_vertical

            node_net[node_id]["left"] = node_left
            node_net[node_id]["right"] = node_right

    edge_net = []
    num_layers = len(net_info)
    for idx in range(num_layers):
        if idx % 2 == 0:
            layer_weight = net_info[idx]["layer_params"][0].detach().numpy()
            num_out, num_in = layer_weight.shape

            for idx_in in range(num_in):
                for idx_out in range(num_out):
                    edge_weight = layer_weight[idx_out][idx_in]
                    edge_x0, edge_y0 = node_net[(idx, idx_in)]["right"]
                    edge_x1, edge_y1 = node_net[(idx + 1, idx_out)]["left"]
                    edge_color = "rgb" + str(WEIGHT_COLOR(edge_weight)[:3])
                    edge_net.append([edge_x0, edge_y0, edge_x1, edge_y1, edge_weight, edge_color])
        else:
            layer_dim = net_info[idx]["out_dim"]
            for idx_out in range(layer_dim):
                edge_x0, edge_y0 = node_net[(idx, idx_out)]["right"]
                edge_x1, edge_y1 = node_net[(idx + 1, idx_out)]["left"]
                edge_weight = 0.0
                edge_color = "black"
                edge_net.append([edge_x0, edge_y0, edge_x1, edge_y1, edge_weight, edge_color])
    return edge_net


def plot_nn_graph(net_info, datasets, in_features, mesh_data, epoch, title,
                  colorscale=COLORSCALE,
                  row_spacing=0.2, col_spacing=0.6):
    ''' Tabular visualization of a multi-layer perceptron neural network. '''

    # Data
    net_hidden_info = net_info[epoch]
    num_layers = len(net_hidden_info)
    out_dims = []
    for layer_dict in net_hidden_info:
        out_dims.append(layer_dict["out_dim"])
    max_out_dim = max(out_dims)
    x_range, y_range, mesh_shape, mesh = mesh_data
    num_input_features = mesh.shape[1]

    # Maximum number of rows
    row_max = max(num_input_features, max_out_dim)

    # Visualization
    column_titles = ["Features"]
    for idx in range(num_layers - 1):
        col_name = str(net_info[epoch][idx]["layer_type"]).split("'")[1].split(".")[-1]
        column_titles.append(col_name)
    column_titles.append("Output")

    edge_info = edge_calc(net_hidden_info, row_max, num_layers + 1, row_spacing, col_spacing)
    fig = make_subplots(rows=row_max, cols=num_layers + 1,
                        column_titles=column_titles,
                        horizontal_spacing=col_spacing / num_layers,
                        vertical_spacing=row_spacing / (row_max - 1), start_cell="top-left")

    # Input layer
    for idx in range(num_input_features):
        fig.add_trace(
            go.Heatmap(
                x=x_range, y=y_range,
                z=mesh[:, idx].tanh().detach().numpy().reshape(mesh_shape),
                zmin=LABEL_RANGE[0],
                zmax=LABEL_RANGE[1],
                opacity=OPACITY,
                colorscale=COLORSCALE,
                showscale=False),
            col=1, row=idx + 1)

        fig.add_annotation(x=X_RANGE[0],
                           y=(Y_RANGE[0] + Y_RANGE[1]) / 2,
                           xshift=X_SHIFT,
                           text=in_features[idx],
                           showarrow=False,
                           col=1, row=idx + 1)

        fig.update_xaxes(range=X_RANGE, showticklabels=False, col=1, row=idx + 1)
        fig.update_yaxes(range=Y_RANGE, showticklabels=False, col=1, row=idx + 1)

    # hidden layers
    for idx_layer in range(num_layers - 1):
        for idx_neuron in range(out_dims[idx_layer]):
            confidence = net_hidden_info[idx_layer]["confidence"][idx_neuron]
            fig.add_trace(
                go.Heatmap(
                    x=x_range, y=y_range,
                    z=confidence,
                    zmin=LABEL_RANGE[0], zmax=LABEL_RANGE[1],
                    opacity=OPACITY,
                    colorscale=colorscale,
                    showscale=False),
                col=idx_layer + 2, row=idx_neuron + 1)

            fig.update_xaxes(range=X_RANGE, showticklabels=False, col=idx_layer + 2, row=idx_neuron + 1)
            fig.update_yaxes(range=Y_RANGE, showticklabels=False, col=idx_layer + 2, row=idx_neuron + 1)

    # output
    confidence = net_hidden_info[num_layers - 1]["confidence"][0]
    fig.add_trace(
        go.Heatmap(
            x=x_range,
            y=y_range,
            z=confidence,
            zmin=LABEL_RANGE[0],
            zmax=LABEL_RANGE[1],
            opacity=OPACITY,
            colorscale=colorscale,
            showscale=False),
        col=num_layers + 1, row=1)

    fig.add_trace(go.Scatter(
        x=datasets[0][0][:, 0],  # Training data feature x
        y=datasets[0][0][:, 1],  # Training data feature y
        mode='markers',
        showlegend=False,
        marker=dict(
            color=datasets[0][1],  # Training data label
            cmin=LABEL_RANGE[0],
            cmax=LABEL_RANGE[1],
            colorscale=colorscale,
            colorbar=dict(thickness=THICKNESS))
        ), col=num_layers + 1, row=1)

    fig.update_xaxes(range=X_RANGE, showticklabels=True, col=num_layers + 1, row=1)
    fig.update_yaxes(range=Y_RANGE, showticklabels=True, col=num_layers + 1, row=1)

    height, width = row_max * RELATIVE_SIZE, (num_layers + 1) * RELATIVE_SIZE * 2
    fig.update_layout(height=height, width=width, autosize=False, showlegend=False, title_text=title)

    for edge in edge_info:
        x0, y0, x1, y1, weight, color = edge
        fig.add_shape(type="line",
                      xref="paper", yref="paper",
                      x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=color, width=LINE_WIDTH),)

    return fig
