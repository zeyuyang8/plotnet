import plotly.graph_objects as go
from .constants import LABEL_RANGE, X_RANGE, Y_RANGE
from .constants import FIG_SIZE, COLORSCALE
from .constants import THICKNESS


def plot_clusters(data, title, colorscale=COLORSCALE):
    """
    Plot clusters by plotly.
    Input:
        data: numpy array of 2D points and labels
        colorscale: plotly colorscale
        title: title of the plot
    Output:
        fig: plotly figure
    """

    fig = go.Figure(data=go.Scatter(
        x=data[0][:, 0],  # x axis
        y=data[0][:, 1],  # y axis
        mode='markers',
        showlegend=False,
        marker=dict(
            color=data[1],  # color by labels
            cmin=LABEL_RANGE[0],
            cmax=LABEL_RANGE[1],
            colorscale=colorscale,
            colorbar=dict(thickness=THICKNESS))
        ),
        layout_yaxis_range=Y_RANGE,
        layout_xaxis_range=X_RANGE)

    fig.update_layout(
        autosize=False,
        height=FIG_SIZE[0],
        width=FIG_SIZE[1],
        title=title)

    fig.show()
    return fig
