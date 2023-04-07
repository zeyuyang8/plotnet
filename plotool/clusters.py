import plotly.graph_objects as go

LABEL_RANGE = [-1, 1]
X_RANGE = [-6, 6]
Y_RANGE = [-6, 6]
FIG_SIZE = [500, 500]  # [height, width]


def plot_clusters(data, title, colorscale='RdBu'):
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
            colorbar=dict(thickness=10))
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
