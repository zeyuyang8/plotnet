import plotly.graph_objects as go


def plot_clusters_plotly(data, cmap_name, figsize, title):
    """
    Plot clusters using plotly. 
    Args:
        data - tuple of (points, labels) where points is a numpy array of points,
            and labels is a numpy array of labels.
        cmap_name - string of the name of the color map to use.
        figsize - tuple of (width, height) of the figure.
        title - string of the title of the figure.
    Returns a plotly figure.
    """
    height, width = figsize
    fig = go.Figure(data=go.Scatter(
        x=data[0][:, 0],
        y=data[0][:, 1],
        mode='markers',
        showlegend=False,
        marker=dict(
            color=data[1],
            cmin=-1,
            cmax=1,
            colorscale=cmap_name, 
            colorbar=dict(thickness=10))
        ),
        layout_yaxis_range=[-6, 6],
        layout_xaxis_range=[-6, 6])

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        title=title)

    fig.show()
    return fig
