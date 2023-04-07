import plotly.graph_objects as go
from .constants import LABEL_RANGE, X_RANGE, Y_RANGE
from .constants import FIG_SIZE, COLORSCALE, OPACITY
from .constants import THICKNESS


def plot_decision(data, x_range, y_range, decisions,
                  title, colorscale=COLORSCALE):
    ''' Plot decision boundary by plotly. '''

    fig = go.Figure(data=go.Scatter(
        x=data[0][:, 0],
        y=data[0][:, 1],
        mode='markers',
        showlegend=False,
        marker=dict(
            color=data[1],
            cmin=LABEL_RANGE[0],
            cmax=LABEL_RANGE[1],
            colorscale=colorscale,
            colorbar=dict(thickness=THICKNESS))
        ),
        layout_yaxis_range=X_RANGE,
        layout_xaxis_range=Y_RANGE,
        )

    fig.add_trace(
        go.Heatmap(
            x=x_range,
            y=y_range,
            z=decisions,
            zmin=LABEL_RANGE[0],
            zmax=LABEL_RANGE[1],
            opacity=OPACITY,
            colorscale=colorscale,
            showscale=False
        ))

    fig.update_layout(
        autosize=False,
        height=FIG_SIZE[0],
        width=FIG_SIZE[1],
        title=title)

    fig.show()
    return fig
