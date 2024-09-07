import numpy as np
import plotly.graph_objects as go


def plot_train_test_curve(train_arr, test_arr, ylabel, title):
    ''' Plot train and test loss curve by plotly. '''
    num_epochs = len(train_arr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(num_epochs) + 1, y=train_arr, name="Train"))
    fig.add_trace(go.Scatter(x=np.arange(num_epochs) + 1, y=test_arr, name="Test"))
    fig.update_layout(autosize=False, title=title, xaxis_title="Epoch", yaxis_title=ylabel)
    fig.show()
    return fig
