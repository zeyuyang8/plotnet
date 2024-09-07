import numpy as np
import plotly.graph_objects as go
from .graph import plot_nn_graph


def animate_nn_graph(net_info, datasets, in_features, mesh_data, max_epoch, key):
    ''' Animation of the neural network training process.'''

    epoch_list = np.array(np.exp2(np.arange(int(np.log2(max_epoch) + 1))), dtype=int)
    if epoch_list[-1] != max_epoch:
        epoch_list = np.append(epoch_list, max_epoch)
    epoch_list = np.insert(epoch_list, 0, 0)
    epoch = 0
    title = f"Neural network architeture at epoch {epoch} - " + key

    # initial guess
    fig = plot_nn_graph(net_info, datasets, in_features, mesh_data, epoch, title)
    num_traces = len(fig["data"]) - 1

    # add frames
    frames = []
    for k in epoch_list:
        title = f"Neural network architeture at epoch {k} - " + key
        fig = plot_nn_graph(net_info, datasets, in_features, mesh_data, k, title)
        frame = go.Frame(
            data=fig["data"],
            layout=fig["layout"],
            traces=list(range(num_traces - 1)) + [num_traces - 1],
            name=str(k))
        frames.append(frame)
    fig["frames"] = frames

    def frame_args(duration):
        return {"frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"}}

    sliders = [{"pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [{"args": [[f.name], frame_args(0)],
                           "label": str(epoch_list[k]),
                           "method": "animate"}
                          for k, f in enumerate(fig.frames)]}]

    fig.update_layout(updatemenus=[{"buttons": [{"args": [None, frame_args(50)],
                                                 "label": "&#9654;",  # play symbol
                                                 "method": "animate"},
                                                {"args": [[None], frame_args(0)],
                                                 "label": "&#9724;",  # pause symbol
                                                 "method": "animate"}],
                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.1,
                                    "y": 0}],
                      sliders=sliders)

    return fig
