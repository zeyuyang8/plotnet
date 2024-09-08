"""Solution for the visualizations of multi-layer perceptron networks."""

import os
import torch
import warnings
import torch.nn as nn
from src.utils.datasets import generate_data
from src.utils.mesh import create_mesh
from src.utils.retrieve import get_net_hidden_info
from src.mlearn.dataloader import TabularDataset, create_dataloaders
from src.mlearn.network import MLP
from src.mlearn.train import train_loop
from src.plotool.clusters import plot_clusters
from src.plotool.decision import plot_decision
from src.plotool.curve import plot_train_test_curve
from src.plotool.graph import plot_nn_graph
from src.plotool.animate import animate_nn_graph

warnings.filterwarnings('ignore')
