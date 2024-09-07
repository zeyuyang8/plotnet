from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np


class TabularDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, idx):
        feature = self._data[0][idx].astype(np.float32)
        label = self._data[1][idx].astype(np.float32)
        sample = (feature, label)
        return sample


def create_dataloaders(trainset, testset, batch_size):
    """ Create dataloaders for training and testing. """
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
