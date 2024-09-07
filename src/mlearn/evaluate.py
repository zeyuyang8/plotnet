import torch
import numpy as np


def eval(model, loss_fn, trainloader, testloader, mesh_shape, mesh, device):
    ''' Evaluate model '''
    train_size = len(trainloader.dataset)
    test_size = len(testloader.dataset)
    train_loss, test_loss = 0, 0
    train_correct, test_correct = 0, 0

    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(-1).type(torch.float)
            outputs = model(inputs)
            train_correct += (outputs * labels > 0).sum().item()
            train_loss += loss_fn(outputs, labels).item() / 2

        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(-1).type(torch.float)
            outputs = model(inputs)
            test_correct += (outputs * labels > 0).sum().item()
            test_loss += loss_fn(outputs, labels).item() / 2

    train_loss /= train_size
    test_loss /= test_size
    train_acc = train_correct / train_size
    test_acc = test_correct / test_size

    with torch.no_grad():
        mesh = mesh.to(device)
        decision_boundary = model(mesh)
        confidence = decision_boundary.detach().numpy().reshape(mesh_shape)
        decisions = np.where(confidence < 0, -1, 1)

    return train_loss, test_loss, train_acc, test_acc, confidence, decisions
