import os
import torch
from .evaluate import eval


def train_loop(model, loss_fn, optimizer, device, n_epochs,
               trainloader, testloader,
               mesh_shape, mesh,
               target_path, model_name, print_every=10):
    ''' Train a model and save training history '''
    # Create target path
    model_path = target_path + model_name + "/"

    # Make directory if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Move model and loss_fn to device
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Keep track of training and test loss
    train_loss_list, test_loss_list = [], []
    train_accuracy_list, test_accuracy_list = [], []
    confidence_list, decisions_list = [], []

    # Save initial guess
    path = model_path + "epoch" + str(0) + ".pt"
    torch.save(model, path)

    # Evaluate initial guess
    model.eval()
    evaluation = eval(model, loss_fn, trainloader, testloader, mesh_shape, mesh, device)
    train_loss, test_loss, train_acc, test_acc, confidence, decisions = evaluation
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_accuracy_list.append(train_acc)
    test_accuracy_list.append(test_acc)
    confidence_list.append(confidence)
    decisions_list.append(decisions)

    # Training
    for epoch in range(n_epochs):
        model.train()
        for batch_id, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(-1).type(torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        evaluation = eval(model, loss_fn, trainloader, testloader, mesh_shape, mesh, device)
        train_loss, test_loss, train_acc, test_acc, confidence, decisions = evaluation
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_accuracy_list.append(train_acc)
        test_accuracy_list.append(test_acc)
        confidence_list.append(confidence)
        decisions_list.append(decisions)

        # Print results
        if (epoch + 1) % print_every == 0 or epoch == n_epochs - 1:
            print("Epoch: {}/{}.. ".format(epoch + 1, n_epochs),
                  "Train Loss: {:.3f}.. ".format(train_loss),
                  "Test Loss: {:.3f}.. ".format(test_loss),
                  "Train Accuracy: {:.3f}.. ".format(train_acc),
                  "Test Accuracy: {:.3f}.. ".format(test_acc))

        # save model
        path = model_path + "epoch" + str(epoch + 1) + ".pt"
        torch.save(model, path)

    # Return training history
    results = (train_loss_list, test_loss_list,
               train_accuracy_list, test_accuracy_list,
               confidence_list, decisions_list)

    return results
