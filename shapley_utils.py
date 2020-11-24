import torch
import torch.nn
import torch.utils.data

from torch.utils.data import DataLoader, SubsetRandomSampler

def get_score(model, val_set, batch_size=16):

    criterion = torch.nn.CrossEntropyLoss() ##### temporary

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True
    )

    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(val_set)

def train_on_subset(model,  train_set, permutation,
                    batch_size=16, epochs=5,
                    criterion=None,
                    optimizer=None):
    debug_step = 100 # debug output running loss every x batches

    if optimizer is None:
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-4, momentum=0.9)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices=permutation)
    )

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print statistics
            if debug_step != -1 and i % debug_step == (debug_step - 1):
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / (debug_step * batch_size)))
                running_loss = 0.0
    print('Finished Training')