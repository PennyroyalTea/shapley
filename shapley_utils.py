import json

import torch
import torch.nn
import torch.utils.data

from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

def get_score(model, val_set, batch_size=16):

    criterion = torch.nn.CrossEntropyLoss() ##### temporary

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True
    )

    total_loss = 0

    model.eval()
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

def train_on_subset(model,
                    train_set, val_set,
                    permutation,
                    exp_id = -1,
                    res_dist = None,
                    batch_size=256, epochs=5,
                    criterion=None,
                    optimizer=None):

    debug_step = 10 # debug output running loss every x batches

    if optimizer is None:
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-4, momentum=0.9)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    print('exp {} : Starting Training'.format(exp_id))

    # turn on cuda
    model = model.cuda()

    # turn on training mode
    model.train()

    # summaryWriter for logging
    writer = SummaryWriter()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices=permutation)
    )

    batches_processed = 0

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # log epoch' val score
        val_score = get_score(model, val_set, batch_size)
        writer.add_scalar('loss/val', val_score, batches_processed)

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

            # update statistics
            running_loss += loss.item()
            batches_processed += 1

            # log running train score
            if debug_step != -1 and i % debug_step == (debug_step - 1):
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / (debug_step * batch_size)))
                writer.add_scalar('loss/train', running_loss / (debug_step * batch_size), batches_processed)

                running_loss = 0.0

    print('exp {} : Finished Training'.format(exp_id))
    return get_score(model, val_set, batch_size)

def save_res(perm_to_score, permutation, score, filename='score.json'):
    perm_to_score[str(permutation)] = score
    with open(filename, 'w') as f:
        json.dump(perm_to_score, f, indent=4)