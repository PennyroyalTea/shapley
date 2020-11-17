import torch
import torch.nn
import torch.utils.data

def get_score(model, val_set, batch_size=16):

    criterion = torch.nn.CrossEntropyLoss() ##### temporary

    val_loader = torch.utils.data.DataLoader(
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