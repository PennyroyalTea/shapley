import torch

import argparse

from shapley_utils import get_score
###
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data as data
###

from shapley import monte_carlo

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#TODO: argparse

# downloading dataset

model = models.resnet18(pretrained=True).cuda()
model.fc = torch.nn.Linear(512, 10).cuda()

data_transforms = T.Compose(
    [T.Resize((224, 224)),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)


val_data = datasets.ImageFolder(
    root= "./imagewoof2-320/val",
    transform=data_transforms,
)

### training
train_data = datasets.ImageFolder(
    root="./imagewoof2-320/train",
    transform=data_transforms
)
trainloader = data.DataLoader(
    train_data,
    shuffle=True,
    batch_size=16
)
def trainloop(loader):
  for epoch in range(5):  # loop over the dataset multiple times

      running_loss = 0.0
      for i, data in enumerate(loader, 0):
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

          # print statistics
          running_loss += loss.item()
          if i % 100 == 99:    # print every 20 mini-batches
              print('[%d, %5d] loss: %.7f' %
                    (epoch + 1, i + 1, running_loss / 100))
              running_loss = 0.0

  print('Finished Training')
  trainloop(trainloader)
###

print(model)

print(get_score(model, val_data))