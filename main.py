import torch

import argparse

from shapley_utils import get_score, train_on_subset
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
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
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

print(model)

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
# trainloader = data.DataLoader(
#     train_data,
#     shuffle=True,
#     batch_size=16
# )

###

train_on_subset(model, train_data, list(range(len(train_data))))

print(get_score(model, val_data))