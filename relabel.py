#!/usr/bin/env python3

# This script takes dataset and generates new one with some datapoints being relabeled to make it noize

import os
import shutil
import random
import pandas as pd

SEED = 23956630
random.seed(SEED)

DATASET_FROM_PATH = 'imagewoof2-320'        # relative path to input dataset
DATASET_TO_PATH = 'imagewoof2-320-modified' # relative path to resulting dataset
DIF_PATH = 'dif.csv'                        # relative path to difference list

p = 0.1                                     # part of datapoints to be relabeled

df = pd.DataFrame(                          # dataframe to keep which images were moved
    columns=['img','from', 'to']
)

# get all image classes (e.g. n02089973, n02086240, etc)
classes = [f.name for f in os.scandir(os.path.join(DATASET_FROM_PATH, 'train')) if f.is_dir()]

# copy DATASET folders but not files
try:
    shutil.rmtree(DATASET_TO_PATH)
except FileNotFoundError:
    pass
shutil.copytree(
        DATASET_FROM_PATH,
        DATASET_TO_PATH,
        ignore=lambda dir, files : [f for f in files if os.path.isfile(os.path.join(dir, f))]
    )

for _class in classes:
    # copy validation without changing
    val_from = os.path.join(DATASET_FROM_PATH, 'val', _class)
    val_to = os.path.join(DATASET_TO_PATH, 'val', _class)
    for file in os.scandir(val_from):
        if file.is_dir():
            continue
        shutil.copy2(os.path.join(val_from, file.name), os.path.join(val_to, file.name))

    #copy train with changing random classes
    train_from = os.path.join(DATASET_FROM_PATH, 'train', _class)
    train_to = os.path.join(DATASET_TO_PATH, 'train', _class)
    for file in os.scandir(train_from):
        if file.is_dir():
            continue
        if random.random() > p:
            # copy to same class
            shutil.copy2(os.path.join(train_from, file.name), os.path.join(train_to, file.name))
        else:
            # copy to random other class
            new_class = _class
            while new_class == _class:
                new_class = random.choice(classes)
            shutil.copy2(os.path.join(train_from, file.name), os.path.join(DATASET_TO_PATH, 'train', new_class))
            df = df.append(                 # ineffective!!
                {'img': file.name, 'from': _class, 'to': new_class},
                ignore_index=True
            )

df.to_csv(DIF_PATH)                         # write difference list to csv file
