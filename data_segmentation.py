# Aditya Kendre
# Data Segmentation

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from mat73 import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

def to_TensorDataset(dataset):
    return TensorDataset(torch.Tensor(dataset['X']), torch.Tensor(dataset['y']))

def data_segmentation(arr, output_shape, splits, axis_split):
    segmented_data = np.split(arr, splits, axis=axis_split)
    for i in range(len(segmented_data)):
        segmented_data[i] = average(segmented_data[i], output_shape)
    return segmented_data

def average(arr, output_shape):
    return np.average(np.reshape(arr, output_shape), axis=-1)

def band_segmentation(arr, freq_axis):
    bands = {"delta":[0, 4],
             "theta":[4, 8],
             "alpha":[8, 14],
             "beta":[14, 60]}
    return np.split(arr, [band[1] for band in list(bands.values())[:-1]], axis=freq_axis)

# Data Segmentation types
def twoD_10_data(data, labels):
    dataset = {'X':[], 'y':[]}
    for datum, label in tqdm(zip(data, labels)):
        datum = data_segmentation(datum[0], [34,34, -1], 13, 3)
        for X in datum:
            dataset['X'].append(X)
            dataset['y'].append(label)
    #np.save('../data/twoD_10_data.npy', np.array([dataset['X'], dataset['y']]))
    torch.save(to_TensorDataset(dataset), '../data/twoD_10_data.pt')

def twoD_data(data, labels):
    dataset = {'X':[], 'y':[]}
    for datum, label in tqdm(zip(data, labels)):
        datum = average(datum[0], [34, 34, -1])
        dataset['X'].append(datum)
        dataset['y'].append(label)

    torch.save(to_TensorDataset(dataset), '../data/twoD_data.pt')

PATH = "../data/raw-data/"
data_files = os.listdir(PATH)

for fn in data_files:
    temp_data = loadmat(PATH+fn)
    labels = temp_data["label"]
    data = temp_data["data"]
    del temp_data
    twoD_10_data(data, labels)


