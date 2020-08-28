import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from mat73 import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

def segmentation(arr, output_shape):
    output_shape.append(-1)
    return np.average(np.reshape(arr, output_shape), axis=-1)

PATH = "../data/raw-data/"
data_files = os.listdir(PATH)
dataset = {'X':[], 'y':[]}

for fn in data_files:
    labels = loadmat(PATH+fn)["label"]
    data = loadmat(PATH+fn)["data"]
    for datum, label in tqdm(zip(data, labels)):
        datum = segmentation(datum[0], [34, 34])
        dataset['X'].append(datum)
        dataset['y'].append(label)

torch.save(TensorDataset(torch.Tensor(dataset['X']), torch.Tensor(dataset['y'])), '../data/twoD_data.pt')
