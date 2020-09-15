# Aditya Kendre
# Data Segmentation
import sys
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from mat73 import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

class Preprocessor():
    def __init__(self):
        PATH = "../data/raw-data/"
        data_files = os.listdir(PATH)
        for fn in data_files:
            temp_data = loadmat(PATH+fn)
            self.labels = temp_data["label"]
            self.data = temp_data["data"]
            del temp_data

    def segmentation(self, arr, output_shape, splits, axis_split):
        segmented_data = np.split(arr, splits, axis=axis_split)
        for i in range(len(segmented_data)):
            segmented_data[i] = self.average(segmented_data[i], output_shape)
        return segmented_data

    def average(self, arr, output_shape):
        return np.average(np.reshape(arr, output_shape), axis=-1)

    def band(self, arr, freq_axis):
        bands = {"delta":[0, 4],
                 "theta":[4, 8],
                 "alpha":[8, 14],
                 "beta":[14, 60]}
        return np.split(arr, [band[1] for band in list(bands.values())[:-1]], axis=freq_axis)


