# Aditya Kendre
# preprocessor.py
# Prerocesses data given data processing funtion

import sys
from mat73 import loadmat
from tqdm import tqdm
import numpy as np
import os
import torch
import importlib

class Segmenter():
    def __init__(self, path="../data/raw-data/"):
        print("File conversion started.")
        self.path = path
        file = os.listdir(self.path)[0]
        file_data = loadmat(self.path+file)
        self.data = file_data["data"]
        self.labels = file_data["label"]
        del file_data
        print("Data file loaded")

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
