import torch
import torch.nn as nn
import importlib.util
from scipy.io import loadmat
import train

# Study Data
labels = torch.load("/content/drive/Shared drives/EEG_Aditya/data/EEG3D99.pt")[1]
study_data = torch.Tensor(loadmat("/content/drive/Shared drives/EEG_Aditya/data/cnn-data/EEG_mean.mat")['mean_alpha'].tolist())
alphatraindataset = torch.utils.data.TensorDataset(study_data[:66], labels[:66])
alphatestdataset = torch.utils.data.TensorDataset(study_data[66:], labels[66:])

# Processed Data
processed_data = torch.load("/content/drive/Shared drives/EEG_Aditya/data/EEG2DALPHA.pt")
traindataset = torch.utils.data.TensorDataset(processed_data['train/val'][0][:-1000], processed_data['train/val'][1][:-1000])
valdataset = torch.utils.data.TensorDataset(processed_data['train/val'][0][-1000:], processed_data['train/val'][1][-1000:])
testdataset = torch.utils.data.TensorDataset(processed_data['test'][0], processed_data['test'][1])

# Import Model
module = importlib.util.spec_from_file_location("Simple2DCNN", "/content/EEG-connectome-analysis/models/Simple2DCNN.py")
lib = importlib.util.module_from_spec(module)
module.loader.exec_module(lib)
model = lib.SimpleCNN()

# Model Functions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# Train Model
train = train.Train(model, [alphatraindataset, alphatestdataset, alphatestdataset], epochs=1000, batch_size=512,
                    criterion=criterion, optimizer=optimizer, tensorboard=True)
