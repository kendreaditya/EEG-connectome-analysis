import torch
import torch.nn as nn
import importlib.util
from scipy.io import loadmat
import train

# Study Data
labels = torch.load("/content/drive/Shared drives/EEG_Aditya/data/EEG3D99.pt")[1]
study_data = loadmat("/content/drive/Shared drives/EEG_Aditya/data/cnn-data/EEG_mean.mat")
alphatraindataset = torch.utils.data.TensorDataset(study_data["mean_alpha"][:66], labels[:66])
alphatestdataset = torch.utils.data.TensorDataset(study_data["mean_alpha"][66:], labels[66:])

# Processed Data
processed_data = torch.load("/content/drive/Shared drives/EEG_Aditya/data/EEG2DALPHA.pt")
traindataset = torch.utils.data.TensorDataset(processed_data['train/val'][0][:-1000], processed_data['train/val'][1][:-1000])
valdataset = torch.utils.data.TensorDataset(processed_data['train/val'][0][-1000:], processed_data['train/val'][1][-1000:])
testdataset = torch.utils.data.TensorDataset(processed_data['test'][0], processed_data['test'][1])

# Import Model
module = importlib.util.spec_from_file_location("Encoder2DCNN", "/content/EEG-connectome-analysis/models/Encoder2DCNN.py")
Encoder = importlib.util.module_from_spec(module)
module.loader.exec_module(Encoder)
model = Encoder.AutoEncoderCNN()

# Model Functions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# Train Model
train = train.Train(model, [alphatraindataset, alphatestdataset, alphatestdataset], epochs=100, batch_size=512,
                    criterion=criterion, optimizer=optimizer, tensorboard=True)
train.train()
