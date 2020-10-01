import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncodeCNN(nn.Module):
    def __init__(self):
        super(AutoEncodeCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.transconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4, 4), stride=2, padding=0)
        self.transconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(4, 4), stride=4, padding=3)

    def forward(self, X):
        # Encoder
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = self.pool(X)

        # Decoder
        X = F.relu(self.transconv1(X))
        X = self.transconv2(X)
        return X

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

        # Dense Layers
        self.fc1 = nn.Linear(in_features=32*4*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=3)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X
