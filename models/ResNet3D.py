import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet4(nn.Module):
    def __init__(self, in_channel):
        super(Resnet4, self).__init__()
        self.layer0 = nn.Sequential(nn.Conv3d(in_channel, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2)),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        self.layer1 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2)),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(3*3*2*64, 256),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 3))

    def forward(self, X):
        # CNN Layer 
        X = self.layer0(X)
        X = self.layer1(X)

        # Flatten
        X = X.reshape(X.size(0), -1)

        # Fully Connected Layer
        X = self.fc1(X)
        X = self.fc2(X)
        return X
