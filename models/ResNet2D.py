import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet(nn.Module):
    def __init__(self, in_channel):
        super(Resnet, self).__init__()
        self.layer0 = nn.Sequential(nn.Conv2d(in_channel, 16, kernel_size=(4, 4), stride=(2, 2)),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(288, 3))

    def forward(self, X):
        # CNN Layer 
        X = self.layer0(X)
        X = self.layer1(X)

        # Flatten
        X = X.reshape(X.size(0), -1)

        # Fully Connected Layer
        X = self.fc(X)
        return X
