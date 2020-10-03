import torch.nn as nn
import torch.nn.functional as F

class ComplexCNN(nn.Module):
    def __init__(self, in_channel):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d( in_channel, 8, 5, padding=2,stride=2)
        self.conv2 = nn.Conv2d( 8,16, 3, padding=1,stride=2)
        self.conv3 = nn.Conv2d(16,32, 3, padding=1,stride=2)
        self.conv4 = nn.Conv2d(32,32, 3, padding=1,stride=2)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32*1*1, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = self.pool1(X)
        X = self.drop1(X)
        X = X.reshape(-1, 32)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X  # logits
