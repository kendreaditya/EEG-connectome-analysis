import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self,):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.conv2d(32, 32, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(17*17*32, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = self.drop1(self.pool1(F.relu(self.conv2(X))))
        X = X.reshape(-1, 17*17*32)
        X = self.drop2(F.relu(self.fc1(X)))
        X = self.fc2(X)
        return X  # logits
