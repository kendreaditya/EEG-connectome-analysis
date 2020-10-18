import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Resnet(nn.Module):
    def __init__(self, input_size, in_channel):
        super(Resnet, self).__init__()

        # Basic Blocks
        self.basic_block1 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                          nn.BatchNorm3d(64, momentum=0.1),
                                          nn.ReLU(),
                                          nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                          nn.BatchNorm3d(64))

        self.basic_block2_1 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(128, momentum=0.1),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(128))

        self.basic_block2_2 = nn.Sequential(nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(128))

        self.basic_block3_1 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(256, momentum=0.1),
                                            nn.ReLU(),
                                            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(256))

        self.basic_block3_2 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(256),
                                            nn.ReLU(),
                                            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                            nn.BatchNorm3d(256))
        # Layers
        self.layer0 = nn.Sequential(nn.Conv3d(in_channel, 64, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        self.layer1 = nn.Sequential(self.basic_block1,
                                    self.basic_block1)

        self.layer2 = nn.Sequential(self.basic_block2_1,
                                    self.basic_block2_2)

        self.layer3 = nn.Sequential(self.basic_block3_1,
                                    self.basic_block3_2)

        conv_output = self.conv_layers(torch.zeros(input_size)).shape

        self.fc1 = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                 nn.Linear(conv_output[-1], 1024),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1024, 3))

    def conv_layers(self, X):
        # CNN Layer 
        X = self.layer0(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)

        # Flatten
        X = X.reshape(X.size(0), -1)
        return X

    def fc_layers(self, X):
        # Fully Connected Layer
        X = self.fc1(X)
        X = self.fc2(X)
        return X

    def forward(self, X):
        X = self.conv_layers(X)
        X = self.fc_layers(X)
        return X
