from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import os
import torchvision.models as models
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.ConvTranspose3d()
