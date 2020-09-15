import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from train import Train

vgg11 = models.vgg11(pretrained=True)
upscale_layer = nn.ConvTranspose2d(1, 3, kernel_size=(4,4), stride=(2,2), padding=(3,3))
vgg11.classifier[6] = nn.Linear(4096, 3, bias=True)
vgg11.features[0] = nn.Sequential(upscale_layer, vgg11.features[0])
