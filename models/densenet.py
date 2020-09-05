import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

densenet = models.densenet161(pretrained=True)

