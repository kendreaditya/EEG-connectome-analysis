from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import datetime

# ADD SOUPORT FOR TPUs
class Train():
    def __init__(self, model, dataset, epochs, batch_size, criterion, optimizer, cross_validation, device='cuda'):
        torch.manual_seed(1)
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.log_name = f"{model.__class__.__name__}|{str(datetime.datetime.now()).replace(" ", "|").replace(":", "~").split(".")[0]}.log"

        self.data_loader(dataset, batch_size, cross_validation)
        if device=='cuda':
            self.cuda_device()

        self.train()
        self.test()

    def cuda_device(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def logger(self, epoch, train_loss, train_acc, val_loss, val_acc):
        if os.path.exists(f"../model-information/{self.log_name}"):
            with open(f"../model-information/{self.log_name}", 'a') as file:
                file.write(str(self.criterion))
                file.write(str(self.optimizer))
                file.wrie("epochs,training loss,training accuracy,validation loss,validation accuracy\n")

        with open(f"../model-information/{self.log_name}", 'a') as file:
            file.write(f"{epoch},{round(float(train_loss),4)},{round(float(train_acc),4)},{round(float(val_loss),4)},{round(float(val_acc),4)}\n")

    def data_loader(self, dataset, batch_size, cross_validation):
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths=cross_validation)
        self.train_dataloader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True)
        self.test_set = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True)

    def train(self):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in epochs:
            for batch_i,(data, target) in enumerate(self.train_dataloader):
                self.model.train()

            else:
                self.model.eval()



    def test(self):
        pass
