from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import datetime

#Precison, Specifity/Sensitivity, Recall, F1, ROC curve,
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

    def logger(self, epoch, batch, train_loss, train_acc, val_loss, val_acc):
        if os.path.exists(f"../model-information/{self.log_name}"):
            with open(f"../model-information/{self.log_name}", 'a') as file:
                file.write(str(self.criterion))
                file.write(str(self.optimizer))
                file.wrie("epochs,batch,training loss,training accuracy,validation loss,validation accuracy\n")

        with open(f"../model-information/{self.log_name}", 'a') as file:
            file.write(f"{epoch},{batch},{round(float(train_loss),4)},{round(float(train_acc),4)},{round(float(val_loss),4)},{round(float(val_acc),4)}\n")

    def data_loader(self, dataset, batch_size, cross_validation):
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths=cross_validation)
        self.train_dataloader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_set, batch_size)
        self.test_set = torch.utils.data.DataLoader(test_set, batch_size)

    def train(self):
        for epoch in epochs:
            for batch_i,(data, targets) in enumerate(self.train_dataloader):
                # Training
                data.to(self.device)
                target.to(self.device)

                train_loss, train_acc = forward_pass(data, targets, train=True)

                # Validation
                val_accs, val_losses = 0,0
                for val_batch_i, (val_data, val_targets) in enumerate(self.val_dataloader):
                    val_data.to(self.device)
                    val_targets.to(self.device)
                    val_loss, val_acc = forward_pass(val_data, val_targets, train=False)
                    val_accs += val_acc
                    val_losses += val_loss
                val_accs, val_losses = val_accs/val_batch_i, val_losses/val_batch_

                # Logs model metrics
                self.logger(epoch, batch_i, train_loss, train_acc, val_losses, val_acces)

    def forward_pass(self, data, targets, train=False):
        if train:
            self.model.zero_grad()
            self.model.train()
        else:
            self.model.eval()

        outputs = self.model(data)
        matches = [torch.argmax(i)==torch.argmax(j) for i,j in zip(outputs, targets)]
        loss = self.criterion(outputs, targets)
        acc = matches.count(True)/len(matches)

        if train:
            loss.backward()
            optimizer.step()

        del outputs
        del matches
        return loss.item(), acc


    def test(self):
        accs, losses = 0,0
        for batch_i, (data, targets) in enumerate(self.test_set):
            data.to(self.device)
            targets.to(self.device)
            loss, acc = forward_pass(data, targets, train=False)
            accs += acc
            losses += loss
        accs, losses = accs/batch_i, losses/batch_i
        return accs, losses

    def graph(self):
        pass
