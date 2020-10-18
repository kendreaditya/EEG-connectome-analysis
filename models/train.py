from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from torch.utils.tensorboard import SummaryWriter

class Train():
    def __init__(self, model, datasets, epochs, batch_size, criterion, optimizer, tensorboard=True, device_type='cuda', metrics_path="/content/drive/Shared drives/EEG_Aditya/model-results/"):
        torch.manual_seed(1)
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_path = metrics_path
        self.shape = np.array([np.array(i) for i in datasets[0][1][0]]).shape
        self.model_name = f"{model.__class__.__name__}--{str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').split('.')[0]}"
        print(self.model_name)
        time.sleep(1)
        self.device(device_type)
        self.data_loader(datasets, batch_size)
        if tensorboard:
            self.tb = SummaryWriter()

        self.train()
        test_acc, test_loss = self.test(self.test_dataloader, log=True)
        print(f"Test Accuracy: {round(test_acc, 4)}, Test Loss: {round(test_loss,4)}")

    def device(self, device):
        self.device_type = device
        if device == "cuda":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def save_model(self):
        torch.save(self.model.state_dict(), f"{self.metrics_path}parameters/{self.model_name}.pt")

    def logger(self, epoch, batch, train_loss, train_acc, val_loss, val_acc):
        if os.path.exists(f"{self.metrics_path}metrics/{self.model_name}.log") == False:
            with open(f"{self.metrics_path}metrics/{self.model_name}.log", 'a') as file:
                file.write(str(self.shape)+"\n")
                file.write(str(self.model)+"\n")
                file.write(str(self.criterion)+"\n")
                file.write(str(self.optimizer)+"\n")
                file.write("epochs,batch,training loss,training accuracy,validation loss,validation accuracy\n")

        with open(f"{self.metrics_path}metrics/{self.model_name}.log", 'a') as file:
            file.write(f"{epoch},{batch},{round(float(train_loss),4)},{round(float(train_acc),4)},{round(float(val_loss),4)},{round(float(val_acc),4)}\n")

    def data_loader(self, datasets, batch_size):
        if self.device_type=="cuda":
            self.train_dataloader = torch.utils.data.DataLoader(datasets[0], batch_size, shuffle=True)
            self.val_dataloader = torch.utils.data.DataLoader(datasets[1], batch_size)
            self.test_dataloader = torch.utils.data.DataLoader(datasets[2], batch_size)

    def train(self):
        min_acc = self.test(self.test_dataloader, log=False)[0]
        for epoch in tqdm(range(self.epochs)):
            for batch_i,(data, targets) in enumerate(self.train_dataloader):
                # Training
                data = data.to(self.device)
                targets = targets.to(self.device)

                train_loss, train_acc = self.forward_pass(data, targets, train=True)

                # Validation
                val_accs, val_losses = 0,0
                for val_batch_i, (val_data, val_targets) in enumerate(self.val_dataloader):
                    val_data = val_data.to(self.device)
                    val_targets = val_targets.to(self.device)
                    val_loss, val_acc = self.forward_pass(val_data, val_targets, train=False)
                    val_accs += val_acc
                    val_losses += val_loss
                val_accs, val_losses = val_accs/(val_batch_i+1), val_losses/(val_batch_i+1)

                # Saving model with lowest validation loss
                if min_acc <= val_accs:
                    self.save_model()
                    min_acc = val_accs

                # Logs model metrics
                self.logger(epoch, batch_i, train_loss, train_acc, val_losses, val_accs)

            # Tensorboard
            self.tensorboard([['Training Loss', train_loss, epoch],
                              ['Validation Loss', val_losses, epoch],
                              ['Traning Accuracy', train_acc, epoch],
                              ['Validation Accuracy', val_accs, epoch],
                              ['Learning Rate', self.optimizer.param_groups[0]['lr'], epoch]])
        print("Finished Training")
        print(f"Training Accuracy: {round(train_acc, 4)}, Training Loss: {round(train_loss,4)}")
        print(f"Validation Accuracy: {round(val_accs, 4)}, Validation Loss: {round(val_losses, 4)}")


    def forward_pass(self, data, targets, train=False):

        if train:
            self.model.zero_grad()
            self.model.train()
        else:
            self.model.eval()

        with (torch.enable_grad() if train else torch.no_grad()):
            outputs = self.model(data)
            matches = [torch.argmax(i)==(j) for i,j in zip(outputs, targets)]
            loss = self.criterion(outputs, targets)
            try:
              acc = matches.count(True)/len(matches)
            except:
              acc = -1
        if train:
            loss.backward()
            self.optimizer.step()

        del outputs
        del matches
        return loss.item(), acc

    def test(self, dataloader, log=True):
        accs, losses = 0,0
        for batch_i, (data, targets) in enumerate(dataloader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            loss, acc = self.forward_pass(data, targets.long(), train=False)
            accs += acc
            losses += loss
        accs, losses = accs/(batch_i+1), losses/(batch_i+1)
        if log:
            self.logger(-1, -1, -1, -1, losses, accs)
        return accs, losses

    def tensorboard(self, *scalars):
        for name, scalar, epoch in scalars[0]:
            self.tb.add_scalar(name, scalar, epoch)

