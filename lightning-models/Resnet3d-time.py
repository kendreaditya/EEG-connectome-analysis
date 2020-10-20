import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning.metrics as metrics
import sklearn
import numpy as np
from pdb import set_trace as bp

class ResNet(pl.LightningModule):
    def __init__(self, input_size=(1,1,130,34,34), in_channel=1):
        super().__init__()
        # Metrics 
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision(num_classes=3)
        self.recall = metrics.Recall(num_classes=3)
        self.fbeta = metrics.Fbeta(num_classes=3, beta=0.5)

        # Layers
        self.layer0 = nn.Sequential(nn.Conv3d(in_channel, 32, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        self.layer1 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(32, momentum=0.1),
                                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64, momentum=0.1),
                                    nn.ReLU())

        self.layer3 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(128, momentum=0.1),
                                    nn.ReLU())

        conv_output = self.conv_layers(torch.zeros(input_size)).shape

        self.fc1 = nn.Sequential(nn.Linear(conv_output[-1], 1024),
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

    def configure_optimizers(self):
        optimzer = torch.optim.SGD(self.parameters(), lr=1e-5)
        return optimzer

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        # Logs metrics
        metrics = self.metrics_step(outputs, y)
        for metric in metrics:
            self.log(f"train-{metric}", metrics[metric])

        # Logs Loss
        self.log("train-loss", loss)

        return loss

    def metrics_step(self, outputs, labels):
        pred = [torch.argmax(out) for out in outputs]

        accuracy_score = self.accuracy(pred, labels)
        # ROC Graph
        # X: Recall (sensitivity)
        # y: Precision (weighted 1-specificity)
        recall_score = self.recall(pred, labels)
        precision_score = self.precision(pred, labels)
        fbeta_score = self.fbeta(pred, labels)

        f1_score = metrics.functional.classification.f1_score(pred, labels, num_classes=3)
        confusion_matrix = metrics.functional.classification.confusion_matrix(pred, labels)
        roc_auc_score = self.multiclass_roc_auc_score(labels, pred)

        learning_rate = self.optimizers()[0].param_groups[0]['lr']

        return {'accuracy': accuracy_score,
                'recall': recall_score,
                'precision': precision_score,
                'fbeta': fbeta_score,
                'f1': f1_score,
                'ROC-AUC': roc_auc_score,
                'confusion-matrix':confusion_matrix,
                'lr': learning_rate}

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        lb = sklearn.preprocessing.LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return sklearn.metrics.roc_auc_score(y_test, y_pred, average=average)

split = "split_1"
band_type = "all"

dataset = torch.load("/content/drive/Shared drives/EEG_Aditya/data/EEG3DTIME_3SPLIT.pt")[split]
#train_dataloader = data.TensorDataset(dataset["train"][band_type], dataset["train"]["labels"])
#test_dataloader = data.TensorDataset(dataset["test"][band_type], dataset["test"]["labels"])

dataset = data.TensorDataset(dataset["train"][band_type], dataset["train"]["labels"])
train, val = data.random_split(dataset, [66, 11], generator=torch.Generator().manual_seed(42))
train_dataloader, val_dataloader = data.DataLoader(train), data.DataLoader(val)

model = ResNet()
wandb_logger = WandbLogger()
wandb_logger.watch(model, log='gradients', log_freq=100)
trainer = Trainer(max_epochs=1000, gpus= 8, logger=wandb_logger, progress_bar_refresh_rate=20)
trainer.fit(model, train_dataloader, val_dataloader)
print("Done training.")

# Debugger
bp()
