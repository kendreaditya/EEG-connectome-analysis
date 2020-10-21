import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score, fbeta_score
from sklearn import preprocessing, metrics
import numpy as np
from pdb import set_trace as bp

class ResNet(pl.LightningModule):
    def __init__(self, input_size=(1,1,130,34,34), in_channel=1):
        super().__init__()
        # Metrics 
        self.criterion = nn.CrossEntropyLoss()

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

        # Run Name
        self.set_model_name()
        self.set_model_notes()

    def set_model_name(self):
        file_name = sys.argv[0].split("/")[-1].replace(".py", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.model_name = f"{file_name}-{timestamp}"

    def set_model_notes(self):
        self.model_notes = str(self.optimizers).split("(")[0][:-1]+str(self.criterion)

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
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, "train")
        metrics["train-loss"] = loss
        for metric in metrics:
            self.log(metric, metrics[metric])
        return metrics

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, "validation")
        metrics["validation-loss"] = loss
        for metric in metrics:
            self.log(metric, metrics[metric])
        return metrics

    def validation_end(self, outputs):
        avg_metrics = {key:[] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])

        avg_metrics = {"avg-"+key:np.mean(avg_metrics[key]) for key in avg_metrics}
        return avg_metrics


    def metrics_step(self, outputs, targets, prefix):
        pred = torch.argmax(outputs, dim=1)

        accuracy_score = accuracy(pred, targets)
        recall_score = recall(pred, targets, num_classes=3)
        precision_score = precision(pred, targets, num_classes=3)
        fbeta = fbeta_score(pred, targets, beta=0.5, num_classes=3)
        f1 = f1_score(pred, targets, num_classes=3)
        roc_auc_score = self.multiclass_roc_auc_score(targets, pred)

        learning_rate = self.optimizers().param_groups[0]['lr']

        return {f'{prefix}-accuracy': accuracy_score,
                f'{prefix}-recall': recall_score,
                f'{prefix}-precision': precision_score,
                f'{prefix}-fbeta': fbeta,
                f'{prefix}-f1': f1,
                f'{prefix}-ROC-AUC': roc_auc_score,
                f'{prefix}-lr': learning_rate}

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        y_test, y_pred = y_test.cpu(), y_pred.cpu()
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return metrics.roc_auc_score(y_test, y_pred, average=average)

split = "split_1"
band_type = "all"

dataset = torch.load("/content/drive/Shared drives/EEG_Aditya/data/EEG3DTIME_3SPLIT.pt")[split]
#train_dataloader = data.TensorDataset(dataset["train"][band_type], dataset["train"]["labels"])
#test_dataloader = data.TensorDataset(dataset["test"][band_type], dataset["test"]["labels"])

dataset = data.TensorDataset(dataset["train"][band_type], dataset["train"]["labels"].long())
train, val = data.random_split(dataset, [55, 11], generator=torch.Generator().manual_seed(42))
train_dataloader, val_dataloader = data.DataLoader(train, batch_size=1024), data.DataLoader(val, batch_size=1024)

model = ResNet()
wandb_logger = WandbLogger(name=model.model_name, notes=model.model_notes, project="eeg-connectome-analysis", save_dir="/content/drive/Shared drives/EEG_Aditya/model-results/wandb", log_model=True)
wandb_logger.watch(model, log='gradients', log_freq=100)
trainer = Trainer(max_epochs=1000, gpus=1, logger=wandb_logger)
trainer.fit(model, train_dataloader, val_dataloader)
print("Done training.")

# Debugger
bp()
# ROC Graph
# X: Recall (sensitivity)
# y: Precision (weighted 1-specificity)
