import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
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
        self.set_model_notes(input_size)

    def set_model_name(self):
        file_name = sys.argv[0].split("/")[-1].replace(".py", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.model_name = f"{file_name}-{timestamp}"

    def set_model_notes(self, input_size):
        optimizer = self.optimizers.__class__
        loss = str(self.criterion)
        self.model_notes = f"{optimizer},{loss}"

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
        optimzer = torch.optim.SGD(self.parameters(), lr=1e-5, momentum=1)
        return optimzer

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets)
        metrics["loss"] = loss
        for key in metrics:
            self.log(f"train-{key}", metrics[key], prog_bar=True, on_step=True, on_epoch=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets)
        metrics["loss"] = loss
        return metrics

    def validation_epoch_end(self, outputs):
        avg_metrics = {key:[] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])
        avg_metrics = {"avg-validation-"+key:torch.as_tensor(avg_metrics[key]).mean() for key in avg_metrics}
        for key in avg_metrics:
            self.log(key, avg_metrics[key], prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets)
        metrics["loss"] = loss
        return metrics

    def test_epoch_end(self, outputs):
        avg_metrics = {key:[] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])

        avg_metrics = {"avg-test"+key:torch.as_tensor(avg_metrics[key]).mean() for key in avg_metrics}
        for key in avg_metrics:
            self.log(key, avg_metrics[key], prog_bar=True, on_step=False, on_epoch=True)

    def metrics_step(self, outputs, targets):
        pred = torch.argmax(outputs, dim=1)

        accuracy_score = accuracy(pred, targets)
        recall_score = recall(pred, targets, num_classes=3)
        precision_score = precision(pred, targets, num_classes=3)
        fbeta = fbeta_score(pred, targets, beta=0.5, num_classes=3)
        f1 = f1_score(pred, targets, num_classes=3)
        roc_auc_score = self.multiclass_roc_auc_score(targets, pred)

        learning_rate = self.optimizers().param_groups[0]['lr']

        return {'accuracy': accuracy_score,
                'recall': recall_score,
                'precision': precision_score,
                'fbeta': fbeta,
                'f1': f1,
                'ROC-AUC': roc_auc_score,
                'lr': learning_rate}

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        y_test, y_pred = y_test.cpu(), y_pred.cpu()
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return metrics.roc_auc_score(y_test, y_pred, average=average)

split = "split_1"
band_type = "all"

# Datasets
dataset = torch.load("/content/drive/Shared drives/EEG_Aditya/data/EEG3DTIME_3SPLIT.pt")[split]
train_val_dataset = data.TensorDataset(dataset["train"][band_type], dataset["train"]["labels"].long())
train_dataset, validation_dataset = data.random_split(train_val_dataset, [44, 22], generator=torch.Generator().manual_seed(1))
test_dataset = data.TensorDataset(dataset["test"][band_type], dataset["test"]["labels"].long())

# Dataloaders
train_dataloader = data.DataLoader(test_dataset, batch_size=256)
validation_dataloader = data.DataLoader(validation_dataset, batch_size=256)
test_dataloader = data.DataLoader(train_dataset, batch_size=256)

run_notes =f"{split},{band_type},({len(train_dataset)},{len(validation_dataset)},{len(test_dataset)})"

model = ResNet()
wandb_logger = WandbLogger(name=model.model_name, notes=run_notes+","+model.model_notes, project="eeg-connectome-analysis", save_dir="/content/drive/Shared drives/EEG_Aditya/model-results/wandb", log_model=True)
wandb_logger.watch(model, log='gradients', log_freq=100)
trainer = pl.Trainer(max_epochs=1000, gpus=1, logger=wandb_logger, precision=16, fast_dev_run=False)
trainer.fit(model, train_dataloader, validation_dataloader)
print("Done training.")
trainer.test(model, test_dataloader)
print("Done testing.")
# Debugger
bp()
# ROC Graph
# X: Recall (sensitivity)
# y: Precision (weighted 1-specificity)