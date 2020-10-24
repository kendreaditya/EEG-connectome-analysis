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

class PrebuiltLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Metrics 
        self.criterion = nn.CrossEntropyLoss()

        # Run Name
        self.set_model_name()
        self.set_model_notes(input_size)

    def set_model_name(self):
        file_name = sys.argv[0].split("/")[-1].replace(".py", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.model_name = f"{file_name}-{timestamp}"

    def set_model_notes(self, input_size):
        loss = str(self.criterion)
        optimizer = str("SGD")
        self.model_notes = f"{optimizer},{loss}"

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
            self.log(f"train-{key}", metrics[key], prog_bar=False, on_step=True, on_epoch=False)
        return metrics

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, prefix="validation-")
        metrics["validation-loss"] = loss
        return metrics

    def validation_epoch_end(self, outputs):
        avg_metrics = {key:[] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])
        avg_metrics = {key:torch.as_tensor(avg_metrics[key]).mean() for key in avg_metrics}
        for key in avg_metrics:
            self.log(key, avg_metrics[key], prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, prefix="test-")
        metrics["test-loss"] = loss
        return metrics

    def test_epoch_end(self, outputs):
        avg_metrics = {key:[] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])

        avg_metrics = {key:torch.as_tensor(avg_metrics[key]).mean() for key in avg_metrics}
        for key in avg_metrics:
            self.log(key, avg_metrics[key], prog_bar=False, on_step=False, on_epoch=True)

    def metrics_step(self, outputs, targets, prefix=""):
        pred = torch.argmax(outputs, dim=1)

        accuracy_score = accuracy(pred, targets)
        recall_score = recall(pred, targets, num_classes=3)
        precision_score = precision(pred, targets, num_classes=3)
        fbeta = fbeta_score(pred, targets, beta=0.5, num_classes=3)
        f1 = f1_score(pred, targets, num_classes=3)
        roc_auc_score = self.multiclass_roc_auc_score(targets, pred)

        learning_rate = self.optimizers().param_groups[0]['lr']

        return {f'{prefix}accuracy': accuracy_score,
                f'{prefix}recall': recall_score,
                f'{prefix}precision': precision_score,
                f'{prefix}fbeta': fbeta,
                f'{prefix}f1': f1,
                f'{prefix}ROC-AUC': roc_auc_score,
                f'{prefix}lr': learning_rate}

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        y_test, y_pred = y_test.cpu(), y_pred.cpu()
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return metrics.roc_auc_score(y_test, y_pred, average=average)
