import prebpl
import sys
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

class ResNet(prebpl.PrebuiltLightningModule):
    def __init__(self, input_size=(1,1,130,34,34), in_channel=1):
        super().__init__()
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
        
        self.set_model_tags(input_size)
        self.model_tags.append(self.file_name)
        

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

# Model init
model = ResNet()

# Logging
model.model_tags.append(split)
model.model_tags.append(band_type)
model.model_tags.append("train:"+str(len(train_dataset)))
model.model_tags.append("validation:"+str(len(validation_dataset)))
model.model_tags.append("test:"+str(len(test_dataset)))

wandb_logger = WandbLogger(name=model.model_name, tags=model.model_tags, project="eeg-connectome-analysis", save_dir="/content/drive/Shared drives/EEG_Aditya/model-results/wandb", log_model=True)
wandb_logger.watch(model, log='gradients', log_freq=100)

# Checkpoints
val_loss_cp = pl.callbacks.ModelCheckpoint(monitor='validation-loss')

trainer = pl.Trainer(max_epochs=1000, gpus=1, logger=wandb_logger, precision=16, fast_dev_run=False,
                     auto_lr_find=False, auto_scale_batch_size=False, log_every_n_steps=1,
                    checkpoint_callback=val_loss_cp)
trainer.fit(model, train_dataloader, validation_dataloader)
print("Done training.")

print("Testing model on last epoch.")
trainer.test(model, test_dataloader)
model_path = val_loss_cp.best_model_path
model_path = model_path[:model_path.rfind('/')]+"lastModel.ckpt"
trainer.save_checkpoint(model_path)

print(f"Testing model with best validation loss\t{val_loss_cp.best_model_score}.")
model = model.load_from_checkpoint(val_loss_cp.best_model_path)
trainer.test(model, test_dataloader)

print("Done testing.")
