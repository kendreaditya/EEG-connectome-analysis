import prebpl
import Densenet3D as dn
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

class Densenet(prebpl.PrebuiltLightningModule):
    def __init__(self, input_size=(1,30,34,34,130), growth_rate=12, block_config=(6, 12, 24), channel_num=256, bn_size=4, dropout_rate=0):
        super().__init__()
        self.model = dn.DenseNet(growth_rate=growth_rate, block_config=block_config, num_init_features=channel_num,
                                 bn_size=bn_size, drop_rate=dropout_rate, num_classes=3, memory_efficient=False,
                                 input_size=input_size, in_channels=input_size[1])

        # Model Tags
        self.set_model_tags(input_size)
        self.model_tags.append(self.file_name)

    def forward(self, X):
        X = self.model(X)
        return X

input_size = {"delta":3,
              "theta":4,
              "alpha":5,
              "beta":18,
              "all":3}

def train(split, band_type):
    # Model init
    model = Densenet(input_size=(1,input_size[band_type],34,34,130))
    train_dataset, validation_dataset, test_dataset = model.datasets("/content/drive/Shared drives/EEG_Aditya/data/EEG4DFREQ_3SPLIT.pt",
                                                                    split, band_type, [45, 21])

    train_dataloader, validation_dataloader, test_dataloader = model.dataloaders(train_dataset, validation_dataset, test_dataset,
                                                                                 batch_size=256)
    # Logging
    model.model_tags.append(split)
    model.model_tags.append(band_type)
    model.model_tags.append("train:"+str(len(train_dataset)))
    model.model_tags.append("validation:"+str(len(validation_dataset)))
    model.model_tags.append("test:"+str(len(test_dataset)))
    model.model_tags.append("seed:"+str(model.seed))

    wandb_logger = WandbLogger(name=model.model_name, tags=model.model_tags, project="eeg-connectome-analysis", save_dir="/content/drive/Shared drives/EEG_Aditya/model-results/wandb", log_model=True)
    wandb_logger.watch(model, log='gradients', log_freq=100)

    # Checkpoints
    val_loss_cp = pl.callbacks.ModelCheckpoint(monitor='validation-loss')

    trainer = pl.Trainer(max_epochs=1000, gpus=1, logger=wandb_logger, precision=16, fast_dev_run=False,
                         auto_lr_find=True, auto_scale_batch_size=True, log_every_n_steps=1,
                        checkpoint_callback=val_loss_cp)
    trainer.fit(model, train_dataloader, validation_dataloader)
    print("Done training.")

    print("Testing model on last epoch.")
    results_last = trainer.test(model, test_dataloader)
    model_path = val_loss_cp.best_model_path
    model_path = model_path[:model_path.rfind('/')]+"lastModel.ckpt"
    trainer.save_checkpoint(model_path)

    print(f"Testing model with best validation loss\t{val_loss_cp.best_model_score}.")
    model = model.load_from_checkpoint(val_loss_cp.best_model_path)
    results = trainer.test(model, test_dataloader)

    print("Done testing.")


# Training for all models
for band_type in ["delta", "theta", "alpha", "beta", "all"]:
    for split in ["split_1", "split_2", "split_3"]:
        for _ in range(10):
            train(split, band_type)
            print(f"Done with {band_type} with split {split}.")
