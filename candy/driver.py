from torch._C import Value
from xvision import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from enum import Enum
from easy_dict import EasyDict

from typing import List

class DriverEvent:
    EVENT_BEFORE_START=0
    EVENT_BEFORE_TRAIN_STEP=1
    EVENT_AFTER_TRAIN_STEP=2
    

def single_or_zip_dataset(dataset):
    if isinstance(dataset, list):
        return ZipDataset(datasets)
    elif isinstance(dataset, Dataset):
        return dataset
    else:
        raise ValueError(
            'only accept torch.utils.data.Dataset or list of Dataset, but given: {}'.format(type(dataset)))


class ZipDataset(Dataset):
    def __init__(self, datasets: list) -> None:
        super().__init__()


class TransformDataset(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()


class DriverOptions(EasyDict):
    def __init__(self) -> None:
        self.workdir = '.'
        self.max_epochs = math.inf
        self.batch_size = 64
        self.betas = (.5, .999)
        self.lr = 1e-3
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')


class Driver:
    def __init__(self, opts: DriverOptions, models,
                 train_datasets,
                 val_datasets,
                 test_datasets=None,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None) -> None:
        self.opts = EasyDict(opts)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.models = models.to(self.opts.device)
        self.val_interval = 1000

        self.train_loader = self._prepare_data_loader(
            train_datasets, train_transform)
        self.val_loader = self._prepare_data_loader(
            val_datasets, train_transform)
        self.test_loader = self._prepare_data_loader(
            test_datasets, test_transform)
        self.optimizers = self.configure_optimizers(self.models)

    def _prepare_data_loader(self, datasets, transform=None):
        dataset = single_or_zip_dataset(datasets)
        dataset = TransformDataset(dataset, transform)
        return dataset

    def configure_optimizers(self, models):
        return [
            torch.optim.Adam(m.parameters(), lr=self.opts.lr,
                             betas=self.opts.betas) for m in models
        ]

    def add_handler(self, event, callback):
        pass

    def on(self, event):
        """
        decorator to add event
        """

    def _optimize_step(self, losses):
        for optim in self.optimizers:
            optim.zero_grad()
        
        for loss in losses:
            loss.backward()
        
        for optim in self.optimizers:
            optim.step()

    def log(self, metrics):
        for k, v in metrics:
            print(f'{k}: {v.item()}')


    def save_checkpoint(self, epoch, step):
        pass

    
    def handle(self, event):
        pass
    

    def fit(self, train_step, val_step=None):
        

        for epoch in self.opts.max_epochs:
            for step, batch in enumerate(self.train_loader):
                output = train_step(self, self.models, batch, step)
                self._optimize_step(output['loss'])
                self.log(output['metric'])
                self.save_checkpoint(epoch, step)
            if val_step is not None:
                output = val_step(self, self.models, batch, i)
                self.log(output['metric'])



    def _prepare_data_loader(self):

        pass

    def _prepare_val_loader(self):
        pass

    def _main_train_loop(self):

        pass

        state = self.state
        self.step = 0
        while True:
            for batch in self._prepare_data_loader():
