import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

class MLP(LightningModule):
    def __init__(self, layer_sizes, lr=3e-4) -> None:
        super().__init__()
        self.lr = lr

        self.layers = nn.Sequential()
        self.layers.append(nn.BatchNorm1d(layer_sizes[0]))
        for i, l in enumerate(layer_sizes[:-1]):
            self.layers.append(nn.Linear(l, layer_sizes[i+1]))
            self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            self.layers.append(nn.ELU())

        self.layers.append(nn.Linear(layer_sizes[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log(stage + '_loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')


class SimpleResNetBlock(nn.Module):
    '''
    A simple fully connected ResNet block.
    '''
    def __init__(self, dim, activation=nn.ELU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            activation(),
        )

    def forward(self, x):
        return self.block(x) + x

class SimpleResNet(LightningModule):
    def __init__(self, depth, layer_size, input_dim, lr=3e-4) -> None:
        super().__init__()
        self.lr = lr

        self.layers = nn.Sequential()
        self.layers.append(nn.BatchNorm1d(input_dim))
        self.layers.append(nn.Linear(input_dim, layer_size))
        for i in range(depth):
            self.layers.append(SimpleResNetBlock(layer_size))

        self.layers.append(nn.Linear(layer_size, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log(stage + '_loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')