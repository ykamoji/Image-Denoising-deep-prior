import numpy as np
import torch
import torch.nn as nn


class EncDec(nn.Module):

    def __init__(self):
        super(EncDec, self).__init__()

        self.resolutions = [1, 16, 32, 64]
        deep = len(self.resolutions) - 1

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.resolutions[i],
                          out_channels=self.resolutions[i + 1],
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          padding_mode='reflect',
                          ),
                nn.BatchNorm2d(self.resolutions[i+1]),
                nn.ReLU(),
            )
            for i in range(deep)
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels=self.resolutions[deep - i],
                          out_channels=self.resolutions[deep - 1 - i],
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect',
                          ),
                nn.BatchNorm2d(self.resolutions[deep - 1 - i]),
                nn.ReLU(),
            )
            for i in range(deep)
        ])

    def forward(self, x):

        for layers in self.encoder:
            x = layers(x)

        for layers in self.decoder:
            x = layers(x)

        return x