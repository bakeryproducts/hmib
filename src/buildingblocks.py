from functools import partial

import timm
import torch
from torch import nn as nn
from torch.nn import functional as F


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, dropout=0.2):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True)
        linear = nn.Linear(in_channels, classes, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class Adapter(torch.nn.Module):
    def __init__(self, decoder_channels, out_channel):
        super().__init__()
        head_dim = out_channel
        adapter_channels = decoder_channels[:-1]
        self.convs = nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(d, head_dim, (1,1)),
                torch.nn.BatchNorm2d(head_dim),
                torch.nn.ReLU(),
            )
            for d in adapter_channels])

    def forward(self, features):
        features = features[1:-1] # drop decoder input and full size
        rr = []
        assert len(features) == len(self.convs), (len(features),len(self.convs))
        for i, f in enumerate(features):
            r = self.convs[i](f)
            rr.append(r)
        return rr
