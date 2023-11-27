
"""A custom neural network model."""

import torch
from torch import nn
from torch import Tensor
from config.configClass import Config
from .CNN import CCNN as cc

size = 64

class ToHiddenLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(ToHiddenLayer, self).__init__()
        self.layernorm = cc.CLayerNorm([size, size])
        self.conv1 = cc.CConv3x3(             1, hidden_channel, stride=2, padding=1) # 64 -> 32
        self.conv2 = cc.CConv3x3(hidden_channel, hidden_channel, stride=2, padding=1) # 32 -> 16

    def forward(self, x:Tensor) -> Tensor:
        out = self.layernorm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class PropagationLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(PropagationLayer, self).__init__()
        self.layernorm = cc.CLayerNorm([size//4, size//4])
        self.conv1 = cc.CResBlock2d(hidden_channel, hidden_channel) # 16 -> 16
        self.conv2 = cc.CResBlock2d(hidden_channel, hidden_channel) # 16 -> 16
        self.conv3 = cc.CResBlock2d(hidden_channel, hidden_channel) # 16 -> 16

    def forward(self, x:Tensor) -> Tensor:
        out = self.layernorm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class ProjectionLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(ProjectionLayer, self).__init__()
        self.Tconv1 = cc.CConvTrans3x3(hidden_channel, hidden_channel, stride=2, padding=1, output_padding=1) # 16 -> 32
        self.Tconv2 = cc.CConvTrans3x3(hidden_channel,              1, stride=2, padding=1, output_padding=1) # 32 -> 64
        self.layernorm = cc.CLayerNorm([size, size])

    def forward(self, x:Tensor) -> Tensor:
        out = self.Tconv1(x)
        out = self.Tconv2(out)
        out = self.layernorm(out)
        return out


class CustomModel(nn.Module):
    def __init__(self, conf:Config):
        """Initialize a neural network model."""
        super(CustomModel, self).__init__()
        self.conf = conf

        hidden_channel = conf.hidden_channel
        self.tohidden = ToHiddenLayer(hidden_channel)
        self.simulation = PropagationLayer(hidden_channel)
        self.project = ProjectionLayer(hidden_channel)

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        out = self.tohidden(x)
        out = self.simulation(out)
        out = self.project(out)
        return out
