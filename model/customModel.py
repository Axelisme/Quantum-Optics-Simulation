
"""A custom neural network model."""

import torch
from torch import nn
from torch import Tensor
from config.configClass import Config
from . import CCNN as cc


class SimulationLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SimulationLayer, self).__init__()
        inner_channel = (in_channel + out_channel) // 2
        self.layer0 = cc.CResBlock(in_channel, inner_channel)
        self.layer1 = cc.CResBlock(inner_channel, out_channel)

    def forward(self, x:Tensor) -> Tensor:
        out = self.layer0(x)
        out = self.layer1(out)
        return out


class ToHiddenLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(ToHiddenLayer, self).__init__()
        self.conv1 = cc.CConv3x3x3(             1, hidden_channel, stride=2, padding=1) # 80 -> 40
        self.conv2 = cc.CConv3x3x3(hidden_channel, hidden_channel, stride=2, padding=1) # 40 -> 20

    def forward(self, x:Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ProjectionLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(ProjectionLayer, self).__init__()
        self.Tconv1 = cc.CConvTrans3x3x3(hidden_channel, hidden_channel, stride=2, padding=1, output_padding=1) # 20 -> 40
        self.Tconv2 = cc.CConvTrans3x3x3(hidden_channel,              1, stride=2, padding=1, output_padding=1) # 40 -> 80

    def forward(self, x:Tensor) -> Tensor:
        out = self.Tconv1(x)
        out = self.Tconv2(out)
        return out


class CustomModel(nn.Module):
    def __init__(self, conf:Config):
        """Initialize a neural network model."""
        super(CustomModel, self).__init__()
        self.conf = conf

        hidden_channel = conf.hidden_channel
        self.to_hidden = ToHiddenLayer(hidden_channel)
        self.simulation = SimulationLayer(hidden_channel, hidden_channel)
        self.project = ProjectionLayer(hidden_channel)

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        out = self.to_hidden(x)
        out = self.simulation(out)
        out = self.simulation(out)
        out = self.project(out)
        return out

