
"""A custom neural network model."""

import torch
from torch import nn
from torch import Tensor
from config.configClass import Config
from . import CCNN as cc

class ToHiddenLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(ToHiddenLayer, self).__init__()
        self.layernorm = cc.CLayerNorm([80, 80])
        self.conv1 = cc.CConv3x3(             1, hidden_channel, stride=2, padding=1) # 80 -> 40
        self.conv2 = cc.CConv3x3(hidden_channel, hidden_channel, stride=2, padding=1) # 40 -> 20

    def forward(self, x:Tensor) -> Tensor:
        out = self.layernorm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class PropagationLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(PropagationLayer, self).__init__()
        self.layernorm = cc.CLayerNorm([20, 20])
        self.conv1 = cc.CResBlock2d(hidden_channel, hidden_channel) # 20 -> 20
        self.conv2 = cc.CResBlock2d(hidden_channel, hidden_channel) # 20 -> 20

    def forward(self, x:Tensor) -> Tensor:
        out = self.layernorm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class SlitLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(SlitLayer, self).__init__()
        self.layernorm = cc.CLayerNorm([10, 20])
        self.conv1 = cc.CResBlock2d(hidden_channel, hidden_channel) # 20 -> 20
        self.conv2 = cc.CResBlock2d(hidden_channel, hidden_channel) # 20 -> 20

    def forward(self, x:Tensor) -> Tensor:
        out = self.layernorm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class ProjectionLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(ProjectionLayer, self).__init__()
        self.Tconv1 = cc.CConvTrans3x3(hidden_channel, hidden_channel, stride=2, padding=1, output_padding=1) # 20 -> 40
        self.Tconv2 = cc.CConvTrans3x3(hidden_channel,              1, stride=2, padding=1, output_padding=1) # 40 -> 80
        self.layernorm = cc.CLayerNorm([80, 80])

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
        self.step_N = 1

        hidden_channel = conf.hidden_channel
        self.tohidden = ToHiddenLayer(hidden_channel)
        self.pre_propagation = PropagationLayer(hidden_channel)
        self.branch1 = SlitLayer(hidden_channel)
        self.branch2 = SlitLayer(hidden_channel)
        self.fusion = PropagationLayer(hidden_channel)
        self.simulation = PropagationLayer(hidden_channel)
        self.project = ProjectionLayer(hidden_channel)

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        out = self.tohidden(x)
        out = self.pre_propagation(out)
        out1 = self.branch1(out[:,:,:10])
        out2 = self.branch2(out[:,:,10:])
        out = torch.cat((out1,out2),2)
        out = self.fusion(out)
        for _ in range(self.step_N):
            out = self.simulation(out)
        out = self.project(out)
        return out

