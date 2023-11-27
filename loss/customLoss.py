
"""define a class to compute loss"""

import torch
from torch import nn
from torch import Tensor
from torch.nn import MSELoss
from model.CNN.CCNN import CLayerNorm

size = 64

class CustomLoss(nn.Module):
    """define a class to compute loss"""
    def __init__(self) -> None:
        """initialize a loss instance"""
        super(CustomLoss, self).__init__()
        self.criterion = MSELoss()
        self.layernorm = CLayerNorm([size, size])

    @torch.compile()
    def forward(self, output: Tensor, label: Tensor) -> Tensor:
        """forward function of loss"""
        label = self.layernorm(label)
        loss = self.criterion(output, label)
        return loss
