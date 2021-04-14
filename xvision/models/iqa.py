import torch
import torch.nn as nn


class IQA(nn.Module):
    def __init__(self, backbone, head=None) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        x = self.backbone(x)
        y = self.head(x)
        return y