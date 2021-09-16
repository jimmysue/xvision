from candy.driver import *

import torch.nn as nn

from torchvision.datasets.mnist import MNIST

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, )

    
    