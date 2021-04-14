import torch.nn as nn

"""
file - model.py
Implements the aesthemic model and emd loss used in paper.

Copyright (C) Yunxiao Shi 2017 - 2020
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class NIMA(nn.Module):

    """Neural IMage Assessment model by Google"""

    def __init__(self, base_model, num_classes=10, dropout=.75):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax(dim=-1))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def nima_vgg16(pretrained=False, **kwargs):
    flag = pretrained == True
    base_model = models.vgg16(pretrained=flag)
    if isinstance(pretrained, str):
        state = torch.load(pretrained)
        base_model.load_state_dict(state)

    return NIMA(base_model, **kwargs)


if __name__ =='__main__':
    model = nima_vgg16(True)
    model.eval()
    input = torch.rand(1, 3, 224,224)
    ret = model(input)
    print(ret.shape)