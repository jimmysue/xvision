import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d


def seperable_conv(inchannels, outchannels):
    return nn.Sequential(
        nn.Conv2d(inchannels, outchannels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outchannels),
        nn.ReLU(),
        nn.Conv2d(outchannels, outchannels, 3, 1, 1, groups=outchannels, bias=False),
        nn.BatchNorm2d(outchannels),
        nn.ReLU()
    )

class DULink(nn.Module):
    def __init__(self, inc, outc, factor=2, transform=None):
        super().__init__()
        self.factor =  factor
        self.transform = transform
        self.pool = nn.AvgPool2d(kernel_size=factor, stride=2)
        self.transform = transform
        self.conv_fn = seperable_conv(inc + 3, outc)

    def forward(self, image):
        if self.transform is None:
            return self.conv_fn(image)
        else:
            down_image = self.pool(image)
            down_feature = self.transform(down_image)

            up_image = F.upsample_nearest(down_image, scale_factor=self.factor)
            up_feature = F.upsample_nearest(down_feature, scale_factor=self.factor)
            dimage = image - up_image
            feature = torch.cat([dimage, up_feature], dim=1)
            return self.conv_fn(feature)

        

class DUnet(nn.Module):
    def __init__(self):
        super().__init__()
        link = DULink(0, 8, 2, None)
        for _ in range(4):
            link = DULink(8, 8, 2, link)
        self.link = link

    def forward(self, x):
        return self.link(x)


if __name__ == '__main__':
    from torch.onnx import export
    from thop import profile, clever_format
    model = DUnet()
    model.eval()

    input = torch.rand(1, 3, 512, 512)
    res = profile(model, (input,))
    print(clever_format(res[0]))
    print(clever_format(res[1]))
    r = model(input)
    export(model, input, 'dunet.onnx')
