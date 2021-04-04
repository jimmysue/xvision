import torch
from torchvision.models import mobilenet_v2


def mbv2(num_points=106):
    return mobilenet_v2(num_classes=2 * num_points)


if __name__ == '__main__':
    model = mbv2()

    input = torch.rand(1, 3, 128, 128)
    pred = model(input)
    print(pred.shape)