from torchvision.models.resnet import resnet18


def resfa(num_points=106):
    return resnet18(num_classes=num_points*2)


if __name__ == '__main__':
    import torch
    model = resfa()

    input = torch.rand(1, 3, 128, 128)
    model.eval()
    shape= model(input)
    
    print(shape.shape)