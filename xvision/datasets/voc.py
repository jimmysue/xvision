from torchvision.datasets import VOCDetection, voc



if __name__ == '__main__':
    data = VOCDetection('./workspace/voc/', download=True)

    for img, ann in data:
        print(ann)