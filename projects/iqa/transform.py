from opencv_transforms import transforms


class Transform:
    def __init__(self, funcs):
        self.transforms = funcs

    def __call__(self, item):
        item['image'] = self.transforms(item['image'])
        return item


if __name__ == '__main__':
    pass
