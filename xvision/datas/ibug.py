from xvision.utils.draw import draw_shapes
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class IBUG(Dataset):
    TRAIN_DIRS = ['afw', 'halen/trainset',  'lfpw/trainset']
    TEST_DIRS = ['ibug', 'halen/testset', 'lpfw/testset', ]
    EXTS = ['.jpg', '.png']

    def __init__(self, root, transform=None, testset=False) -> None:
        super().__init__()
        self.testset = testset
        if transform:
            assert callable(transform), 'transform should be callable'
        self.transform = transform
        if self.testset:
            self.data = list(self.parse(root, IBUG.TEST_DIRS))
        else:
            self.data = list(self.parse(root, IBUG.TRAIN_DIRS))

    @property
    def shapes(self):
        shapes = [v['shape'] for v in self.data]
        return np.stack(shapes, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        item = self.data[index].copy()
        item['image'] = cv2.imread(str(item['path']), cv2.IMREAD_COLOR)
        if self.transform:
            item = self.transform(item)
        return item

    @staticmethod
    def parse(root, dirs):
        root = Path(root)
        dirs = [root/dir for dir in dirs]
        for dir in dirs:
            for ext in IBUG.EXTS:
                for imgpath in dir.rglob(f'*{ext}'):

                    txt = imgpath.with_suffix('.pts')
                    with open(txt, 'rt') as f:
                        lines = f.readlines()
                        coords = lines[3:3+68]
                        coords = [l.strip().split() for l in coords]
                        shape = np.array(coords).astype(np.float32).reshape(-1, 2)
                        yield {
                            'path': imgpath,
                            'shape': shape -1 # 0 base
                        }


if __name__ == '__main__':
    from xvision.utils.draw import draw_points
    root = '/Users/jimmy/Documents/data/FA/IBUG'

    data = IBUG(root, testset=True)

    for item in data:
        image = item['image']
        shape = item['shape']
        lt = shape.min(axis=0)
        rb = shape.max(axis=0)
        size = (rb - lt).max()
        radius =int( max(2, size * 0.03))
        draw_shapes(image, shape, radius)
        cv2.imshow("v", image)
        k = cv2.waitKey()
        if k == ord('q'):
            break
