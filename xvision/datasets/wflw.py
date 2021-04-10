import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from xvision.transforms.shapes import calc_mean_shape
from xvision.transforms.boxes import bbox_affine
from xvision.transforms.umeyama import umeyama


class WFLW(Dataset):
    """WFLW dataset
    """
    __symmetry__ = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                    17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                    0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41,
                    40, 39, 38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69,
                    68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65, 82, 81, 80, 79,
                    78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93,
                    97, 96]

    def __init__(self, label_file, image_dir, transform=None) -> None:
        super().__init__()
        if transform:
            assert callable(transform), 'transform should be callable'
        self.data = list(self.parse(label_file, image_dir))
        self.transform = transform

    @property
    def shapes(self):
        shapes = [v['shape'] for v in self.data]
        return np.stack(shapes, axis=0)

    @property
    def meanshape(self):
        shapes = data.shapes
        mirrors = shapes[:, self.__symmetry__, :]
        mirrors[:, :, 0] = -mirrors[:, :, 0]
        shapes = np.concatenate([shapes, mirrors], axis=0)
        return calc_mean_shape(shapes)

    @property
    def meanbbox(self):
        meanshape = self.meanshape
        bboxes = []
        for item in self.data:
            shape = item['shape']
            bbox = item['bbox']  # n, 4
            matrix = umeyama(shape, meanshape)
            bbox = bbox_affine(bbox, matrix)
            bboxes.append(bbox)
        return np.stack(bboxes, 0).mean(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index].copy()
        item['image'] = cv2.imread(str(item['path']), cv2.IMREAD_COLOR)
        if self.transform:
            item = self.transform(item)
        return item

    @staticmethod
    def parse(label_file, image_dir):
        image_dir = Path(image_dir)
        with open(label_file) as f:
            for line in f:
                items = line.strip().split()
                shape = np.array(items[:196]).astype(np.float32).reshape(-1, 2)
                bbox = np.array(
                    items[196:196+4]).astype(np.float32).reshape(-1)
                attr = [int(v) for v in items[200:200+6]]
                keys = ['pose', 'expression', 'illumination',
                        'makeup', 'occlusion', 'blur']
                item = dict(zip(keys, attr))
                item['shape'] = shape
                item['bbox'] = bbox
                item['path'] = image_dir / items[-1]
                yield item
