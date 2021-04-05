import cv2
import numpy as np
from numbers import Number

from xvision.transforms.umeyama import umeyama
from xvision.transforms.warp import warp_affine
from xvision.transforms import matrix2d
from xvision.transforms.shapes import *


class Transform(object):
    def __init__(self, dsize, padding, meanshape, augments=None) -> None:
        super().__init__()
        if isinstance(dsize, Number):
            dsize = (int(dsize), int(dsize))
        elif isinstance(dsize, (tuple, list)) and len(dsize) == 2:
            width, height = dsize
            dsize = (int(width), int(height))
        else:
            raise ValueError(f'Invalid dsize: {dsize}')
        meanshape = np.array(meanshape)
        unit = to_unit_shape(meanshape)
        ref = (unit - 0.5) * (1 - padding) * \
            np.array(dsize).min() + np.array(dsize) / 2
        self.ref = ref
        self.dsize = dsize
        self.padding = padding
        self.augments = augments

    def __call__(self, item) -> dict:
        image = item['image']
        shape = item['shape']
        matrix = umeyama(shape, self.ref)

        image = warp_affine(image, matrix, self.dsize)
        shape = shape_affine(shape, matrix)

        return {
            'image': image,
            'shape': shape.astype(np.float32)
        }


if __name__ == '__main__':
    from xvision.datasets.jdlmk import JDLandmark
    from xvision.utils.draw import draw_shapes
    from config import cfg
    root = '/Users/jimmy/Documents/data/FA/IBUG'
    landmark = '/Users/jimmy/Documents/data/FA/JD-landmark/Train/landmark'
    picture = '/Users/jimmy/Documents/data/FA/JD-landmark/Train/picture'
    data = JDLandmark(landmark, picture)

    transform = Transform((128, 128), 0.2, cfg.meanshape)

    data.transform = transform

    for item in data:
        image = item['image']
        shape = item['shape']
        lt = shape.min(axis=0)
        rb = shape.max(axis=0)
        size = (rb - lt).max()
        radius = int(max(2, size * 0.03))
        draw_shapes(image, shape, radius)
        cv2.imshow("v", image)
        k = cv2.waitKey(500)
        if k == ord('q'):
            break
