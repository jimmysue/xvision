import cv2
import numpy as np
from numbers import Number
from typing import Dict
from xvision.transforms.umeyama import umeyama
from xvision.transforms.warp import warp_affine
from xvision.transforms import matrix2d
from xvision.transforms.shapes import *
from xvision.utils.draw import draw_points


def _to_size(dsize):
    if isinstance(dsize, Number):
        dsize = (int(dsize), int(dsize))
    elif isinstance(dsize, (tuple, list)) and len(dsize) == 2:
        width, height = dsize
        dsize = (int(width), int(height))
    else:
        raise ValueError(f'Invalid dsize: {dsize}')
    return dsize


class CacheTransform(object):
    def __init__(self, dsize, padding, meanshape) -> None:
        super().__init__()
        self.dsize = _to_size(dsize)
        self.padding = padding
        self.meanshape = meanshape
        meanshape = np.array(meanshape)
        unit = to_unit_shape(meanshape)
        self.cache_size = tuple(
            (np.array(self.dsize) * 2).astype(np.int32).tolist())
        self.ref = (unit - 0.5) * (1 - padding) * \
            np.array(dsize).min() + np.array(dsize)

    def __call__(self, item) -> Dict:
        image = cv2.imread(str(item['path']), cv2.IMREAD_COLOR)
        shape = item['shape']
        matrix = umeyama(shape, self.ref)
        image = warp_affine(image, matrix, self.cache_size)
        shape = shape_affine(shape, matrix)
        return {
            'image': image,
            'shape': shape
        }


class Transform(object):
    def __init__(self, dsize, padding, meanshape, symmetry=None, augments=None) -> None:
        super().__init__()
        dsize = _to_size(dsize)
        meanshape = np.array(meanshape)
        unit = to_unit_shape(meanshape)
        ref = (unit - 0.5) * (1 - padding) * np.array(dsize).min() + np.array(dsize) / 2
        self.ref = ref
        self.dsize = dsize
        self.padding = padding
        self.augments = augments
        self.symmetry = symmetry

    def _augment(self, item, matrix):
        r = self.augments['rotate']
        s = self.augments['scale']
        t = self.augments['translate']
        symmetry = self.symmetry
        scale = np.exp(np.random.uniform(-np.log(1 + s), np.log(1 + s)))
        translate = np.random.uniform(-t, t) * np.array(self.dsize)
        angle = np.random.uniform(-r, r)
        jitter = matrix2d.translate(
            translate) @ matrix2d.center_rotate_scale_cw(np.array(self.dsize)/2, angle, scale)

        if self.symmetry and np.random.choice([True, False]):
            # mirror
            jitter = matrix2d.hflip(self.dsize[0]) @ jitter
            shape = item['shape']
            shape = shape[..., symmetry, :]
            item['shape'] = shape
        return jitter @ matrix

    def __call__(self, item) -> dict:
        image = item['image']
        shape = item['shape']

        matrix = umeyama(shape, self.ref)

        if self.augments:
            matrix = self._augment(item, matrix)
        image = item['image']
        shape = item['shape']
        image = warp_affine(image, matrix, self.dsize)
        shape = shape_affine(shape, matrix)

        return {
            'image': image,
            'shape': shape.astype(np.float32)
        }


if __name__ == '__main__':
    import torch
    from xvision.datasets.jdlmk import JDLandmark
    from xvision.utils.draw import draw_shapes
    from config import cfg
    from torch.utils.data import DataLoader
    from xvision.models.fa import resfa, mbv2
    from xvision.datasets.mmap import MMap
    root = '/Users/jimmy/Documents/data/FA/IBUG'
    landmark = '/Users/jimmy/Documents/data/FA/FLL2/landmark'
    picture = '/Users/jimmy/Documents/data/FA/FLL2/picture'
    ctrans = CacheTransform(cfg.size, cfg.padding, cfg.meanshape)

    MMap.create('cache.npy', JDLandmark.parse(landmark, picture), ctrans, 6)

    transform = Transform(cfg.dsize, cfg.padding, cfg.meanshape, cfg.augments)
    data = JDLandmark(landmark, picture, transform)
    loader = DataLoader(data, batch_size=1)

    model = mbv2()

    state = torch.load(
        '/Users/jimmy/Documents/github/xvision/workspace/fa/step-00016600.pth', map_location='cpu')
    model.load_state_dict(state['model'])
    model.eval()

    for item in data:

        image = item['image']
        shape = item['shape']

        tensor = torch.from_numpy(image).unsqueeze(
            0).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            pred = model(tensor) * 128
        pred = pred.detach().numpy().reshape(-1, 2)
        draw_points(image, pred, (0, 0, 255))
        draw_points(image, shape, (0, 255, 0))
        cv2.imshow("v", image)
        k = cv2.waitKey()
        if k == ord('q'):
            exit()
