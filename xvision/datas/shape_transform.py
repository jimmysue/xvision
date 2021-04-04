import cv2
import numpy as np
from numpy.core.fromnumeric import mean

from xvision.transforms import matrix2d
from xvision.transforms.warp import warp_affine

def _dsize(dsize):
    return dsize


class ShapeTransform(object):
    def __init__(self, dsize: int, padding: float, meanshape: np.ndarray, augments=None) -> None:
        super().__init__()
        self.dsize = dsize
        self.padding = float
        self.meanshape = meanshape
        self.augments = augments

    
    def __call__(self, item) -> dict:
        shape = item['shape']

        