import numpy as np
import tqdm
from sklearn.preprocessing import normalize

from xvision.transforms.umeyama import umeyama
from xvision.transforms import matrix2d


def to_unit_shape(shape):
    lt = shape.min(axis=0)
    rb = shape.max(axis=0)
    ctr = (rb + lt) / 2.0
    size = (rb - lt).max()
    scale = 1 / size
    m = matrix2d.translate(
        [0.5, 0.5]) @ matrix2d.scale(scale) @ matrix2d.translate(-ctr)
    return shape @ m[:2, :2].T + m[:2, 2]


def procrustes(shapes, iter=3):
    # shapes: [n, p, 2]
    shapes = shapes.copy()
    meanshape = shapes.mean(axis=0)
    meanshape = normalize(meanshape)

    for _ in tqdm.tqdm(range(iter)):
        for i, shape in enumerate(shapes):
            m = umeyama(shape, meanshape)
            shape = shape @ m[:2, :2].T + m[:2, 2]
            shapes[i, ...] = shape
        meanshape = shapes.mean(axis=0)
        meanshape = normalize(meanshape)
    return shapes


def calc_mean_shape(shapes: np.ndarray, iter=2):
    # shapes: [n, p, 2]
    shapes = procrustes(shapes)
    meanshape = shapes.mean(axis=0)
    return to_unit_shape(meanshape)


def shape_affine(shape, matrix):
    return shape @ matrix[:2, :2].T + matrix[:2, 2]


if __name__ == '__main__':
    from xvision.utils.draw import draw_shapes
    from xvision.datasets.jdlmk import JDLandmark

    root = '/Users/jimmy/Documents/data/FA/IBUG'
    landmark = '/Users/jimmy/Documents/data/FA/JD-landmark/Train/landmark'
    picture = '/Users/jimmy/Documents/data/FA/JD-landmark/Train/picture'
    data = JDLandmark(landmark, picture)

    meanshape = calc_mean_shape(data.shape)