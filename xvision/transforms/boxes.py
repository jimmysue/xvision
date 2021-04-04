import numpy as np

# naming convention:
# bbox means bounding box encoded with bounding points
# cbox means bounding box encoded with center and size
# abox means affine box encoded as affine matrix
#                                  [  u_x,   u_y, cx]
#                                  [  v_x,   v_y, cy]
#                                  [    0,     0,  1]
# rbox means bounding rotated box encoding with [cx, cy, w, h, r]
# where r is box rotated angle in radian, and the anchor is clockwise angle
# in image coordinate

# copy from https://github.com/AffineVision/AffineVision/blob/master/affinevision/transforms/boxes.py


def bbox2abox(bboxes, radians=None):
    # bboxes: [*, 4]
    # radians: box angle in radian
    vectors = (bboxes[..., 2:] - bboxes[..., :2]) / 2
    centers = (bboxes[..., 2:] + bboxes[..., :2]) / 2

    x_vec = vectors[..., 0]
    y_vec = vectors[..., 1]
    zeros = np.zeros(x_vec.shape)
    ones = np.ones(x_vec.shape)
    aboxes = np.stack([x_vec, zeros, centers[..., 0], zeros,
                       y_vec, centers[..., 1], zeros, zeros, ones], axis=-1)
    # reshape
    shape = (*x_vec.shape, 3, 3)
    aboxes = aboxes.reshape(shape)

    if radians is not None:
        cos = np.cos(radians)
        sin = np.sin(radians)
        rotate = np.stack([cos, -sin, zeros, sin, cos, zeros,
                           zeros, zeros, ones], axis=-1).reshape(shape)
        aboxes = rotate @ aboxes

    return aboxes


def abox2bbox(aboxes):
    """covnert affine boxes to bounding point box

    reference: https://www.iquilezles.org/www/articles/ellipses/ellipses.htm

    Args:
        aboxes ([np.ndarray]): affine boxes shape with [*, 3, 3]
    """

    c = aboxes[..., :2, 2]
    e = np.linalg.norm(aboxes[..., :2, :2], ord=2, axis=-1)
    bboxes = np.concatenate([c - e, c + e], axis=-1)
    return bboxes


def bbox2cbox(bboxes):
    sizes = bboxes[..., 2:] - bboxes[..., :2]
    centers = (bboxes[..., :2] + bboxes[..., 2:]) / 2
    cboxes = np.concatenate([centers, sizes], axis=-1)
    return cboxes


def cbox2bbox(cboxes):
    halfs = cboxes[..., 2:] / 2
    lt = cboxes[..., :2] - halfs
    rb = cboxes[..., 2:] + halfs
    cboxes = np.concatenate([lt, rb], axis=-1)
    return cboxes


def cbox2abox(cboxes):
    bboxes = cbox2bbox(cboxes)
    return bbox2abox(bboxes)


def abox2cbox(aboxes):
    bboxes = abox2bbox(aboxes)
    return bbox2cbox(bboxes)


def rbox2abox(rboxes):
    radians = rboxes[:, -1]
    cboxes = rboxes[:, :4]
    bboxes = cbox2bbox(cboxes)
    aboxes = bbox2abox(bboxes, radians)
    return aboxes


def abox2rbox(aboxes):
    # aboxes [*, 3, 3]
    radians = np.arctan2(aboxes[..., 1, 0], aboxes[..., 0, 1])
    sizes = np.linalg.norm(aboxes[..., :2, :2], ord=2, axis=-2)
    centers = aboxes[..., :2, 2]
    rboxes = np.concatenate([centers, sizes, radians[..., None]], axis=-1)
    return rboxes


def bbox_affine(bboxes, matrix):
    aboxes = bbox2abox(bboxes)
    aboxes = matrix @ aboxes
    return abox2bbox(aboxes)


def bbox2rect(bboxes):
    # [*, 4]
    sizes = bboxes[..., 2:] - bboxes[..., :2]
    return np.concatenate([bboxes[..., :2], sizes], axis=-1)
