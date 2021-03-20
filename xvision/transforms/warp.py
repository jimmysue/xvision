import cv2

from .matrix2d import translate


def _pixel_matrix(matrix):
    return translate(-0.5) @ matrix @ translate(0.5)


def warp_affine(src, M, dsize, *args, **kwargs):
    M = _pixel_matrix(M)
    return cv2.warpAffine(src, M[:2, :], dsize, *args, **kwargs)
