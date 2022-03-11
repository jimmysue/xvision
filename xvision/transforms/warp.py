import cv2

from .matrix2d import translate


def _pixel_matrix(matrix):
    # matrix 是 浮点坐标 -> 浮点坐标, opencv 要 pixel 坐标 -> pixel -> 坐标
    # 因此需要需要如下过程
    # 1. pixel coords -> float coords         translate(0.5)
    # 2. float coords -> float coords         matrix
    # 3. float coords -> pixel coords         translate(-0.5)
    return translate(-0.5) @ matrix @ translate(0.5)


def warp_affine(src, M, dsize, *args, **kwargs):
    M = _pixel_matrix(M)
    return cv2.warpAffine(src, M[:2, :], dsize, *args, **kwargs)
