import numpy as np


def identity():
    return np.eye(3, dtype=np.float32)


def scale(s):
    matrix = identity()
    np.fill_diagonal(matrix[:2, :2], s)
    return matrix


def translate(t):
    matrix = identity()
    matrix[:2, 2] = t
    return matrix


def rotate(radian):  # clockwise
    cos = np.cos(radian)
    sin = np.sin(radian)
    return np.array(
        [
            [cos, -sin, 0],
            [sin,  cos, 0],
            [0,    0, 1]
        ], dtype=np.float32)


def center_rotate_scale_cw(center, angle, s):
    center = np.array(center)
    return translate(center) @ rotate(angle * np.pi / 180) @ scale(s) @ translate(-center)


def hflip(width):
    return np.array([
        [-1, 0, width],
        [0,  1, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def vflip(height):
    return np.array([
        [1, 0, 0],
        [0, -1, height],
        [0, 0, 1]
    ], dtype=np.float32)


def shear(shr_x, shr_y):
    m = identity()
    m[0, 1] = shr_x
    m[1, 0] = shr_y
    return m
