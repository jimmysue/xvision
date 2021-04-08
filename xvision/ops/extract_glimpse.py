import torch
import torch.nn as nn

from typing import Union, Tuple

"""
tf.image.extract_glimpse(
    input, size, offsets, centered=True, normalized=True, noise='uniform',
    name=None
)
"""


def extract_glimpse(input: torch.Tensor, size: Tuple[int, int], offsets, centered=True, normalized=True, mode='bilinear'):
    # similar usage with ft.image.extract_glimpse:
    #   https://www.tensorflow.org/api_docs/python/tf/image/extract_glimpse
    # input: [B, C, H, W]
    # size:  [int, int]  specified the size of glimpse, height comes first
    # offsets: [B, 2]
    W, H = input.size(-1), input.size(-2)

    if normalized and centered:
        offsets = (offsets + 1) * offsets.new_tensor([W/2, H/2])
    elif normalized:
        offsets = offsets * offsets.new_tensor([W, H])
    elif centered:
        raise ValueError(
            f'Invalid parameter that offsets centered but not normlized')

    h, w = size
    xs = torch.arange(0, w, dtype=input.dtype,
                      device=input.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=input.dtype,
                      device=input.device) - (h - 1) / 2.0

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2

    offsets_grid = offsets[:, None, None, :] + grid[None, ...]

    # normalised grid  to [-1, 1]
    offsets_grid = (
        offsets_grid - offsets_grid.new_tensor([W/2, H/2])) / offsets_grid.new_tensor([W/2, H/2])

    return torch.nn.functional.grid_sample(
        input, offsets_grid, mode=mode, align_corners=False)


def extract_multiple_glimpse(input: torch.Tensor, size: Tuple[int, int], offsets, centered=True, normalized=True, mode='bilinear'):
    # offsets: [B, n, 2]
    patches = []
    for i in range(offsets.size(-2)):
        patch = extract_glimpse(
            input, size, offsets[:, i, :], centered, normalized, mode)
        patches.append(patch)
    return torch.stack(patches, dim=1)
