import torch

from torchvision.ops.boxes import box_area, box_iou as box_iou_pairwise


def bbox2cbox(bboxes):
    # bboxes: [*, 4]
    sizes = bboxes[..., 2:] - bboxes[..., :2]
    centers = (bboxes[..., :2] + bboxes[..., 2:]) / 2
    cboxes = torch.cat([centers, sizes], dim=-1)
    return cboxes


def cbox2bbox(cboxes):
    halfs = cboxes[..., 2:] / 2
    lt = cboxes[..., :2] - halfs
    rb = cboxes[..., :2] + halfs
    bboxes = torch.cat([lt, rb], dim=-1)
    return bboxes


def box_iou_itemwise(bboxes1, bboxes2):
    # bboxes1: [*, 4]
    # bboxes2: [*, 4]
    area1 = box_area(bboxes1)
    area2 = box_area(bboxes2)

    lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
    rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])

    sizes = rb - lt
    sizes = torch.clamp_min(sizes, 0)

    inter_area = torch.prod(sizes, dim=-1)

    return inter_area / (area1 + area2 - inter_area)


def box_diou_pairwise(bboxes1, bboxes2):
    # bboxes []
    raise NotImplementedError
    pass


def box_diou_itemwise(bboxes1, bboxes2):
    raise NotImplementedError
    pass