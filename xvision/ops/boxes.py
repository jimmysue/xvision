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
    # bboxes1 [m, 4]
    # bboxes2 [n, 4]

    # Distance-IoU = IoU - pho^2 / c^2
    # where pho is the distance of boxex center
    # and c is diagonal length of enclosing box

    # calc enclosing box
    lt = torch.min(bboxes1[:, None, :2], bboxes2[None, :, :2])  # [m, n, 2]
    rb = torch.max(bboxes1[:, None, 2:], bboxes2[None, :, 2:])  # [m, n, 2]

    c2 = torch.square(lt - rb).sum(-1)

    ctr1 = (bboxes1[:, 2:] - bboxes1[:, :2]) / 2   # [m, 2]
    ctr2 = (bboxes2[:, 2:] - bboxes2[:, :2]) / 2   # [n, 2]

    pho2 = torch.square(ctr1[:, None, :] - ctr2[None, :, :]).sum(-1)
    diou = box_iou_pairwise - pho2 / c2
    return diou


def box_diou_itemwise(bboxes1, bboxes2):
    raise NotImplementedError
    pass


if __name__ == '__main__':

    box1 = torch.tensor([0, 0, 1, 1]).reshape(-1, 4)
    box2 = torch.tensor([0, 0, 1, 1]).reshape(-1, 4)

    diou = box_diou_pairwise(box1, box2)
    iou = box_iou_pairwise(box1, box2)
