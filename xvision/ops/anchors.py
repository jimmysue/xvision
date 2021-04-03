import torch
import torch.nn as nn

from numbers import Number

from xvision.ops.boxes import bbox2cbox, cbox2bbox, box_iou_pairwise


def _canonical_anchor(anchor):
    x, y, r = 0, 0, 0
    if isinstance(anchor, Number):
        w, h = anchor, anchor
    elif isinstance(anchor, (list, tuple)) and len(anchor) == 1:
        w = h = anchor[0]
    elif isinstance(anchor, (list, tuple)) and len(anchor) == 2:
        w, h = anchor
    elif isinstance(anchor, (list, tuple)) and len(anchor) == 4:
        x, y, w, h = anchor
    elif isinstance(anchor, (list, tuple)) and len(anchor) == 5:
        x, y, w, h, r = anchor
    else:
        raise ValueError(f'Invalid anchor setting with value: {anchor}')
    return [x, y, w, h]


class BBoxAnchors(nn.Module):
    def __init__(self, dsize, strides, fsizes, layouts, iou_threshold=0.3, encode_mean=None, encode_std=None):
        super().__init__()

        self.dsize = dsize
        self.strides = strides
        self.fsizes = fsizes
        self.layouts = layouts
        self.iou_threshold = iou_threshold

        if (encode_mean):
            encode_mean = torch.tensor(
                encode_mean, dtype=torch.float32).reshape(-1)
            assert encode_mean.numel() == 4, "encode_mean should be of 4-element"
        else:
            encode_mean = torch.zeros(4, dtype=torch.float32)

        if (encode_std):
            encode_std = torch.tensor(
                encode_std, dtype=torch.float32).reshape(-1)
            assert encode_std.numel() == 4, "encode_std should be of 4-element"
        else:
            encode_std = torch.ones(4, dtype=torch.float32)
        self.register_buffer('encode_mean', encode_mean)
        self.register_buffer('encode_std', encode_std)
        anchors = self.generate_anchors(strides, fsizes, layouts)
        self.register_buffer('anchors', anchors)

    @staticmethod
    def generate_layer_anchors(stride, fsize, layout, device=None):
        device = torch.device('cpu') if device is None else device
        layout = [_canonical_anchor(v) for v in layout]
        layout = torch.tensor(
            layout, dtype=torch.float32, device=device)  # [k, 5]

        # generate offset grid
        fw, fh = fsize
        vx = torch.arange(0.5, fw, dtype=torch.float32, device=device) * stride
        vy = torch.arange(0.5, fh, dtype=torch.float32, device=device) * stride
        vy, vx = torch.meshgrid(vy, vx)
        offsets = torch.stack([vx, vy], dim=-1)  # [fh, fw, 2]

        anchors = layout.repeat(fh, fw, 1, 1)  # [fh, fw, k, 5]
        anchors[:, :, :, :2] += offsets[:, :, None, :]  # [fh, fw, k, 5]
        return anchors

    @staticmethod
    def generate_anchors(strides, fsizes, layouts, device=None):
        anchors = []
        for stride, fsize, layout in zip(strides, fsizes, layouts):
            layer_anchors = BBoxAnchors.generate_layer_anchors(
                stride, fsize, layout, device)
            layer_anchors = layer_anchors.reshape(-1, 4)
            anchors.append(layer_anchors)
        anchors = torch.cat(anchors, dim=0)
        return anchors

    def update(self, fsizes):
        device = self.anchors.device
        self.anchors = self.generate_anchors(
            self.strides, fsizes, self.layouts, device)

    def encode_bboxes(self, bboxes):
        # bboxes: [*, k, 4]
        # self.anchors: [k, 4]
        cboxes = bbox2cbox(bboxes)
        centers = (cboxes[..., :2] - self.anchors[..., :2]) / \
            self.anchors[..., 2:]  # [*, k, 2]
        sizes = cboxes[..., 2:] / self.anchors[..., 2:]
        sizes = torch.log(sizes)    # [*, k, 2]
        deltas = torch.cat([centers, sizes], dim=-1)
        deltas = (deltas - self.encode_mean) / self.encode_std
        return deltas

    def decode_bboxes(self, deltas):
        deltas = (deltas * self.encode_std) + self.encode_mean
        sizes = torch.exp(deltas[..., 2:]) * self.anchors[..., 2:]
        centers = deltas[..., :2] * \
            self.anchors[..., 2:] + self.anchors[..., :2]
        cboxes = torch.cat([centers, sizes], dim=-1)
        return cbox2bbox(cboxes)

    def encode_points(self, points):
        # points: [*, k, p, 2]
        deltas = (points - self.anchors[..., None, :2]
                  ) / self.anchors[..., None, 2:]
        deltas = (deltas - self.encode_mean[:2]) / self.encode_std[:2]
        return deltas

    def decode_points(self, deltas):
        # deltas: [*, k, p, 2]
        deltas = (deltas * self.encode_std[:2]) + self.encode_mean[:2]
        points = deltas * self.anchors[...,
                                       None, 2:] + self.anchors[..., None, :2]
        return points

    def match(self, labels, bboxes):
        # labels: [n]
        # bboxes: [n, 4]
        # points: [n, p, 2] (p is number of points)
        # iou_threshold: threshold to determine positive anchor
        banchors = cbox2bbox(self.anchors)
        # [k, n] where k is number of anchors
        iou = box_iou_pairwise(banchors, bboxes)
        # find max iou of anchor to determine gt index
        max_iou_of_anchor, box_indice = iou.max(dim=1)  # [k]
        max_iou_of_bbox, anchor_indice = iou.max(dim=0)  # [n]

        # make sure each target assigend an anchor
        for target_index, anchor_index in enumerate(anchor_indice):
            max_iou_of_anchor[anchor_index] = max_iou_of_bbox[target_index]
            box_indice[anchor_index] = target_index
        # find max iou of each box to determine denominant

        denominant = max_iou_of_bbox                    # [n]
        denominant[denominant < self.iou_threshold] = self.iou_threshold     # [n]
        denominant = denominant[box_indice]             # [k]
        max_iou_of_anchor[max_iou_of_anchor < self.iou_threshold / 2] = 0
        scores = max_iou_of_anchor / denominant         # [k]

        labels = labels[box_indice]
        ignores = labels <= 0
        # set ignore as background score
        scores[ignores] = 0
        return scores, box_indice

    def forward(self, labels, bboxes, *others):  # we don't encode
        # labels: [B, n]
        # bboxes: [B, n, 4]
        # points: [B, n, p, 2]
        batch_scores = []
        batch_bboxes = []
        batch_others = [[] for _ in others]
        for label, bbox, *other in zip(labels, bboxes, *others):
            score, indice = self.match(label, bbox)
            batch_scores.append(score)
            batch_bboxes.append(bbox[indice])
            for i, v in enumerate(other):
                batch_others[i].append(v[indice])

        scores = torch.stack(batch_scores, 0)   # B, k
        bboxes = torch.stack(batch_bboxes, 0)   # B, k, 4
        res = (scores, bboxes)
        if batch_others:
            batch_others = [torch.stack(v) for v in batch_others]
            res = (*res, *batch_others)
        return res

    def split(self, scores):
        # [B, k, +]
        # return [ [B, h1, w1, +], [B, h2, w2, +], [B, h3, w3, +]] where h*, w* is feature size of each level anchor
        # h, w, k, 1
        last = 0
        nums = []
        sizes = []
        for i, (w, h) in enumerate(self.fsizes):
            k = len(self.layouts[i])
            nums.append(
                (last, last + w * h * k,  k)
            )
            last = last + w * h * k
            sizes.append((w, h))

        res = []
        for (s, l, k), (w, h) in zip(nums, sizes):
            r = scores[:, s:l, ...]
            shape = [r.shape[0], h, w] + [k] + list(r.shape[3:])
            r = r.reshape(shape)
            for i in range(k):
                res.append(r[:, :, :, i, ...])

        return res
