from pickle import NONE
from xvision.ops.anchors import BBoxAnchors
import torch
import torch.nn as nn
from xvision.ops import multibox
from xvision.ops.boxes import *
from xvision.ops.multibox import *


class Prior(nn.Module):
    """检测先验框生成, 匹配, 编解码等

    该对象是检测先验框的基类, 不同的先验框对象继承该类, 实现对应的行为.

    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets=None):
        """输出loss或者输出检测结果

        当提供`targets`的时候, 输出loss, 当只有predictions的时候则输出预测结果, 并且检测框做了解码.
        本函数不做非极大值抑制, 非极大值抑制交给下游处理!

        Args:
            predictions ([list, tuple]): 检测网络的输出原始输出列表, 列表中, 每个元素是每个尺度层的输出信息, 
                这些信息包括但不限于: 分类结果, 框回归结果, 关键点结果等. 且这些输出保持原始的内存排列, 即不经过
                `Tensor.permute` 操作. `Prior` 将通过这些输出获取每个尺度层的输出大小, 并对其进行必要的`permute`

            targets ([tuple], optional): 图像的标签信息, 一般情况包含: (标签, 框列表), 可能还包含人脸点和人脸点
            标志位等. 在预测时该参数省略, Prior将输出预测结果
        """
        raise NotImplementedError


class Detector(nn.Module):
    def __init__(self, prior: Prior, backbone: nn.Module, head: nn.Module = None):
        super().__init__()
        self.prior = prior
        self.backbone = backbone
        self.head = head

    def predict(self, x):
        feats = self.backbone(x)
        if self.head is not None:
            if isinstance(self.head, nn.ModuleList):
                return [head(feat) for head, feat in zip(self.head, feats)]
            elif isinstance(feats, (tuple, list)):
                return [self.head(feat) for feat in feats]
            else:
                return [self.head(feats)]
        else:
            return feats

    def forward(self, x, targets=None):
        predicts = self.predict(x)
        return self.prior(predicts, targets)


class BBoxPrior(Prior):
    def __init__(self, num_classes=1, anchor_layouts=None, iou_threshold=None, encode_mean=None, encode_std=None):
        super().__init__()
        self.num_classes = num_classes
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

        if anchor_layouts:
            self.strides, self.bases = self.generate_base_anchors(
                anchor_layouts)
        else:
            # make layouts default to None, so that we can load from state_dict
            self.strides, self.bases = [], []

        self.register_buffer('anchors', None)
        self.register_buffer('encode_mean', encode_mean)
        self.register_buffer('encode_std', encode_mean)


    @property
    def _state_keys(self):
        return ['num_classes', 'iou_threshold', 'encode_mean','encode_std', 'bases', 'strides', 'anchors']

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for k in self._state_keys:
            try:
                destination[f'{prefix}{k}'] = self.__getattr__(k)
            except :
                destination[f'{prefix}{k}'] = self.__dict__[k]
     
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for k in self._state_keys:
            self.__setattr__(k, state_dict[f'{prefix}{k}'])


    @staticmethod
    def generate_base_anchors(layouts):
        strides = []
        anchors = []
        for item in layouts:
            stride = item['stride']
            scales = torch.tensor(item['scales']).float() * (stride)
            aspects = torch.tensor(item['aspects']).float()
            # TODO: support offsets
            sizes = torch.tensor([1, 1], dtype=torch.float32)  # [2]
            sizes = scales[..., None] * sizes[None, ...]      # [s, 2]
            ratios = torch.stack(
                [torch.sqrt(aspects), 1 / torch.sqrt(aspects)], dim=-1)  # [r, 2]
            sizes = ratios[None, ...] * sizes[:, None, :]
            sizes = sizes.reshape(-1, 2)
            # add center
            centers = sizes.new_zeros(sizes.shape)
            bases = torch.cat([centers, sizes], dim=-1)
            anchors.append(bases)
            strides.append(stride)
        return strides, anchors

    @staticmethod
    def generate_layer_anchors(stride, fsize, layout, device=None):
        device = torch.device('cpu') if device is None else device
        layout = torch.tensor(layout, dtype=torch.float32,
                              device=device)  # [k, 5]

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
            layer_anchors = BBoxPrior.generate_layer_anchors(
                stride, fsize, layout, device)
            layer_anchors = layer_anchors.reshape(-1, 4)
            anchors.append(layer_anchors)
        anchors = torch.cat(anchors, dim=0)
        return anchors

    def update(self, fsizes, device):
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
        denominant[denominant <
                   self.iou_threshold] = self.iou_threshold     # [n]
        denominant = denominant[box_indice]             # [k]
        max_iou_of_anchor[max_iou_of_anchor < self.iou_threshold / 2] = 0
        scores = max_iou_of_anchor / denominant         # [k]

        labels = labels[box_indice]
        ignores = labels <= 0
        # set ignore as background score
        # TODO: support ignore scores with negative values,
        #       and sigmoid focal loss should take care of this also
        scores[ignores] = 0

        # scatter to construct confidence tensor
        # scores: [k]
        labels[ignores] = 0
        conf = scores.new_zeros(scores.size(0), self.num_classes + 1)
        # conf: [k, c] # c is the number of classes
        # index: [k, 1]
        labels = labels.unsqueeze(-1)  # [k, 1]
        scores = scores.unsqueeze(-1)  # [k, 1]
        conf.scatter_(dim=1, index=labels, src=scores)
        return conf[..., 1:], box_indice

    def match_batch(self, labels, bboxes, *others):
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

        scores = torch.stack(batch_scores, 0)   # B, k, c
        bboxes = torch.stack(batch_bboxes, 0)   # B, k, 4
        res = (scores, bboxes)
        if batch_others:
            batch_others = [torch.stack(v) for v in batch_others]
            res = (*res, *batch_others)
        return res

    def forward(self, predictions, targets=None):
        logits = []
        pred_deltas = []
        fsizes = []
        for score, bbox in predictions:
            w, h = score.size(-1), score.size(-2)
            fsizes.append((w, h))
            score = score.permute(0, 3, 1, 2).reshape(-1, self.num_classes)
            bbox = bbox.permute(0, 3, 1, 2).reshape(-1, 4)
            logits.append(score)
            pred_deltas.append(bbox)

        logits = torch.cat(logits)
        pred_deltas = torch.cat(pred_deltas)

        self.update(fsizes, logits.device)

        if targets:
            scores, bboxes = self.match_batch(*targets)
            deltas = self.encode_bboxes(bboxes)
            return score_box_loss(scores, deltas, logits, pred_deltas)
        else:
            scores = torch.sigmoid(logits)
            bboxes = self.decode_bboxes(pred_deltas)
            return scores, bboxes


class BBoxShapePrior(BBoxPrior):
    def __init__(self, num_classes=1, num_points=5, anchor_layouts=None, iou_threshold=0.35, encode_mean=None, encode_std=None):
        super().__init__(num_classes=num_classes, anchor_layouts=anchor_layouts,
                         iou_threshold=iou_threshold, encode_mean=encode_mean, encode_std=encode_std)
        self.num_points = num_points

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

    @property
    def _state_keys(self):
        return super()._state_keys + ['num_points']

    def forward(self, predictions, targets=None):
        logits = []
        pred_deltas = []
        pred_shapes = []
        fsizes = []
        for logit, delta, shape in predictions:
            w, h = logit.size(-1), logit.size(-2)
            fsizes.append((w, h))
            logit = logit.permute(0, 3, 1, 2).reshape(-1, self.num_classes)
            delta = delta.permute(0, 3, 1, 2).reshape(-1, 4)
            shape = shape.permute(0, 3, 1, 2).reshape(-1, self.num_points, 2)
            logits.append(logit)
            pred_deltas.append(delta)
            pred_shapes.append(shape)

        logits = torch.cat(logits)
        pred_deltas = torch.cat(pred_deltas)
        pred_shapes = torch.cat(pred_shapes)

        self.update(fsizes, logits.device)

        if targets:
            # calculate loss
            scores, bboxes, *others = self.match_batch(*targets)
            deltas = self.encode_bboxes(bboxes)
            if others:
                shapes, mask = others
                shapes = self.encode_points(shapes)
                return score_box_point_loss(
                    scores, deltas, shapes, logits, pred_deltas, pred_shapes, mask)
            else:
                return score_box_loss(scores, deltas, logits, pred_deltas)
        else:
            scores = torch.sigmoid(logits)
            bboxes = self.decode_bboxes(pred_deltas)
            shapes = self.decode_points(pred_shapes)
            return scores, bboxes, shapes


if __name__ == '__main__':

    from torchvision.models.mobilenet import mobilenet_v2

    backbone = mobilenet_v2()

    from projects.retinanet.config import cfg
    prior = BBoxShapePrior(1, 10, cfg.anchors, 0.35, None, None)

    detector = Detector(prior, backbone)

    state = detector.state_dict()
    detector = Detector(BBoxShapePrior(), backbone)
    detector.load_state_dict(state)
    detector.cuda()
    
    torch.save({
        'prior': prior.state_dict(),
    }, 'prior.pt')

    state = torch.load('prior.pt')

    new_prior = BBoxPrior()
    new_prior.load_state_dict(state['prior'])
    prior.load_state_dict(state['prior'])
    print(prior.anchors)
