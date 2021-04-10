import torch
import torch.nn as nn

from typing import List, Tuple


class DetectNet(nn.Module):
    def __init__(self, backbone, head, num_classes, anchors, iou_threshold=0.35, score_threshold=0.5, encode_mea=None, encode_std=None):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.head = head
        self.anchors = anchors
        self.prior

    def forward_train(self, images, targets):
        pass

    def inference(self, images):
        features = self.backbone(images)  # list of tensor or tensor
        sizes = []
        if isinstance(features, (list, tuple)):
            confs = []
            boxes = []
            if isinstance(self.head, nn.ModuleList):
                heads = self.head
            else:
                heads = [self.head] * len(features)
            for head, feat in zip(features, heads):
                w, h = feat.size(-1), feat.size(-2)
                sizes.append((w, h))
                conf, box = head(feat)
                # conf: [B, KA, H, W]  where K is num_class, A is number of anchors
                # box:  [B, 4A, H, W]
                conf = conf.permute(0, 2, 3, 1).reshape(-1,
                                                        self.num_classes)  # [BHWA, K]
                box = box.permute(0, 2, 3, 1).reshape(-1, 4)
                confs.append(conf)
                boxes.append(box)
                return torch.cat(confs), torch.cat(boxes)
        else:
            sizes.append((features.size(-1), features.size(-2)))
            conf, box = self.head(features)
            conf = conf.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            box = box.permute(0, 2, 3, 1).reshape(-1, 4)
            return conf, box

    def forward_eval(self, images):
        pass

    def forward(self, images, targets: Tuple[List, List] = None):
        confs, boxes = self.inference(images)
        if targets is not None:
            return self.forward_train(images, targets)
        else:
            return self.forward_eval(images)
