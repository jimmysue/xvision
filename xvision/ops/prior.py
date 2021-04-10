import torch.nn  as nn

class PriorBase(nn.Module):
    def __init__(self):
        super().__init__()

    def update(self, sizes):
        raise NotImplementedError
    
    def encode_bbox(self, bboxes):
        raise NotImplementedError
    
    def decode_bbox(self, bboxes):
        raise NotImplementedError
    
    def forward(self, label, boxes):
        raise NotImplementedError

class BBoxPrior(nn.Module):
    def __init__(self, num_classes, anchor_layouts, iou_threshold, score_threshold)