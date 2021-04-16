from numpy.linalg.linalg import det
import torch
import torch.nn as nn

from torchvision.ops.boxes import nms

class Predictor:
    def __init__(self, detector, image_mean=[0,0,0], image_std=[1, 1, 1], score_threshold=0.1, iou_threshold=0.5, device=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')
        else:
            device = torch.device(device)
        
        self.device = device
        self.model = detector.to(device)
        self.model.eval()
        self.mean = image_mean
        self.std = image_std

    @torch.no_grad()
    def predict(self, image):
        tensor = torch.from_numpy(image).to(self.device).float()
        tensor = (tensor - tensor.new_tensor(self.mean)) / tensor.new_tensor(self.std)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        scores, boxes, points = self.model(tensor)
        scores = scores[0].squeeze(-1).detach()   # [k]
        boxes = boxes[0].detach()     # [k, 4]
        points = points[0].detach()   # [k, p, 2]

        fg = scores > self.score_threshold

        scores = scores[fg]
        boxes = boxes[fg]
        points = points[fg]
        keep = nms(boxes, scores, self.iou_threshold)

        scores = scores[keep]
        boxes = boxes[keep]
        points = points[keep]

        # normalize coordidates
        return scores.cpu().numpy(), boxes.cpu().numpy(), points.cpu().numpy()