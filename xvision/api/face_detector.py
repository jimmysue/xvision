import torch
from .detector import Detector



class _FaceDetector(Detector):
    def __init__(self, detector, device) -> None:
        super().__init__()
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceDetector(Detector):
    def __init__(self, ) -> None:
        super().__init__()