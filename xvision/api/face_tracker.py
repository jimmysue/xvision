from .detector import Detector
from .face_alignmentor import FaceAlignmentor
from .face_detector import FaceDetector

class FaceTracker(Detector):
    def __init__(self, face_detector: FaceDetector, face_alignmentor: FaceAlignmentor, **kwargs) -> None:
        super().__init__()
        self.face_detector = face_detector
        self.face_alignmentor = face_alignmentor

    
    