import numpy as np


class Detector(object):
    def detect(self, image: np.ndarray) -> dict:
        raise NotImplementedError
