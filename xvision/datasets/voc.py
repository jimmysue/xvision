import cv2
import xml.etree.ElementTree as ET
from torchvision.datasets import VOCDetection as _Voc
from typing import Optional, Callable, Tuple, Any


class VOCDetection(_Voc):
    """VOC detection dataset that using cv2 backend
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target



if __name__ == '__main__':
    data = VOCDetection('/Users/jimmy/Documents/data/', download=False)

    for img, ann in data:
        cv2.imshow("img", img)
        cv2.waitKey()
