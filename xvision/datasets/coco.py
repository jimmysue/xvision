import tqdm
import cv2
import os
import numpy as np
from typing import Dict, Any
from torchvision.datasets.coco import CocoDetection as _Coco


class CocoDetection(_Coco):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile, transforms=transforms)
        cats = self.coco.getCatIds()
        label = list(range(1, len(cats) + 1))        # starts from 1
        self._cat_label_map = dict(zip(cats, label))
        self._label_cat_map = dict(zip(label, cats))

    def cat2label(self, cat):
        return self._cat_label_map[cat]

    def label2cat(self, label):
        return self._label_cat_map[label]

    def __getitem__(self, index: int) -> Dict[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        item = {k: [v[k] for v in target] for k in target[0].keys()}
        item = {k: np.array(v) if k !=
                'segmentation' else v for k, v in item.items()}
        item['label'] = np.array([self.cat2label(cat)
                                 for cat in item['category_id']])
        item['bbox'][:, 2:] += item['bbox'][:, :2]
        path = coco.loadImgs(img_id)[0]['file_name']
        item['image'] = cv2.imread(os.path.join(self.root, path))
        if self.transforms is not None:
            item = self.transforms(item)
        return item


if __name__ == '__main__':
    from xvision.utils.draw import draw_bbox
    data = CocoDetection(root='/Users/jimmy/Documents/data/COCO/val2017',
                         annFile='/Users/jimmy/Documents/data/COCO/annotations/instances_val2017.json')

    ids = set()
    for item in tqdm.tqdm(data):
        image = item['image']
        bbox = item['bbox']
        draw_bbox(image, bbox, thickness=2)
        cv2.imshow('v', image)
        cv2.waitKey()
    print(ids)
