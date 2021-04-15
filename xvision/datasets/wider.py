import numpy as np
import cv2
import random
from pathlib import Path
from numbers import Number

import torch
from torch.utils.data import Dataset

from xvision.transforms import matrix2d
from xvision.transforms.warp import warp_affine
from xvision.transforms.boxes import bbox_affine


class WiderFace(Dataset):
    def __init__(self, label_file, image_dir, transform=None, with_shapes=False, min_face=0) -> None:
        super().__init__()
        if transform:
            assert callable(transform), "transform must be callable"
        self.data = list(WiderFace.parse(
            label_file, image_dir, with_shapes, min_face))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        item = self.data[index].copy()
        item["image"] = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
        if self.transform:
            item = self.transform(item)
        return item

    @staticmethod
    def parse(label_file, image_dir, with_shapes, min_face):
        image_dir = Path(image_dir)
        if not with_shapes:
            def parse_annotation(fd):
                name, rects = None, []
                for line in fd:
                    if line.startswith('#'):
                        name = line[1::].strip()
                        break
                    else:
                        items = line.strip().split()
                        items = [float(v) for v in items]
                        rects.append(items[:4])
                return name, rects
            fd = open(label_file)
            name, _ = parse_annotation(fd)

            while name is not None:
                next_name, rects = parse_annotation(fd)
                bbox = np.array(rects, np.float32)  # n x 4
                sizes = np.min(bbox[:, 2:], axis=-1)
                bbox[:, 2:] += bbox[:, :2]
                fullpath = image_dir / name
                keep = sizes > min_face
                bbox = bbox[keep, :]
                if bbox.size > 0:
                    yield {
                        "path": fullpath,
                        "bbox": bbox
                    }
                name = next_name
        else:

            def parse_bbox_lmk(fd):
                name, rects, pts = None, [], []
                for line in fd:
                    if (line.startswith('#')):
                        name = line[1:].strip()
                        break
                    else:
                        items = line.strip().split()
                        items = [float(v) for v in items]
                        rect = items[:4]
                        lmk = items[4:-1]
                        rects.append(rect)
                        pts.append(lmk)
                return name, rects, pts
            fd = open(label_file)
            name, _, _ = parse_bbox_lmk(fd)
            while name is not None:
                next_name, rects, pts = parse_bbox_lmk(fd)
                bbox = np.array(rects, np.float32)
                sizes = np.min(bbox[:, 2:], axis=-1)
                bbox[:, 2:] += bbox[:, :2]

                pts_score = np.array(pts).reshape(bbox.shape[0], -1, 3)
                scores = pts_score[:, :, 2]
                pts = pts_score[:, :, :2]
                valids = scores >= 0
                mask = np.all(valids, axis=-1)

                keep = sizes > min_face

                bbox = bbox[keep, :]
                pts = pts[keep, ...]
                mask = mask[keep]

                if bbox.size > 0:
                    yield {
                        "path": image_dir / name,
                        "bbox": bbox,
                        "shape": pts,
                        "mask": mask
                    }
                name = next_name


class ValTransform(object):
    def __init__(self, dsize) -> None:
        super().__init__()
        if isinstance(dsize, Number):
            self.dsize = (int(dsize), int(dsize))
        elif len(dsize) == 2:
            w, h = dsize
            self.dsize = (int(w), int(h))
        else:
            raise ValueError(f'Invalid dsize: {dsize}')

    def __call__(self, item):
        image = item['image']
        bboxes = item['bbox'].reshape(-1, 4)
        # calc transform matrix
        h, w = image.shape[:2]
        dw, dh = self.dsize
        scale = min(dw / w, dh / h)
        matrix = matrix2d.scale(scale)
        image = warp_affine(image, matrix, self.dsize)
        bboxes = bbox_affine(bboxes, matrix).astype(np.float32)

        ret = {
            'image': image,
            'bbox': bboxes,
            'label': np.ones(bboxes.shape[0], dtype=np.int64)
        }

        if 'shape' in item:
            shapes = item['shape'].reshape(-1, 2)
            shapes = shapes @ matrix[:2, :2].T + matrix[:2, 2]
            ret['shape'] = shapes.reshape(
                bboxes.shape[0], -1, 2).astype(np.float32)
            ret['mask'] = item['mask']
        return ret


class TrainTransform(ValTransform):
    # random
    def __init__(self, dsize, **augments) -> None:
        super().__init__(dsize)
        # random interpolate choid from below
        self.inters = augments.get("inters", [cv2.INTER_LINEAR])
        self.rotation = augments['rotation']  # random rotate
        self.min_face, self.max_face = augments['min_face'], augments['max_face']
        self.symmetry = augments['symmetry']

    def _transform(self, item, matrix, inter, mirror):
        image = item['image']
        bboxes = item['bbox']
        image = warp_affine(image, matrix, self.dsize,
                            flags=inter, borderMode=cv2.BORDER_REPLICATE)
        bboxes = bbox_affine(bboxes, matrix).astype(np.float32)

        ret = {
            'image': image,
            'bbox': bboxes,
            'label': np.ones(bboxes.shape[0], dtype=np.int64)
        }

        if 'shape' in item:
            points = item['shape'].reshape(-1, 2)
            points = points @ matrix[:2, :2].T + matrix[:2, 2]
            points = points.reshape(bboxes.shape[0], -1, 2).astype(np.float32)
            if mirror:
                points = points[:, self.symmetry, :]
            ret['shape'] = points
            ret['mask'] = item['mask']
        return ret

    def _augment(self, item, dsize):
        image = item['image']
        boxes = item['bbox']  # [n, 4]
        h, w = image.shape[:2]
        dw, dh = dsize
        # random choose one face as interest region,
        box = random.choice(boxes)
        size = np.random.choice(np.sqrt(np.prod(boxes[:, 2:] - boxes[:, :2], axis=-1)))
        max_size = min(size, self.max_face)
        max_scale = max_size / size
        min_scale = min(dw / w, dh / h)
        scale = random.uniform(min_scale, max_scale)
        ibox = np.array([0, 0, w, h], dtype=np.float32)
        matrix = matrix2d.scale(scale)
        tbox = bbox_affine(ibox, matrix)
        dbox = np.array([0, 0, dw, dh], dtype=np.float32)
        fbox = bbox_affine(box, matrix)

        # determine translation bound
        tlow = tbox[:2] - dbox[:2]
        thigh = tbox[2:] - dbox[2:]
        flow = fbox[2:] - dbox[2:]
        fhigh = fbox[:2] - dbox[:2]

        thigh = np.minimum(thigh, fhigh)
        tlow = np.maximum(tlow, flow)

        high = np.maximum(tlow, thigh)
        low = np.minimum(tlow, thigh)
        txty = np.random.uniform(low, high)
        translate = matrix2d.translate(-txty)
        matrix = translate @ matrix
        angle = np.random.uniform(-self.rotation, self.rotation)
        rotate = matrix2d.center_rotate_scale_cw([dw/2, dh/2], angle, 1.0)
        matrix = rotate @ matrix

        mirror = random.choice([True, False])
        if mirror:
            matrix = matrix2d.hflip(dw) @ matrix

        item = self._transform(
            item, matrix, random.choice(self.inters), mirror)
        # remove face exceed image
        boxes = item['bbox'].copy()
        area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=-1)
        boxes[boxes < 0] = 0
        boxes[:, 2:] = np.minimum(boxes[:, 2:], [[dw, dh]])
        inner = np.prod(boxes[:, 2:] - boxes[:, :2], axis=-1)
        ratio = inner / area
        discard = ratio < 0.4
        item['label'][discard] = 0
        return item

    def __call__(self, item):
        item = self._augment(item, self.dsize)
        return item


def wider_collate(items):
    items = {
        k: [torch.from_numpy(v[k]) for v in items] for k in items[0].keys()
    }
    items['image'] = torch.stack(items['image'])
    return items


if __name__ == '__main__':
    import torch
    import tqdm
    from collections import defaultdict
    from matplotlib import pyplot as plt
    from xvision.utils.draw import *
    from projects.fd.config import cfg
    train = "/Users/jimmy/Documents/data/WIDER/retinaface_gt_v1.1/train/label.txt"
    dir = "/Users/jimmy/Documents/data/WIDER/WIDER_train/images"

    transform = TrainTransform((320, 320), **cfg.augments)
    data = WiderFace(train, dir, with_shapes=True,
                     min_face=1, transform=transform)

    areas = defaultdict(float)
    count = 0
    for item in tqdm.tqdm(data):
        image = item['image']
        boxes = item['bbox']
        area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=-1)
        sizes = np.sqrt(area).astype(np.int32)
        valid = item['label'] > 0
        area = area[valid]
        sizes = sizes[valid]
        for s, a in zip(sizes, area):
            areas[s] += a
    
    plt.bar(areas.keys(), areas.values())
    plt.show()
  
