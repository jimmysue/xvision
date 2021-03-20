import numpy as np
import cv2
from pathlib import Path
from numbers import Number
from matplotlib.pyplot import sca

import torch
from torch.utils.data import Dataset

from xvision.transforms import matrix2d
from xvision.transforms.warp import warp_affine
from xvision.transforms.boxes import bbox_affine


class WiderFace(Dataset):
    def __init__(self, label_file, image_dir, transform=None, with_points=False, min_size=0) -> None:
        super().__init__()
        if transform:
            assert callable(transform), "transform must be callable"
        self.data = list(WiderFace.parse(
            label_file, image_dir, with_points, min_size))
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
    def parse(label_file, image_dir, with_points, min_size):
        image_dir = Path(image_dir)
        if not with_points:
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
                keep = sizes > min_size
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

                keep = sizes > min_size

                bbox = bbox[keep, :]
                pts = pts[keep, ...]
                mask = mask[keep]

                if bbox.size > 0:
                    yield {
                        "path": image_dir / name,
                        "bbox": bbox,
                        "point": pts,
                        "mask": mask
                    }
                name = next_name


class BasicTransform(object):
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
        scale = max(dw / w, dh / h)
        ht, wt = h * scale, w * scale

        ph = ht - dh
        pw = wt - dw

        tx = np.random.uniform(-pw, 0)
        ty = np.random.uniform(-ph, 0)

        matrix = matrix2d.translate([tx, ty]) @ matrix2d.scale(scale)

        image = warp_affine(image, matrix, self.dsize)
        bboxes = bbox_affine(bboxes, matrix).astype(np.float32)

        ret = {
            'image': image,
            'bbox': bboxes,
            'label': np.ones(bboxes.shape[0], dtype=np.int64)
        }

        if 'point' in item:
            points = item['point'].reshape(-1, 2)
            points = points @ matrix[:2, :2].T + matrix[:2, 2]
            ret['point'] = points.reshape(
                bboxes.shape[0], -1, 2).astype(np.float32)
            ret['mask'] = item['mask']
        return ret


def wider_collate(items):
    items = {
        k: [torch.from_numpy(v[k]) for v in items] for k in items[0].keys()
    }
    items['image'] = torch.stack(items['image'])
    return items


if __name__ == '__main__':
    import torch
    import tqdm
    from xvision.utils.draw import *
    from torch.utils.data import DataLoader
    from xvision.ops.anchors import BBoxAnchors
    from projects.face_detection.config import cfg
    train = "/Users/jimmy/Documents/data/WIDER/retinaface_gt_v1.1/train/label.txt"
    dir = "/Users/jimmy/Documents/data/WIDER/WIDER_train/images"

    transform = BasicTransform((320, 320))
    data = WiderFace(train, dir, with_points=True,
                     min_size=10, transform=transform)

    loader = DataLoader(data, batch_size=128, shuffle=True,
                        num_workers=8, collate_fn=wider_collate)
    anchors = BBoxAnchors(dsize=cfg.dsize, strides=cfg.strides,
                          fsizes=cfg.fsizes, layouts=cfg.layouts)

    # for batch in tqdm.tqdm(loader):
    #     image = batch['image']
    #     point = batch['point']
    #     label = batch['label']
    #     bbox = batch['bbox']
    #     mask = batch['mask']
    #     scores, bboxes, points = anchors(label, bbox, point, mask)

    val = "/Users/jimmy/Documents/data/WIDER/retinaface_gt_v1.1/val/label.txt"
    dir = "/Users/jimmy/Documents/data/WIDER/WIDER_val/images"

    transform = BasicTransform((320, 320))
    data = WiderFace(val, dir, with_points=False,
                     min_size=10, transform=transform)

    loader = DataLoader(data, batch_size=128, shuffle=True,
                        num_workers=8, collate_fn=wider_collate)
    anchors = BBoxAnchors(dsize=cfg.dsize, strides=cfg.strides,
                          fsizes=cfg.fsizes, layouts=cfg.layouts)

    for batch in tqdm.tqdm(loader):
        image = batch['image']
        label = batch['label']
        bbox = batch['bbox']
        scores, bboxes = anchors(label, bbox)
