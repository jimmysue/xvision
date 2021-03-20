import torch
from torch.utils.data import DataLoader
from xvision.utils.draw import *
from xvision.ops.anchors import BBoxAnchors
from xvision.data.wider import WiderFace, BasicTransform
from config import cfg

if __name__ == '__main__':

    val = "/Users/jimmy/Documents/data/WIDER/retinaface_gt_v1.1/train/label.txt"
    dir = "/Users/jimmy/Documents/data/WIDER/WIDER_train/images"

    transform = BasicTransform((320, 320))

    data = WiderFace(val, dir, with_points=True,
                     min_size=10, transform=None)

    anchors = BBoxAnchors(dsize=cfg.dsize, strides=cfg.strides, fsizes=cfg.fsizes, layouts=cfg.layouts)

    for v in data:
        for i in range(1):
            item = transform(v)
            image = item['image']
            bbox = item['bbox']
            point = item['point']
            label = item['label']
            masks = item['mask']
            h, w = image.shape[:2]
            point = [torch.from_numpy(point)]
            labels = [torch.from_numpy(label)]
            bbox = [torch.from_numpy(bbox)]
            masks= [torch.from_numpy(masks)]
            scores, bboxes, points, masks = anchors(labels, bbox, point, masks)
            bboxes = anchors.encode_bboxes(bboxes)
            points = anchors.encode_points(points)
            bboxes = anchors.decode_bboxes(bboxes)
            points = anchors.decode_points(points)

            splits = anchors.split(scores)

            for i, score in enumerate(splits):
                score = score.numpy().squeeze(0)
                score = cv2.resize(score, (w, h))
                name = f'score={i}'
                cv2.imshow(name, score)

            bbox = bboxes.numpy()
            point = points.numpy()
            draw_bbox(image, bbox, thickness=2)
            draw_points(image, point, radius=5)
            cv2.imshow("v", image)
            cv2.waitKey()
            pass
