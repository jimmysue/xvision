#! /usr/bin/env python
import torch
import cv2
import numpy as np
import tqdm
import logging
from copy import copy
from pathlib import Path
from xvision.utils import Saver
from xvision.models import fd as models
from xvision.ops.anchors import BBoxAnchors
from xvision.transforms.warp import warp_affine
from xvision.transforms import matrix2d
from xvision.models.fd.predictor import Predictor
from xvision.transforms.boxes import bbox2rect, bbox_affine
from xvision.utils.draw import draw_bbox, draw_points
from xvision.models.detection import Detector, BBoxShapePrior

def main(args):
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.__dict__[args.model.name]()
    prior = BBoxShapePrior(args.num_classes, 5, args.anchors, args.iou_threshold, args.encode_mean, args.encode_std)
    
    detector = Detector(prior, model)

    state = Saver.load_best_from_folder(workdir, map_location='cpu')

    # TODO: fix model saving when trained with nn.DataParallel
    #       remove below lines, when train.py save model properly
    model_state = {}
    for k, v in state['model'].items():
        if k.startswith('backbone.module'):
            k = k.replace('backbone.module', 'backbone')
        model_state[k] = v
    detector.load_state_dict(model_state, strict=True)
    predictor = Predictor(detector, args.image_mean, args.image_std, args.test.score_threshold, args.test.iou_threshold, device)
    output_dir = workdir / 'result'
    image_dir = Path(args.test.image_dir)
    for imagefile in tqdm.tqdm(image_dir.rglob('*.jpg')):
        relpath = imagefile.relative_to(image_dir)
        outpath = output_dir / relpath
        outpath.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(imagefile), cv2.IMREAD_COLOR)
        if args.test.long_size > 0:
            scale = (args.test.long_size / np.array(image.shape[:2])).min()
            w, h = (np.array(image.shape[:2]) * scale).astype(np.int32)[::-1]

            matrix = matrix2d.scale(scale)
            warped = warp_affine(image, matrix, (w, h))
            score, boxes, points = predictor.predict(warped)
            n = score.size
            if n > 0:
                matrix = np.linalg.inv(matrix)
                boxes = bbox_affine(boxes, matrix)
                # k, p, 2
                points = points.reshape(-1, 2) @ matrix[:2, :2].T + matrix[:2, 2]
                points = points.reshape(score.size, -1, 2)
        else:
            score, boxes, points = predictor.predict(image)
        n = score.size
        if n > 0:
            draw_bbox(image, boxes)
            draw_points(image, points)
        cv2.imwrite(str(outpath), image)

        # write output txt
        resfile = outpath.with_suffix('.txt')
        with open(resfile, 'wt') as f:
            f.write('{}\n'.format(imagefile.name))
            f.write(f'{n}\n')
            if n > 0:
                boxes = bbox2rect(boxes)
                for score, box in zip(score.tolist(), boxes):
                    box = box.reshape(-1).tolist()
                    line = ' '.join(str(v) for v in box)
                    f.write(f'{line} {score}\n')


if __name__ == '__main__':
    from config import cfg

    cfg.parse_args()
    cfg.freeze()
    main(cfg)
