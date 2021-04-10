from projects.fa.transform import Transform
from config import cfg

import numpy as np
import cv2

from xvision.utils.draw import draw_bbox, draw_points
from transform import *
from xvision.datasets.wflw import WFLW


label = '/Users/jimmy/Documents/data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
image = '/Users/jimmy/Documents/data/WFLW/WFLW_images'

data = WFLW(label, image)

t = Transform(cfg.dsize, cfg.padding, cfg.data.meanshape, cfg.data.meanbbox)

data.transform = t

for item in data:
    image = item['image']
    shape = item['shape']
    draw_points(image, shape)
    cv2.imshow('v', image)
    cv2.waitKey()

meanbbox = np.array(cfg.data.meanbbox)
meanshape = np.array(cfg.data.meanshape)


image = np.ones((384, 384, 3))

meanshape = (meanshape - 0.5) * 224 + 192
meanbbox = (meanbbox - 0.5) * 224 + 192

draw_points(image, meanshape, color=(1, 0, 0), radius=3)
draw_bbox(image, meanbbox, (0, 0, 1), thickness=2)

cv2.line(image, (0, 0), (384, 384), color=(.5, .5, .5), thickness=1)
cv2.line(image, (384, 0), (0, 384), color=(.5, .5, .5), thickness=1)
cv2.imwrite('projects/fa/images/meanshape-meanbbox.png', (image*255).astype(np.uint8))
cv2.imshow("v", image)
cv2.waitKey()