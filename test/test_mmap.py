from xvision.datasets.mmap import _structured_dtype, create_mmap_dataset, MMap
import numpy as np
import cv2

from xvision.datasets.wflw import WFLW


def transform(item):
    image = cv2.imread(str(item['path']))
    image =cv2.resize(image, (128, 128))
    item.pop('path')
    return image, item

label = '/Users/jimmy/Documents/Data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
images = '/Users/jimmy/Documents/Data/WFLW/WFLW_images'
gen = WFLW.parse(label, images)
fp = MMap.create('tuple.npy', gen, transform)

fp = MMap('tuple.npy')

for (image, item ), origin in zip(fp,WFLW.parse(label, images)) :
    # image = v['image']
    item.pop('shape')
    item.pop('bbox')
    origin.pop('path')
    origin.pop('bbox')
    origin.pop('shape')
    print(item)
    print(origin)
    cv2.imshow("v", image)
    cv2.waitKey()
