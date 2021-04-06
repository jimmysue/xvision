from xvision.datasets.mmap import structured_dtype, create_mmap_dataset
import numpy as np
import cv2

from xvision.datasets.wflw import WFLW


def transform(item):
    image = cv2.imread(str(item['path']))
    image =cv2.resize(image, (128, 128))
    return {
        'image': image,
        'shape': item['shape']
    }

label = '/Users/jimmy/Documents/Data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
images = '/Users/jimmy/Documents/Data/WFLW/WFLW_images'
gen = WFLW.parse(label, images)
fp = create_mmap_dataset('mmap.npy', gen, transform, 8)


for v in fp:
    image = v['image']
    cv2.imshow("v", image)
    cv2.waitKey()
