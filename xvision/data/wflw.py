import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class WFLW(Dataset):
    """WFLW dataset
    """

    def __init__(self, label_file, image_dir, transform=None) -> None:
        super().__init__()
        if transform:
            assert callable(transform), 'transform should be callable'
        self.data = list(self.parse(label_file, image_dir))
        self.transform = transform

    @property
    def shapes(self):
        shapes = [v['shape'] for v in self.data]
        return np.stack(shapes, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index].copy()
        item['image'] = cv2.imread(str(item['path']), cv2.IMREAD_COLOR)
        if self.transform:
            item = self.transform(item)
        return item

    @staticmethod
    def parse(label_file, image_dir):
        image_dir = Path(image_dir)
        with open(label_file) as f:
            for line in f:
                items = line.strip().split()
                shape = np.array(items[:196]).astype(np.float32).reshape(-1, 2)
                bbox = np.array(items[196:196+4]).astype(np.float32).reshape(-1)
                attr = [int(v) for v in items[200:200+6]]
                keys = ['pose', 'expression', 'illumination',
                        'makeup', 'occlusion', 'blur']
                item = dict(zip(keys, attr))
                item['shape'] = shape
                item['bbox'] = bbox
                item['path'] = image_dir / items[-1]
                yield item


if __name__ == '__main__':
    from xvision.utils.draw import draw_points

    label = '/Users/jimmy/Documents/Data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
    image = '/Users/jimmy/Documents/Data/WFLW/WFLW_images'
    data = WFLW(label, image)
    shapes = data.shapes
    print(shapes.shape)
    for item in data:
        image = item.pop('image')
        shape = item.pop('shape')
        draw_points(image, shape, radius=5)
        print(item)
        cv2.imshow('v', image)
        k = cv2.waitKey()
        if k == ord('q'):
            break
