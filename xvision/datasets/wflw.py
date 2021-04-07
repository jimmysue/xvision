import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class WFLW(Dataset):
    """WFLW dataset
    """
    __symmetry__ = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                    17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                    0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41,
                    40, 39, 38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69,
                    68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65, 82, 81, 80, 79,
                    78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93,
                    97, 96]

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
    
    @property
    def meanshape(self):
        shapes = data.shapes
        mirrors = shapes[:, self.__symmetry__, :]
        mirrors[:, :, 0] = -mirrors[:, :, 0]
        shapes = np.concatenate([shapes, mirrors], axis=0)
        return calc_mean_shape(shapes)

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
                bbox = np.array(
                    items[196:196+4]).astype(np.float32).reshape(-1)
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
    from xvision.transforms.shapes import calc_mean_shape
    label = '/Users/jimmy/Documents/Data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
    image = '/Users/jimmy/Documents/Data/WFLW/WFLW_images'
    data = WFLW(label, image)

    shapes = data.shapes
    mirrors = shapes[:, WFLW.__symmetry__, :]
    mirrors[:, :, 0] = -mirrors[:, :, 0]
    shapes = np.concatenate([shapes, mirrors], axis=0)
    meanshape = calc_mean_shape(shapes)

    np.set_printoptions(formatter={"float_kind": lambda x: "{:.4f}".format(x)})
    print(meanshape)

    image = np.ones((1080, 1080, 3), dtype=np.float32)
    draw_points(image, (meanshape - 0.5) * 900 + 540, radius = 4, plot_index=True)
    cv2.imshow("v", image)
    cv2.waitKey()

    # calc mean shape



    # calc meanshape

    print(shapes.shape)
    for item in data:
        image = item.pop('image')
        shape = item.pop('shape')
        draw_points(image, shape, radius=2, plot_index=True)
        cv2.imshow('v', image)
        k = cv2.waitKey()
        if k == ord('q'):
            break
