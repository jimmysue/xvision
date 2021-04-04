from xvision.transforms.shapes import calc_mean_shape
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset


class JDLandmark(Dataset):
    __symmetry__ = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45,
                    44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52,
                    53, 54, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 79, 78, 77, 76, 75,
                    82, 81, 80, 79, 70, 69, 68, 67, 66, 74, 73, 72, 71, 90, 89, 88, 87, 86,
                    85, 84, 95, 94, 93, 92, 91, 100, 99, 98, 97, 96, 103, 102, 101, 105, 104]
    
    def __init__(self, landmark, picture, transform=None) -> None:
        """JingDong Landmark

        Args:
            landmark (str): folder to landmark annotations
            picture (str): folder to picutures
        """
        super().__init__()
        if transform:
            assert callable(transform), 'transform should be callable'
        self.transform = transform
        self.landmark = landmark
        self.picture = picture
        self.data = list(self.parse(landmark, picture))

    @ property
    def shapes(self):
        shapes = [v['shape'] for v in self.data]
        return np.stack(shapes, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        item = self.data[index].copy()
        item['image'] = cv2.imread(str(item['path']), cv2.IMREAD_COLOR)
        if self.transform:
            item = self.transform(item)
        return item

    @ staticmethod
    def parse(landmark, picture):
        landmark = Path(landmark)
        picture = Path(picture)

        for imgpath in picture.rglob('*.jpg'):
            annpath = (landmark / imgpath.name).with_suffix('.jpg.txt')
            with open(annpath, 'rt') as f:
                num_points = int(f.readline())
                points = f.readlines()
                points = [v.strip().split() for v in points]
                shape = np.array(points).astype(
                    np.float32).reshape(num_points, 2)

                yield {
                    'path': imgpath,
                    'shape': shape
                }


if __name__ == '__main__':
    from xvision.utils.draw import draw_shapes
    root = '/Users/jimmy/Documents/data/FA/IBUG'
    landmark = '/Users/jimmy/Documents/data/FA/JD-landmark/Train/landmark'
    picture = '/Users/jimmy/Documents/data/FA/JD-landmark/Train/picture'
    data = JDLandmark(landmark, picture)
    shapes = data.shapes

    mirrors = shapes[:, JDLandmark.__symmetry__, :]
    mirrors[:, :, 0] = -mirrors[:, :, 0]
    shapes = np.concatenate([shapes, mirrors], axis=0)
    meanshape = calc_mean_shape(shapes)
    np.set_printoptions(formatter={"float_kind": lambda x: "{:.4f}".format(x)})
    print(meanshape)

    image = np.ones((256, 256, 3), dtype=np.float32)
    draw_shapes(image, meanshape * 256)
    cv2.imshow("v", image)
    cv2.waitKey()

    # calc mean shape

    for item in data:
        image = item['image']
        shape = item['shape']
        lt = shape.min(axis=0)
        rb = shape.max(axis=0)
        size = (rb - lt).max()
        radius = int(max(2, size * 0.03))
        draw_shapes(image, shape, radius)
        cv2.imshow("v", image)
        k = cv2.waitKey()
        if k == ord('q'):
            break
