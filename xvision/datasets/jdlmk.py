import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset


class JDLandmark(Dataset):
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
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> dict:
        item = self.data[index].copy()
        item['image'] = cv2.imread(str(item['path']), cv2.IMREAD_COLOR)
        if self.transform:
            item = self.transform(item)
        return item

    @staticmethod
    def parse(landmark, picture):
        landmark = Path(landmark)
        picture= Path(picture)

        for imgpath in picture.rglob('*.jpg'):
            annpath = (landmark / imgpath.name).with_suffix('.jpg.txt')
            with open(annpath, 'rt') as f:
                num_points = int(f.readline())
                points = f.readlines()
                points = [v.strip().split() for v in points]
                shape = np.array(points).astype(np.float32).reshape(num_points, 2)
                
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

    for item in data:
        image = item['image']
        shape = item['shape']
        lt = shape.min(axis=0)
        rb = shape.max(axis=0)
        size = (rb - lt).max()
        radius =int( max(2, size * 0.03))
        draw_shapes(image, shape, radius)
        cv2.imshow("v", image)
        k = cv2.waitKey()
        if k == ord('q'):
            break
