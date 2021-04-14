"""
file - dataset.py
Customized dataset class to loop through the AVA dataset and apply needed image augmentations for training.

Copyright (C) Yunxiao Shi 2017 - 2020
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os
import numpy as np
import cv2
import tqdm
import shutil

from pathlib import Path
import torch
from torch.utils import data
import torchvision.transforms as transforms


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, image_dir, transform=None, cache_dir=None):
        # 更改了原始实现, 现在的实现csv文件没有序号列
        self.data = self.parse(csv_file, image_dir)
        self.transform = transform
        self.image_dir = image_dir
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if not self.cache_dir:
            image = cv2.imread(str(item['path']), cv2.IMREAD_COLOR)
        else:
            fullpath = self.cache_dir / item['path']
            image = cv2.imread(str(fullpath), cv2.IMREAD_COLOR)
            if not image:
                src = Path(self.image_dir) / item['path']
                dst = Path(self.cache_dir) / item['path']
                shutil.copy(src, dst)
                image = cv2.imread(str(dst), cv2.IMREAD_COLOR)
        item['image'] = image
        item.pop('path')
        if self.transform:
            item['image'] = self.transform(item['image'])
        return item

    @staticmethod
    def parse(csv_file, image_dir):
        data = []
        with open(csv_file) as f:
            for line in tqdm.tqdm(f):
                items = line.strip().split(',')
                name = items[0]
                hist = np.array(items[1:]).astype(np.float32)
                path = Path(name).with_suffix('.jpg')
                data.append(
                    {
                        'path': path,
                        'annotations': hist
                    }
                )

        return data


if __name__ == '__main__':
    from opencv_transforms import transforms
    # sanity check
    root = 'data/images'
    csv_file = 'data/trainset.csv'
    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dset = AVADataset(csv_file=csv_file, image_dir=root,
                      transform=train_transform)

    train_loader = data.DataLoader(
        dset, batch_size=4, shuffle=True, num_workers=4)

    for i, data in enumerate(train_loader):
        images = data['image']
        print(images.size())
        labels = data['annotations']
        print(labels.size())
