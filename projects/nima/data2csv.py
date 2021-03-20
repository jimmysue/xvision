import argparse
import numpy as np
import tqdm
import cv2

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpgl', '-l', type=Path, required=True,
                        help='path to jpg list file')
    parser.add_argument('--anno', '-a', type=Path,
                        help='path to AVA.txt', required=True)
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='path to output file')
    parser.add_argument('--image-dir', type=Path,
                        help='if this set, check image exist')

    args = parser.parse_args()

    with open(args.jpgl) as f:
        lines = f.readlines()
        ids = np.array(lines).astype(np.int32).tolist()

    with open(args.anno) as f:
        lines = f.readlines()
        lines = [np.array(l.strip().split()).astype(np.int32).tolist() for l in lines]

    # data to map
    kv = {}
    for line in lines:
        id = line[1]
        kv[id] = line

    f = open(args.output, 'wt')

    res = []
    for id in tqdm.tqdm(ids):
        if id in kv:
            items = kv[id]
            ratings = items[2:2+10]
            hist = np.array(ratings).astype(np.float) /  np.sum(ratings)
            hist = hist.tolist()
            histstr = ', '.join(str(v) for v in hist)
            name = f'{id}.jpg'
            line = f'{name},{histstr}\n'
            if args.image_dir:
                imgpath = args.image_dir / name
                image = cv2.imread(str(imgpath), cv2.IMREAD_COLOR)
                if image is not None:
                    f.write(line)
            else:
                f.write(line)
    f.close()
