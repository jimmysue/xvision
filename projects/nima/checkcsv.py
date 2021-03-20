import argparse
import multiprocessing as mp
import cv2
import tqdm
from pathlib import Path

def check(path, line):
    image = cv2.imread(str(path))
    if image is None:
        return False, line
    else:
        return True, line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--output', '-o', type=Path)

    args = parser.parse_args()
    paths = []
    lines = []
    with open(args.csv) as f:
        for line in f:
            lines.append(line)
            line = line.strip().split(',')
            filename = (args.image_dir / line[0]).with_suffix('.jpg')
            paths.append(filename)

    print("total images: {}".format(len(paths)))

    
    pool = mp.Pool(8)

    valids = []
    for path, line in tqdm.tqdm(zip(paths, lines)):
        valids.append(
            pool.apply_async(check, (path, line))
        )
    
    results = []
    for v in tqdm.tqdm(valids):
        ok, line = v.get()
        if ok:
            results.append(line)
        else:
            print("invalid: {}".format(line))

    with open(args.output, 'wt') as of:
        of.writelines(results)
