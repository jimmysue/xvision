import torch
import os
from collections import deque
from pathlib import Path

from torch.serialization import save


class Saver:
    STEP = 'step-{:08d}.pth'
    BEST = 'best.pth'

    def __init__(self, folder, keep_num=-1) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.keep_num = keep_num
        if keep_num > 0:
            self.histories = deque()
        else:
            self.histories = None

    def save(self, step, state):
        filename = self.folder / Saver.STEP.format(step)
        torch.save(state, filename)
        if self.histories is not None:
            self.histories.append(filename)
            while(len(self.histories) > self.keep_num):
                toremove = self.histories.popleft()
                os.unlink(toremove)

    def save_best(self, state):
        filename = self.folder / Saver.BEST
        torch.save(state, filename)

    @staticmethod
    def load_from_folder(folder, step=-1, *args, **kwargs):
        folder = Path(folder)
        if (step < 0):
            # glob
            files = list(folder.glob('step-*.pth'))
            sorted(files)
            if (files):
                return torch.load(files[-1], *args, **kwargs)
            else:
                raise ValueError('no checkpoint to load')
        else:
            filename = folder / Saver.STEP.format(step)
            return torch.load(filename, *args, **kwargs)

    @staticmethod
    def load_best_from_folder(folder, *args, **kwargs):
        filename = Path(folder) / Saver.BEST
        return torch.load(filename, *args, **kwargs)


if __name__ == '__main__':

    a = [{'dict': 23}] * 100

    saver = Saver('./ckpt', keep_num=10)

    for i, d in enumerate(a):
        saver.save(i, d)
