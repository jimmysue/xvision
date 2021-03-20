from types import *
from torch.utils.data import Dataset, DataLoader


def repeat_loader(loader):
    while True:
        for batch in loader:
            yield batch