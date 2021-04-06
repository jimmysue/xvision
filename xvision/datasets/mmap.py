import numpy as np
from numbers import Number
from types import GeneratorType
from torch.utils.data import Dataset


def structured_dtype(value):
    if isinstance(value, tuple):
        # f0, f1...
        return np.dtype([('', structured_dtype(v)) for v in value])
    elif isinstance(value, dict):
        return np.dtype([(k, structured_dtype(v)) for k, v in value.items()])
    elif isinstance(value, np.ndarray):
        return np.dtype((value.dtype, value.shape))
    elif isinstance(value, Number):
        return np.dtype(type(value))
    else:
        raise ValueError(f'Unsupport structured type: {value}')


def structured_assign(dst, value):
    if isinstance(value, tuple):
        for i, v in enumerate(value):
            structured_assign(dst[f'f{i}'], v)
    elif isinstance(value, dict):
        for k, v in value.items():
            structured_assign(dst[k], v)
    else:
        dst = value


def create_mmap_dataset(filename, data, transform):
    if isinstance(data, GeneratorType):
        data = list(data)
    item = data[0]
    data_type = structured_dtype(item)
    shape = (len(data),)
    fp = np.memmap(filename, data_type, 'w+', shape=shape)
    for i, v in enumerate(data):
        v = transform(v)
        structured_assign(fp[i], v)
    

class MMap(Dataset):
    def __init__(self, filename, transform=None):
        super().__init__()
        if transform:
            assert callable(transform), 'transform should be callable'
        self.filename = filename
        self.transform = transform
        self.mmap = np.lib.format.open_memmap(filename, mode='r')


if __name__ == '__main__':
    fp = np.memmap('filename.npy', dtype='float32', mode='w+', shape=(3, 4))
