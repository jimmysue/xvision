import numpy as np
import tqdm
from numbers import Number
from types import GeneratorType
from torch.utils.data import Dataset
from joblib import Parallel, delayed


def _structured_dtype(value):
    if isinstance(value, tuple):
        # f0, f1...
        return np.dtype([('', _structured_dtype(v)) for v in value])
    elif isinstance(value, dict):
        return np.dtype([(k, _structured_dtype(v)) for k, v in value.items()])
    elif isinstance(value, np.ndarray):
        return np.dtype((value.dtype, value.shape))
    elif isinstance(value, Number):
        return np.dtype(type(value))
    else:
        raise ValueError(f'Unsupport structured type with value: {value}')


def _structured_assign(dst, key, value):
    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            _structured_assign(dst[key], f'f{i}', v)
    elif isinstance(value, dict):
        for k, v in value.items():
            _structured_assign(dst[key], k, v)
    elif isinstance(value, np.ndarray):
        dst[key][...] = value
    else:
        dst[key] = value


def _is_tuple_fileds(fileds):
    for i, key in enumerate(fileds.keys()):
        if key != f'f{i}':
            return False
    return True


def _structured_unfold(value):
    dtype = value.dtype
    if dtype.fields:
        if _is_tuple_fileds(dtype.fields):
            return tuple([_structured_unfold(value[k]) for k in dtype.fields.keys()])
        else:
            return {k: _structured_unfold(value[k]) for k in dtype.fields.keys()}
    else:
        return value


def create_mmap_dataset(filename, data, transform, num_workers=1):
    if isinstance(data, GeneratorType):
        data = list(data)
    item = transform(data[0].copy())
    data_type = _structured_dtype(item)
    shape = (len(data),)
    fp = np.lib.format.open_memmap(
        filename, mode='w+', dtype=data_type, shape=shape)

    def process(i, v):
        v = transform(v)
        _structured_assign(fp, i, v)

    if num_workers > 1:
        Parallel(n_jobs=num_workers)(delayed(process)(i, v)
                                     for i, v in enumerate(tqdm.tqdm(data, desc=f'memmaping to {filename}')))
    else:
        for i, v in enumerate(tqdm.tqdm(data, desc=f'memmaping to {filename}')):
            v = transform(v)
            _structured_assign(fp, i, v)
    fp.flush()
    return np.lib.format.open_memmap(filename, mode='r')


class MemMap(Dataset):
    def __init__(self, filename, transform=None):
        super().__init__()
        if transform:
            assert callable(transform), 'transform should be callable'
        self.filename = filename
        self.transform = transform
        self.mmap = np.lib.format.open_memmap(filename, mode='r')

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, index):
        item = self.mmap[index].copy()
        item = _structured_unfold(item)
        if self.transform:
            item = self.transform(item)
        return item
    
    create = create_mmap_dataset



if __name__ == '__main__':
    fp = np.memmap('filename.npy', dtype='float32', mode='w+', shape=(3, 4))
