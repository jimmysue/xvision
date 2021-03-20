import copy
import numpy as np
from torch.utils.data import Dataset

def _make_inheritance(cls, transform):
    class _DerivedClass(cls): 
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item = transform(item)
            return item
    return _DerivedClass

class TDataset(Dataset):
    def __or__(self, transform):
        assert callable(transform), "transform should be callable"

        result = copy.copy(self)
        result.__class__ = _make_inheritance(self.__class__, transform)
        return result
