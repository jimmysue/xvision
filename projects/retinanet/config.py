from xvision.utils.config import CfgNode as CN


_C = CN()

_C.anchors = [
    {'stride': 32,  'size': None, 'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspect': [0.5, 1.0, 2.0]},
    {'stride': 64,  'size': None, 'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspect': [0.5, 1.0, 2.0]},
    {'stride': 128, 'size': None, 'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspect': [0.5, 1.0, 2.0]},
    {'stride': 256, 'size': None, 'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspect': [0.5, 1.0, 2.0]},
    {'stride': 512, 'size': None, 'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspect': [0.5, 1.0, 2.0]},
]


print(f'{_C}')