from xvision.utils.config import CfgNode as CN


_C = CN()
cfg = _C
_C.anchors = [
    {'stride': 32,  'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspects': [0.5, 1.0, 2.0]},
    {'stride': 64,  'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspects': [0.5, 1.0, 2.0]},
    {'stride': 128, 'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspects': [0.5, 1.0, 2.0]},
    {'stride': 256, 'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspects': [0.5, 1.0, 2.0]},
    {'stride': 512, 'scales': [2**0, 2**(1/3), 2 ** (2/3)], 'aspects': [0.5, 1.0, 2.0]},
]


print(f'{_C}')