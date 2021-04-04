from xvision.utils.config import CfgNode as CN

_C = CN()
cfg = _C

_C.workdir = 'workspace/fa'
_C.dsize = 128
_C.lr = 0.2
_C.momentum = 0.9
_C.weight_decay = 5e-4
_C.batch_size = 256
_C.start_step = 0       # for resume
_C.total_steps = 10000   # training steps
_C.num_workers = 8
_C.eval_interval = 100    # evaluate interval


# datasets

_C.wflw = 'path/to/wflw/data'
_C.ibug = 'path/to/ibug/data'

_C.augments = CN()
_C.augments.rotate = 15
_C.augments.scale = 0.2
_C.augments.translate = 0.15

# pose balance distribution
