from xvision.utils.config import CfgNode as CN

_C = CN()
cfg = _C

_C.workdir = 'workspace/nima'
_C.lr = 0.2
_C.momentum = 0.9
_C.weight_decay = 5e-4
_C.batch_size = 256
_C.start_step = 0       # for resume
_C.total_steps = 10000   # training steps
_C.num_workers = 8
_C.eval_interval = 5    # evaluate interval
