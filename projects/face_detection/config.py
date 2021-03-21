from xvision.utils.config import CfgNode as CN

_C = CN()
cfg = _C

# workspace
_C.workdir = "workspace/fd"

# anchor layouts and model structure
# data size
_C.dsize = (320, 320)   # input image size
_C.strides = [8, 16, 32, 64]    # strides of features
# feature map size each stride
_C.fsizes = [(40, 40), (20, 20), (10, 10), (5, 5)]
_C.layouts = [  # anchor layout
    [10, 16, 24],
    [32, 48],
    [64, 96],
    [128, 192, 256]
]
_C.iou_threshold = 0.3
_C.encode_mean = None
_C.encode_std = [.1, .1, .2, .2]  # center .1, size: .2

_C.model = CN(recursive=False)
_C.model.name = "Slim"
_C.model.args = []
_C.model.kwargs = CN()

# training parameters
_C.lr = 0.01
_C.momentum = 0.9
_C.weight_decay = 5e-4
_C.gamma = 0.1
_C.batch_size = 128
_C.start_step = 0       # for resume
_C.total_steps = 1000   # training steps
_C.num_workers = 8
_C.device = None        # auto determine
_C.eval_interval = 10    # evaluate interval

# data parameters
_C.train_label = '/Users/jimmy/Documents/data/WIDER/retinaface_gt_v1.1/train/label.txt'
_C.train_image = '/Users/jimmy/Documents/data/WIDER/WIDER_train/images'
_C.val_label = '/Users/jimmy/Documents/data/WIDER/retinaface_gt_v1.1/val/label.txt'
_C.val_image = '/Users/jimmy/Documents/data/WIDER/WIDER_val/images'

