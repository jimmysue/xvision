import cv2
from xvision.utils.config import CfgNode as CN

_C = CN()
cfg = _C

# workspace
_C.workdir = "workspace/fd"

# anchor layouts and model structure
# data size
_C.num_classes = 1
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
_C.iou_threshold = 0.35
_C.encode_mean = None
_C.encode_std = [.1, .1, .2, .2]  # center .1, size: .2

_C.model = CN(recursive=False)
_C.model.name = "RFB"
_C.model.args = []
_C.model.kwargs = CN()

# training parameters
_C.lr = 0.2
_C.momentum = 0.9
_C.weight_decay = 5e-4
_C.gamma = 0.1
_C.batch_size = 256
_C.start_step = 0       # for resume
_C.total_steps = 10000   # training steps
_C.num_workers = 8
_C.device = None        # auto determine
_C.eval_interval = 100    # evaluate interval
_C.augments = CN()
_C.augments.inters = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA]
_C.augments.rotation = 5
_C.augments.max_face = 256
_C.augments.min_face = 6
_C.augments.symmetry = [1, 0, 2, 4, 3] # shape symmetry indice for mirror augmentation

# data parameters
_C.train_label = '/dockerdata/train/label.txt'
_C.train_image = '/dockerdata/WIDER_train/images'
_C.val_label = '/dockerdata/val/label.txt'
_C.val_image = '/dockerdata/WIDER_val/images'

# configuration for test
_C.test = CN(new_allowed=True)
_C.test.image_dir = '/Users/jimmy/Documents/data/WIDER/WIDER_val/images'
_C.test.result_dir = 'result'
_C.test.iou_threshold = 0.5
_C.test.score_threshold = 0.1
_C.test.long_size = 320    # long size when resize input image
