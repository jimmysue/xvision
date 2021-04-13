import cv2
from xvision.utils.config import CfgNode as CN

_C = CN()
cfg = _C

# workspace
_C.workdir = "workspace/retinaface_res50pfn"

# anchor layouts and model structure
# data size
_C.num_classes = 1
_C.dsize = (840, 840)   # input image size

_C.anchors = [
    {'stride': 8, 'scales':  [2, 4], 'aspects':[1]},
    {'stride': 16, 'scales': [4, 8], 'aspects': [1]},
    {'stride': 32, 'scales': [8, 16], 'aspects': [1]},
]
_C.iou_threshold = 0.35
_C.encode_mean = None
_C.encode_std = [.1, .1, .2, .2]  # center .1, size: .2
_C.image_mean = [104, 117, 123] # bgr order
_C.model = CN(recursive=False)
_C.model.name = "retinaface_res50fpn"
_C.model.args = []
_C.model.kwargs = CN()
_C.model.kwargs.pretrained_path = 'workspace/resnet50-19c8e357.pth'

# training parameters
_C.lr = 0.2
_C.momentum = 0.9
_C.weight_decay = 5e-4
_C.gamma = 0.1
_C.batch_size = 256
_C.start_step = 0       # for resume
# _C.total_steps = 10000   # training steps
_C.num_workers = 8
_C.num_epochs = 100      #
_C.device = None        # auto determine
_C.eval_interval = 100    # evaluate interval
_C.augments = CN()
_C.augments.inters = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA]
_C.augments.rotation = 5
_C.augments.max_face = 512
_C.augments.min_face = 6
_C.augments.symmetry = [1, 0, 2, 4, 3] # shape symmetry indice for mirror augmentation

# data parameters
_C.train_label = '/dockerdata/WIDER_face/train/label.txt'
_C.train_image = '/dockerdata/WIDER_face/train/images'
_C.val_label = '/dockerdata/WIDER_face/val/label.txt'
_C.val_image = '/dockerdata/WIDER_face/val/images'

# configuration for test
_C.test = CN(new_allowed=True)
_C.test.image_dir = '/Users/jimmy/Documents/data/WIDER/WIDER_val/images'
_C.test.result_dir = 'result'
_C.test.iou_threshold = 0.5
_C.test.score_threshold = 0.1
_C.test.long_size = 320    # long size when resize input image
