from xvision.utils.config import CfgNode as CN


_C = CN()
cfg = _C

_C.workdir = 'workspace/iqa'

_C.lr = 0.01           # max lr in OneCycleLR 
_C.warmup_lr = 0.001   
_C.final_lr = 0.00001
_C.num_epochs = 100
_C.momentum = .9
_C.weight_decay = 5e-5
_C.batch_size = 128
_C.num_workers = 32

_C.ava = CN()  # ava data info
_C.ava.images = '/dockerdata/AVA/'
_C.ava.train_labels = 'workspace/iqa/AVA_dataset/train_labels.csv'
_C.ava.val_labels = 'workspace/iqa/AVA_dataset/val_labels.csv'
_C.ava.train_cache = '/dockerdata/AVA_dataset/train.npy'  # memmap cache dataset to speed up training
_C.ava.val_cache = '/dockerdata/AVA_dataset/val.npy'     # memmap cache dataset to speed up training

_C.model = CN()
_C.model.name = 'nima_vgg16'
_C.model.kwargs = CN()
_C.model.kwargs.pretrained = ''
