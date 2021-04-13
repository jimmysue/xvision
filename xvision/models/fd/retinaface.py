import logging
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F

from xvision.models.fd.moduels import MobileNetV1 as MobileNetV1
from xvision.models.fd.moduels import FPN as FPN
from xvision.models.fd.moduels import SSH as SSH

# copy from https://github.com/biubug6/Pytorch_Retinaface


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(
            inchannels, self.num_anchors, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        return self.conv1x1(x)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors*4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        return self.conv1x1(x)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors*10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        return self.conv1x1(x)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load(
                    "./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            flag = cfg['pretrain'] == True
            backbone = models.resnet50(pretrained=flag)
            if isinstance(cfg['pretrain'], str):
                logging.info('Load pretrained from path: {}'.format(cfg['pretrain']))
                backbone.load_state_dict(torch.load(cfg['pretrain']))

        self.body = _utils.IntermediateLayerGetter(
            backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(
            fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(
            fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(
            fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        predictions = []
        for clshead, boxhead, lmkhead, feat in zip(self.ClassHead, self.BboxHead, self.LandmarkHead, features):
            predictions.append(
                (
                    clshead(feat), boxhead(feat), lmkhead(feat)
                )
            )

        return predictions


def retinaface_res50fpn(pretrained=False):
    cfg = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'pretrain': True,
        'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        'in_channel': 256,
        'out_channel': 256
    }
    cfg['pretrain'] = pretrained
    return RetinaFace(cfg)


if __name__ == '__main__':
    model = retinaface_res50fpn()
    inputs = torch.rand(1, 3, 840, 840)

    res = model(inputs)

    for score, box, lmk in res:
        print('{}, {}, {}'.format(score.shape, box.shape, lmk.shape))

