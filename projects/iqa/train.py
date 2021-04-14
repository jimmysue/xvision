import logging
import time
import cv2

import torch

from pathlib import Path
from opencv_transforms import transforms
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

from xvision.utils.logger import set_logger
from xvision.models import iqa
from xvision.ops.utils import group_parameters
from xvision.datasets import AVADataset
from xvision.utils.saver import Saver
from xvision.datasets.loader import repeat_loader
from xvision.ops.emd_loss import emd_loss
from xvision.utils.meter import MetricLogger, SmoothedValue
from xvision.datasets.memmap import MemMap

from transform import Transform


class BatchProcessor(object):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def __call__(self, data):
        images = data['image'].to(self.device, non_blocking=True)
        labels = data['annotations'].to(self.device, non_blocking=True).float()
        return images, labels


def train_steps(model, loader, optimizer, lr_scheduler, criterion, batch_process, steps):
    model.train()
    meter = MetricLogger()
    meter.add_meter('lr', SmoothedValue(window_size=1, fmt='{global_avg:.6f}'))
    for _ in range(steps):
        data = next(loader)
        images, labels = batch_process(data)
        outputs = model(images)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        meter.meters['loss'].update(loss.item())
    meter.meters['lr'].update(optimizer.param_groups[0]['lr'])
    return meter


@torch.no_grad()
def evaluate(model, loader, criterion, process):
    model.eval()
    meter = MetricLogger()
    for batch in loader:
        images, labels = process(batch)
        outputs = model(images)
        loss = criterion(outputs, labels)
        meter.meters['loss'].update(loss.item(), n=images.size(0))
    return meter


def cache_transform(item):
    image = cv2.imread(str(item['path']))
    image = cv2.resize(image, (256, 256))  # FIXME: remove hard codes
    return {
        'image': image,
        'annotations': item['annotations']
    }


def create_memmap(label, image, path, parallel):
    gen = AVADataset.parse(csv_file=label, image_dir=image)
    MemMap.create(path, gen, cache_transform, parallel)


def main(cfg):
    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    set_logger(workdir / 'log.txt')
    saver = Saver(workdir, keep_num=10)
    logging.info(f'config: \n{cfg}')
    logging.info(f'use device: {device}')

    model = iqa.__dict__[cfg.model.name](**cfg.model.kwargs)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model_dp = nn.DataParallel(model)
    else:
        model_dp = model

    train_transform = Transform(
        transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))

    val_transform = Transform(
        transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor()]))

    if not Path(cfg.ava.train_cache).exists():
        create_memmap(cfg.ava.train_labels, cfg.ava.images,
                      cfg.ava.train_cache, cfg.num_workers)
    if not Path(cfg.ava.val_cache).exists():
        create_memmap(cfg.ava.train_labels, cfg.ava.images,
                      cfg.ava.val_cache, cfg.num_workers)

    trainset = MemMap(cfg.ava.train_cache, train_transform)
    valset = MemMap(cfg.ava.val_cache, val_transform)

    total_steps = len(trainset) // cfg.batch_size * cfg.num_epochs
    eval_interval = len(trainset) // cfg.batch_size
    logging.info(f'total steps: {total_steps}, eval interval: {eval_interval}')
    model_dp.train()
    parameters = group_parameters(model)
    optimizer = SGD(parameters, cfg.lr, cfg.momentum,
                    weight_decay=cfg.weight_decay)

    lr_scheduler = OneCycleLR(optimizer, max_lr=cfg.lr, div_factor=cfg.lr/cfg.warmup_lr,
                              total_steps=total_steps, pct_start=0.01, final_div_factor=cfg.warmup_lr/cfg.final_lr)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size,
                                             shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    curr_loss = 1e9
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'step': 0,  # init step,
        'cfg': cfg,
        'loss': curr_loss
    }

    saver.save(0, state)

    trainloader = repeat_loader(train_loader)
    batch_processor = BatchProcessor(device)
    start = time.time()
    for step in range(0, total_steps, eval_interval):
        num_steps = min(step + eval_interval, total_steps) - step
        step += num_steps
        trainmeter = train_steps(model_dp, trainloader, optimizer,
                                 lr_scheduler, emd_loss, batch_processor, num_steps)
        valmeter = evaluate(model_dp, val_loader, emd_loss, batch_processor)
        finish = time.time()
        img_s = cfg.batch_size * eval_interval / (finish - start)
        loss = valmeter.meters['loss'].global_avg

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'step': step,  # init step,
            'cfg': cfg,
            'loss': loss
        }
        saver.save(step, state)

        if loss < curr_loss:
            curr_loss = loss
            saver.save_best(state)

        logging.info(
            f'step: [{step}/{total_steps}] img_s: {img_s:.2f} train: [{trainmeter}] eval:[{valmeter}]')
        start = time.time()


if __name__ == '__main__':
    from config import cfg
    cfg.parse_args()
    cfg.freeze()
    main(cfg)
