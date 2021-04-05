import time
from os import getloadavg
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from xvision.utils.logger import get_logger
from xvision.utils.saver import Saver
from xvision.models import fa as models
from xvision.ops.utils import group_parameters
from xvision.datasets.jdlmk import JDLandmark
from xvision.utils.meter import MetricLogger, SmoothedValue
from xvision.ops.euclidean_loss import euclidean_loss
from xvision.ops.nme import IbugScore

from transform import Transform


def process_batch(batch, device):
    return {
        k: v.to(device, non_blocking=True) for k, v in batch.items()
    }


def train_steps(model, loader, optimizer, lr_scheduler, score_fn, device, num_steps):
    model.train()
    meter = MetricLogger()
    meter.add_meter('lr', SmoothedValue(1, fmt='{value:.6f}'))
    meter.add_meter('loss', SmoothedValue())
    meter.add_meter('score', SmoothedValue())
    for _ in range(num_steps):
        batch = next(loader)
        batch = process_batch(batch, device)
        image = batch['image'].permute(0, 3, 1, 2).float()
        shape = batch['shape']
        pred = model(image).reshape(shape.shape)
        loss = euclidean_loss(pred, shape, reduction='mean')
        score = score_fn(pred, shape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        meter.update(loss=loss.item(), score=score.item())
    meter.update(lr=optimizer.param_groups[0]['lr'])
    return meter


@torch.no_grad()
def evaluate(model, loader, score_fn, device):
    model.eval()
    meter = MetricLogger()
    meter.add_meter('loss', SmoothedValue(fmt='{global_avg: .4f}'))
    meter.add_meter('score', SmoothedValue(fmt='{global_avg: .2f}'))
    for batch in loader:
        batch = process_batch(batch, device)
        image = batch['image'].permute(0, 3, 1, 2).float()
        shape = batch['shape']
        preds = model(image).reshape(shape.shape)
        loss = euclidean_loss(preds, shape, reduction='mean')
        score = score_fn(preds, shape)
        meter.update(loss=loss.item(), score=score.item())
    return meter


def main(args):
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(workdir/'log.txt')
    logger.info(f'config:\n{args}')
    saver = Saver(workdir, keep_num=10)
    # dump all configues
    with open(workdir / 'config.yml', 'wt') as f:
        args.dump(stream=f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'use device: {device}')

    model = models.__dict__[args.model.name]()

    model.to(device)
    parameters = group_parameters(model, bias_decay=0)
    optimizer = SGD(parameters, args.lr, args.momentum,
                    weight_decay=args.weight_decay)
    lr_scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=20,
                              total_steps=args.total_steps, pct_start=0.1, final_div_factor=100)

    # datasets
    datadir = Path(args.jdlmk)
    valtransform = Transform(args.dsize, args.padding, args.meanshape)
    traintransform = Transform(args.dsize, args.padding, args.meanshape, args.augments)
    traindata = JDLandmark(datadir / 'Train/landmark',
                           datadir / 'Train/picture', traintransform)
    valdata = JDLandmark(datadir / 'Val/landmark',
                         datadir / 'Val/picture', valtransform)

    trainloader = DataLoader(traindata, args.batch_size, shuffle=True,
                             drop_last=True, num_workers=args.num_workers, pin_memory=True)
    valloader = DataLoader(valdata, args.batch_size, False,
                           num_workers=args.num_workers, pin_memory=False)

    def repeat(loader):
        while True:
            for batch in loader:
                yield batch

    best_loss = 1e-9
    state = {
        'model': model,
        'optmizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'step': 0,
        'loss': best_loss
    }
    score_fn = IbugScore(args.left_eye, args.right_eye)
    saver.save(0, state)
    repeatloader = repeat(trainloader)
    start = time.time()
    for step in range(0, args.total_steps, args.eval_interval):
        num_steps = min(args.eval_interval, args.total_steps - step)
        step += num_steps
        trainmeter = train_steps(
            model, repeatloader, optimizer, lr_scheduler, score_fn, device, num_steps)
        evalmeter = evaluate(model, valloader, score_fn, device)
        curr_loss = evalmeter.meters['loss'].global_avg
        finish = time.time()
        img_s = num_steps * args.batch_size / (finish - start)
        logger.info(
            f'step: [{step}/{args.total_steps}] img/s: {img_s:.2f} train: [{trainmeter}] eval: [{evalmeter}]')

        state = {
            'model': model,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'step': step,
            'loss': curr_loss
        }
        saver.save(step, state)

        if curr_loss < best_loss:
            saver.save_best(state)
            best_loss = curr_loss

        start = time.time()

if __name__ == '__main__':
    from config import cfg
    cfg.parse_args()
    cfg.freeze()
    main(cfg)
