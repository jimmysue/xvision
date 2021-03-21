#! /usr/bin/env python
import time
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from pathlib import Path
from xvision.utils import get_logger, Saver
from xvision.data import WiderFace, wider_collate, BasicTransform
from xvision.data.loader import repeat_loader
from xvision.model import fd as fd_models
from xvision.model import initailize_model
from xvision.ops.anchors import BBoxAnchors
from xvision.ops.multibox import score_box_point_loss, score_box_loss
from xvision.utils.meter import MetricLogger, SmoothedValue
from xvision.data.wider import WiderFace


def batch_to(batch, device):
    image = batch.pop('image').to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255
    batch = {
        k: [i.to(device, non_blocking=True) for i in v] for k, v in batch.items()
    }
    batch['image'] = image
    return batch


def evaluate(model, dataloader, prior, device):
    model.eval()
    meter = MetricLogger()
    meter.add_meter('total', SmoothedValue(fmt='{global_avg:.4f}'))
    meter.add_meter('score', SmoothedValue(fmt='{global_avg:.4f}'))
    meter.add_meter('box', SmoothedValue(fmt='{global_avg:.4f}'))

    for batch in dataloader:
        batch = batch_to(batch, device)
        image = batch['image']
        box = batch['bbox']
        label = batch['label']
        pred_score, pred_box, pred_point = model(image)
        pred_score = pred_score.squeeze(-1)
        pred_point = pred_point.reshape(pred_point.shape[0], pred_point.shape[1], -1, 2)
        
        with torch.no_grad():
            target_score, target_box= prior(label, box)
            box_delta =  prior.encode_bboxes(target_box)

        score_loss, box_loss = score_box_loss(target_score, box_delta, pred_score, pred_box)
        loss = score_loss + box_loss
        meter.meters['score'].update(score_loss.item())
        meter.meters['box'].update(box_loss.item())
        meter.meters['total'].update(loss.item())

    model.train()
    return meter

def main(args):
    # prepare workspace
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(workdir / 'log.txt')
    logger.info(f'config: \n{args}')

    saver = Saver(workdir, keep_num=10)
    
    # prepare dataset
    train_transform = BasicTransform(dsize=args.dsize)
    trainset = WiderFace(args.train_label, args.train_image, min_size=10, with_points=True, transform=train_transform)
    valset = WiderFace(args.val_label, args.val_image, transform=train_transform)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers, pin_memory=True, 
        collate_fn=wider_collate, drop_last=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=wider_collate)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.device:
        logger.info(f'user specify device: {args.device}')
        device = torch.device(args.device)

    logger.info(f'use device: {device}')
    # model
    model = initailize_model(fd_models, args.model.name, *args.model.args, **args.model.kwargs)
    prior = BBoxAnchors(args.dsize, args.strides, args.fsizes, args.layouts, args.iou_threshold, args.encode_mean, args.encode_std)
    # optimizer and lr scheduler
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = OneCycleLR(optimizer, max_lr = args.lr, total_steps = args.total_steps, pct_start=0.1, final_div_factor=1000)
    trainloader = repeat_loader(trainloader)
    
    model.to(device)
    prior.to(device)
    model.train()
    best_loss = 1e9
    state ={
        'model': model.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': 0,
        'loss': best_loss
    }
    saver.save(0, state)

    
    def reset_meter():
        meter = MetricLogger()
        meter.add_meter('lr', SmoothedValue(1, fmt='{value:.5f}'))
        return meter
    
    train_meter = reset_meter()
    start = time.time()
    for step in range(args.start_step, args.total_steps):
        batch = next(trainloader)
        batch = batch_to(batch, device)
        image = batch['image']
        box = batch['bbox']
        point = batch['point']
        mask = batch['mask']
        label = batch['label']

        pred_score, pred_box, pred_point = model(image)
        pred_score = pred_score.squeeze(-1)
        pred_point = pred_point.reshape(pred_point.shape[0], pred_point.shape[1], -1, 2)
        
        with torch.no_grad():
            target_score, target_box, target_point, point_mask = prior(label, box, point, mask)
            box_delta =  prior.encode_bboxes(target_box)
            point_delta = prior.encode_points(target_point)

        score_loss, box_loss, point_loss = score_box_point_loss(target_score, box_delta, point_delta, pred_score, pred_box, pred_point, point_mask)
        loss = score_loss + box_loss + point_loss

        train_meter.meters['score'].update(score_loss.item())
        train_meter.meters['box'].update(box_loss.item())
        train_meter.meters['point'].update(point_loss.item())
        train_meter.meters['total'].update(loss.item())
        train_meter.meters['lr'].update(optimizer.param_groups[0]['lr'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if (step + 1) % args.eval_interval == 0:
            duration = time.time() - start
            img_s = args.eval_interval * args.batch_size / duration
            eval_meter = evaluate(model, valloader, prior, device)

            logger.info(f'Step [{step + 1}/{args.total_steps}] img/s: {img_s:.2f} train: [{train_meter}] eval: [{eval_meter}]')
            train_meter = reset_meter()
            start = time.time()
            curr_loss = eval_meter.meters['total'].global_avg
            state ={
                'model': model.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': curr_loss,

            }
            saver.save(step + 1, state)
            
            if (curr_loss < best_loss):
                best_loss = curr_loss
                saver.save_best(state)


                



if __name__ == '__main__':
    import sys
    from config import cfg

    value_args = []
    for arg in sys.argv[1::]:
        if arg.startswith('@'):
            cfg.merge_from_file(arg[1::])
        else:
            value_args.append(arg)
    if value_args:
        cfg.merge_from_list(value_args)
        
    cfg.freeze()
    main(cfg)
