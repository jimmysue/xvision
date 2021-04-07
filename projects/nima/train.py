#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file - main.py
Main script to train the aesthetic model on the AVA dataset.

Copyright (C) Yunxiao Shi 2017 - 2020
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import argparse
import os
import time
from xvision.utils.logger import get_logger

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from opencv_transforms import transforms
from pathlib import Path
import torchvision.datasets as dsets
import torchvision.models as models

from xvision.datasets import AVADataset
from xvision.ops.functional import emd_loss
from xvision.models import NIMA
from xvision.utils import get_logger, MetricLogger, SmoothedValue

def main(config):
    device = torch.device("cuda" if torch.cuda.is_axvisionlable() else "cpu")

    workdir = Path(config.work_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(workdir / 'log.txt')
    tbdir = workdir / 'tb'
    writer = SummaryWriter(log_dir=tbdir)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()])

    base_model = models.vgg16(pretrained=False)
    model = NIMA(base_model)

    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pth' % config.warm_start_epoch)))
        logger.info('Successfully loaded model epoch-%d.pth' % config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    logger.info('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        try:
            logger.info(f'trainset: {config.train_csv_file}, valset: {config.val_csv_file}')
            trainset = AVADataset(csv_file=config.train_csv_file, image_dir=config.img_path, transform=train_transform)
            valset = AVADataset(csv_file=config.val_csv_file, image_dir=config.img_path, transform=val_transform)

            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                shuffle=True, num_workers=config.num_workers, drop_last=False, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
                shuffle=False, num_workers=config.num_workers, pin_memory=True)
            # for early stopping
            count = 0
            init_val_loss = float('inf')
            train_losses = []
            val_losses = []
            for epoch in range(config.warm_start_epoch, config.epochs):
                batch_losses = []
                try:
                    start = time.time()
                    meter = SmoothedValue(fmt='{global_avg: .2f}')
                    for i, data in enumerate(train_loader):
                        data_time = time.time() - start
                        images = data['image'].to(device, non_blocking=True)
                        labels = data['annotations'].to(device, non_blocking=True).float()
                        outputs = model(images)
                        optimizer.zero_grad()

                        loss = emd_loss(labels, outputs)
                        batch_losses.append(loss.item())

                        loss.backward()

                        optimizer.step()
                        iter_time = time.time() - start
                        data_pct = data_time / iter_time * 100
                        meter.update(data_pct)
                        logger.info('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.item()))
                        logger.info('Data/Iter: {}'.format(meter))
                        writer.add_scalar('batch train loss', loss.item(), i + epoch * (len(trainset) // config.train_batch_size + 1))
                        start = time.time()
                except Exception as e:
                    logger.exception(e)

                avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
                train_losses.append(avg_loss)
                logger.info('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))

                # exponetial learning rate decay
                if config.decay:
                    if (epoch + 1) % 10 == 0:
                        conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                        dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                        optimizer = optim.SGD([
                            {'params': model.features.parameters(), 'lr': conv_base_lr},
                            {'params': model.classifier.parameters(), 'lr': dense_lr}],
                            momentum=0.9
                        )

                # do validation after each epoch
                batch_val_losses = []
                for data in val_loader:
                    images = data['image'].to(device)
                    labels = data['annotations'].to(device).float()
                    with torch.no_grad():
                        outputs = model(images)
                    val_loss = emd_loss(labels, outputs)
                    batch_val_losses.append(val_loss.item())
                avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
                val_losses.append(avg_val_loss)
                logger.info('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
                writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)

                # Use early stopping to monitor training
                if avg_val_loss < init_val_loss:
                    init_val_loss = avg_val_loss
                    # save model weights if val loss decreases
                    logger.info('Saving model...')
                    torch.save(model.state_dict(),  workdir / f'epoch-{epoch + 1}.pth')
                    logger.info('Done.\n')
                    # reset count
                    count = 0
                elif avg_val_loss >= init_val_loss:
                    count += 1
                    if count == config.early_stopping_patience:
                        logger.info('Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                        break

            print('Training completed.')
        except Exception as e:
            logger.error(f"exception: {e}")
            logger.exception(e)
        '''
        # use tensorboard to log statistics instead
        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./loss.png')
        '''

    if config.test:
        model.eval()
        # compute mean score
        test_transform = val_transform
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.img_path, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        for data in test_loader:
            image = data['image'].to(device)
            output = model(image)
            output = output.view(10, 1)
            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2
            predicted_std = predicted_std ** 0.5
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--img_path', type=str, default='/data/dataset/AVA_dataset/images/')
    parser.add_argument('--train_csv_file', type=str, default='/data/dataset/AVA_dataset/csv/train.csv')
    parser.add_argument('--val_csv_file', type=str, default='/data/dataset/AVA_dataset/csv/test.csv')
    parser.add_argument('--test_csv_file', type=str, default='/data/dataset/AVA_dataset/csv/test.csv')

    # training parameters
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--conv_base_lr', type=float, default=5e-3)
    parser.add_argument('--dense_lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./ckpts')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true')
    parser.add_argument('--work_dir', type=str, default='./nima')

    config = parser.parse_args()
    main(config)


