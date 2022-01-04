import argparse
import time

import torch
from torch.cuda.amp import autocast
from torch.optim import lr_scheduler

from constants import *
from training import models
from training.tools.metrics import AverageMeter, ProgressMeter, accuracy, eval_metrics

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', metavar='ARCH', default='xception', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: xception)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--prefix', type=str, default=FACE_FORENSICS,
                        choices=[FACE_FORENSICS, FACE_FORENSICS_DF, FACE_FORENSICS_F2F,
                                 FACE_FORENSICS_FSW, FACE_FORENSICS_NT, FACE_FORENSICS_FSH,
                                 CELEB_DF, DEEPER_FORENSICS, DFDC],
                        help='dataset')
    parser.add_argument('--compression-version', type=str, default='c23', choices=['c23', 'c40'])
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--use-amp', action='store_true',
                        help='Automatic Mixed Precision')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    return args


def create_optimizer(model, args):
    scheduler = None
    if args.distributed and str(args.arch).startswith('sifdnet'):
        # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        # optimizer = torch.optim.Adam([
        #     {'params': model.module.encoder.parameters()},
        #     {'params': model.module.classifier.parameters()},
        #     {'params': model.module.aspp.parameters()},
        #     {'params': model.module.alam1.parameters(), 'lr': 1e-3},
        #     {'params': model.module.alam2.parameters(), 'lr': 1e-3},
        #     {'params': model.module.decoder.parameters(), 'lr': 1e-2},
        # ], args.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        # optimizer = torch.optim.Adam([
        #     {'params': model.encoder.parameters()},
        #     {'params': model.classifier.parameters()},
        #     {'params': model.alam1.parameters(), 'lr': 1e-3},
        #     {'params': model.alam2.parameters(), 'lr': 1e-3},
        #     {'params': model.aspp.parameters(), 'lr': 1e-3},
        #     {'params': model.decoder.parameters(), 'lr': 1e-3},
        # ], args.lr)

    return optimizer, scheduler


def error_threshold(masks, tau=0.25):
    masks[masks >= tau] = 1.
    masks[masks < tau] = 0.
    return masks.long()


def train(train_loader, model, scaler, optimizer, loss_functions, epoch, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # pw_acc = AverageMeter('Pixel-wise Acc', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, top1], logger, prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for batch_idx, sample in enumerate(train_loader):
        if args.gpu is not None:
            images = sample['images'].cuda(args.gpu, non_blocking=True)
            labels = sample['labels'].cuda(args.gpu, non_blocking=True)
            masks = sample['masks'].cuda(args.gpu, non_blocking=True)
        else:
            images, labels, masks = sample['images'], sample['labels'], sample['masks']

        # compute output
        with autocast(enabled=args.use_amp):
            lambda1, lambda2 = 1., 1.
            outputs = model(images)
            if isinstance(outputs, tuple):
                if len(outputs) > 2:
                    labels_pred, masks_pred, masks_alam = outputs
                    # classification Loss
                    loss_classifier = loss_functions['classifier_loss'](labels_pred, labels)
                    # ALAM Loss
                    loss_alam = torch.sum(torch.tensor(
                        [loss_functions['map_loss'](masks_, masks) for masks_ in masks_alam]
                    ))
                    # Binary Map Loss
                    masks_binary = error_threshold(torch.clone(masks)).squeeze()
                    loss_decoder_mask = loss_functions['binary_map_loss'](masks_pred, masks_binary)
                    # Overall Loss
                    loss = lambda1 * loss_classifier + lambda2 * (loss_alam + loss_decoder_mask)
                elif len(outputs) == 2:
                    labels_pred, masks_pred = outputs
                    loss_classifier = loss_functions['classifier_loss'](labels_pred, labels)
                    if masks_pred.shape[1] == 1:
                        loss_masks = loss_functions['map_loss'](masks_pred, masks)
                    else:
                        masks_binary = error_threshold(torch.clone(masks)).squeeze()
                        loss_masks = loss_functions['binary_map_loss'](masks_pred, masks_binary)
                    loss = lambda1 * loss_classifier + lambda2 * loss_masks
            else:
                labels_pred = outputs
                loss = loss_functions['classifier_loss'](labels_pred, labels)

        # measure accuracy and record loss
        acc1, = accuracy(labels_pred, labels)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do Adam step
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scaler.update()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % args.print_freq == 0:
            progress.display(batch_idx + 1)
        if (batch_idx + 1) % 3000 == 0:
            break


def validate(val_loader, model, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    miou = AverageMeter('Mean IoU', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1, miou], logger, prefix='Validation: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, sample in enumerate(val_loader):
            if args.gpu is not None:
                images = sample['images'].cuda(args.gpu, non_blocking=True)
                labels = sample['labels'].cuda(args.gpu, non_blocking=True)
                masks = sample['masks']
            else:
                images, labels, masks = sample['images'], sample['labels'], sample['masks']
            masks = error_threshold(masks)

            # compute output
            with autocast(enabled=args.use_amp):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    if len(outputs) > 2:
                        labels_pred, masks_pred, masks_alam = outputs
                    elif isinstance(outputs, tuple) and len(outputs) == 2:
                        labels_pred, masks_pred = outputs
                else:
                    labels_pred = outputs

            # measure accuracy and record loss
            acc1, = accuracy(labels_pred, labels)
            top1.update(acc1[0].cpu(), images.size(0))
            if isinstance(outputs, tuple):
                if masks_pred.shape[1] == 1:
                    masks_pred = error_threshold(masks_pred)
                else:
                    masks_pred = torch.argmax(masks_pred, dim=1)
                overall_acc, avg_jacc = eval_metrics(masks.cpu(), masks_pred.cpu(), 2)
                miou.update(avg_jacc, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(val_loader):
                progress.display(batch_idx + 1)
            if (batch_idx + 1) % 1000 == 0:
                break

    return top1.avg, miou.avg
