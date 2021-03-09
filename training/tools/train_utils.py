import argparse
import time

import torch
from torch.cuda.amp import autocast

from preprocessing.constants import CELEB_DF, FACE_FORENSICS_DF, FACE_FORENSICS_FSH, DEEPER_FORENSICS, DFDC
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
    parser.add_argument('--prefix', type=str, default=CELEB_DF,
                        choices=[CELEB_DF, FACE_FORENSICS_DF, FACE_FORENSICS_FSH, DEEPER_FORENSICS, DFDC],
                        help='(default: celeb-df)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
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


def train(train_loader, model, scaler, optimizer, loss_functions, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    pw_acc = AverageMeter('Pixel-wise Acc', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, top1, pw_acc], prefix="Epoch: [{}]".format(epoch))

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
                labels_pred, masks_pred = outputs
                loss_classifier = loss_functions['classifier_loss'](labels_pred, labels)
                if masks_pred.size(1) == 1:
                    loss_mask = loss_functions['map_loss'](masks_pred, masks)
                    loss = lambda1 * loss_classifier + lambda2 * loss_mask
                else:
                    binary_masks = masks.clone()
                    binary_masks[binary_masks >= 0.25] = 1.
                    binary_masks[binary_masks < 0.25] = 0.
                    binary_masks = binary_masks.squeeze().long()
                    loss_binary_mask = loss_functions['binary_map_loss'](masks_pred, binary_masks)
                    # loss_mask = loss_functions['map_loss'](masks_pred[:, -1], masks.squeeze())
                    loss = lambda1 * loss_classifier + lambda2 * loss_binary_mask
            else:
                labels_pred = outputs
                loss = loss_functions['classifier_loss'](labels_pred, labels)

        # measure accuracy and record loss
        acc1, = accuracy(labels_pred, labels)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        # if isinstance(outputs, tuple):
        #     if masks_pred.size(1) == 1:
        #         overall_acc = eval_metrics(masks.cpu(), masks_pred.cpu(), 256)
        #     else:
        #         mask_pred1 = torch.argmax(masks_pred, dim=1)
        #         overall_acc = eval_metrics(masks.cpu(), mask_pred1.cpu(), 256)
        #     pw_acc.update(overall_acc, images.size(0))

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


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    pw_acc = AverageMeter('Pixel-wise Acc', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1, pw_acc], prefix='Validation: ')

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

            # compute output
            with autocast(enabled=args.use_amp):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    labels_pred, masks_pred = outputs
                else:
                    labels_pred = outputs

            # measure accuracy and record loss
            acc1, = accuracy(labels_pred, labels)
            top1.update(acc1[0].cpu(), images.size(0))
            if isinstance(outputs, tuple):
                if masks_pred.size(1) == 1:
                    overall_acc = eval_metrics(masks.cpu(), masks_pred.cpu(), 256)
                else:
                    mask_pred1 = torch.argmax(masks_pred, dim=1)
                    overall_acc = eval_metrics(masks.cpu(), mask_pred1.cpu(), 256)
                pw_acc.update(overall_acc, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(val_loader):
                progress.display(batch_idx + 1)
            if (batch_idx + 1) % 1000 == 0:
                break

    return top1.avg, pw_acc.avg
