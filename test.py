import os
import pickle
import re
import time

import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.cuda.amp import autocast

from training import models
from training.datasets import get_test_dataloader
from training.tools.metrics import AverageMeter, ProgressMeter
from training.tools.metrics import accuracy, eval_metrics, show_metrics
from training.tools.train_utils import parse_args

torch.backends.cudnn.benchmark = True

PICKLE_FILE = "plot/{}/{}_{}.pickle"


def test(test_loader, model, args):
    y_true = []
    y_pred = []
    y_score = []

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    pw_acc = AverageMeter('Pixel-wise Acc', ':6.2f')
    f1_score = AverageMeter('F1-score', ':6.2f')
    mae = AverageMeter('MAE', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, top1, f1_score, pw_acc, mae], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, sample in enumerate(test_loader):
            images, labels, masks = sample['images'].cuda(), sample['labels'].cuda(), sample['masks'].cuda()
            masks[masks >= 0.25] = 1.0
            masks[masks < 0.25] = 0.0
            y_true.extend(labels.tolist())

            # compute output
            with autocast(enabled=args.use_amp):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    labels_pred, masks_pred = outputs
                else:
                    labels_pred = outputs

            pred = torch.argmax(labels_pred, dim=1)
            y_pred.extend(pred.tolist())
            score = torch.nn.functional.softmax(labels_pred, dim=1)
            score = score[:, 1]
            y_score.extend(score.tolist())

            # measure accuracy and record loss
            acc1, = accuracy(labels_pred, labels)
            top1.update(acc1[0], images.size(0))
            if isinstance(outputs, tuple):
                if masks_pred.size(1) == 1:
                    masks_pred[masks_pred >= 0.25] = 1.0
                    masks_pred[masks_pred < 0.25] = 0.0
                    overall_acc, f1 = eval_metrics(masks.cpu(), masks_pred.cpu(), 2)
                    mean_avg_err = F.l1_loss(masks.cpu(), masks_pred.cpu())
                else:
                    mask_pred1 = torch.argmax(masks_pred, dim=1)
                    overall_acc, f1 = eval_metrics(masks.cpu(), mask_pred1.cpu(), 2)
                    mean_avg_err = F.l1_loss(masks.squeeze().cpu(), mask_pred1.cpu())
                pw_acc.update(overall_acc, images.size(0))
                f1_score.update(f1, images.size(0))
                mae.update(mean_avg_err, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(test_loader):
                progress.display(batch_idx + 1)

    with open(PICKLE_FILE.format(args.prefix, args.arch, args.prefix), "wb") as f:
        pickle.dump([y_true, y_pred, y_score], f)

    return pw_acc.avg, mae.avg


def main():
    args = parse_args()
    print(args)
    os.makedirs(os.path.join('plot', args.prefix), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'results'), exist_ok=True)
    if args.resume:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        print("Loading checkpoint {}".format(args.resume))
        model = models.__dict__[args.arch](pretrained=False, num_classes=2)
        model.cuda()
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)

        print("Initializing Data Loader")
        test_loader = get_test_dataloader(model.default_cfg, args)

        print("Starting test")
        pw_acc, mae = test(test_loader, model, args)
        # pw_acc, mae = 0., 0.

        with open(PICKLE_FILE.format(args.prefix, args.arch, args.prefix), "rb") as f:
            y_true, y_pred, y_score = pickle.load(f)
        print(len(y_true), len(y_pred), len(y_score))
        show_metrics(y_true, y_pred, y_score, pw_acc, mae)


if __name__ == '__main__':
    main()
