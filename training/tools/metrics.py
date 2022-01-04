import numpy as np
import torch
from sklearn import metrics

EPS = 1e-10


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.logger = logger
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def _fast_hist(gt, pred, num_classes):
    # mask1 = (true >= 0) & (true < num_classes)
    mask2 = (pred >= 0) & (pred < num_classes)
    hist = torch.bincount(num_classes * gt[mask2] + pred[mask2], minlength=num_classes ** 2, )
    hist = hist.reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc * 100.


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    Args:
        hist: confusion matrix.
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def eval_metrics(gt, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        gt: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    if num_classes == 2:
        gt = gt.long()
        pred = pred.long()
    else:
        gt = (gt * 255.).to(torch.int32)
        pred = (pred * 255.).to(torch.int32)
    hist = torch.zeros((num_classes, num_classes))
    for t, p in zip(gt, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    # avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    # avg_dice = dice_coefficient(hist)
    # return overall_acc, avg_per_class_acc, avg_jacc, avg_dice
    return overall_acc, avg_jacc


def f1_score(gt, pred):
    assert gt.ndim == 1
    assert pred.ndim == 1

    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def calc_tpr(fpr, tpr):
    """
    :param fpr:
    :param tpr:
    :return: tpr_0_01, tpr_0_02, tpr_0_05, tpr_0_10, tpr_0_20, tpr_0_50, tpr_1_00, tpr_2_00, tpr_5_00
    """
    tpr_0_01 = -1
    tpr_0_02 = -1
    tpr_0_05 = -1
    tpr_0_10 = -1
    tpr_0_20 = -1
    tpr_0_50 = -1
    tpr_1_00 = -1
    tpr_2_00 = -1
    tpr_5_00 = -1
    for i in range(len(fpr)):
        if fpr[i] > 0.0001 and tpr_0_01 == -1:
            tpr_0_01 = tpr[i - 1]
        if fpr[i] > 0.0002 and tpr_0_02 == -1:
            tpr_0_02 = tpr[i - 1]
        if fpr[i] > 0.0005 and tpr_0_05 == -1:
            tpr_0_05 = tpr[i - 1]
        if fpr[i] > 0.001 and tpr_0_10 == -1:
            tpr_0_10 = tpr[i - 1]
        if fpr[i] > 0.002 and tpr_0_20 == -1:
            tpr_0_20 = tpr[i - 1]
        if fpr[i] > 0.005 and tpr_0_50 == -1:
            tpr_0_50 = tpr[i - 1]
        if fpr[i] > 0.01 and tpr_1_00 == -1:
            tpr_1_00 = tpr[i - 1]
        if fpr[i] > 0.02 and tpr_2_00 == -1:
            tpr_2_00 = tpr[i - 1]
        if fpr[i] > 0.05 and tpr_5_00 == -1:
            tpr_5_00 = tpr[i - 1]
    return tpr_0_01, tpr_0_02, tpr_0_05, tpr_0_10, tpr_0_20, tpr_0_50, tpr_1_00, tpr_2_00, tpr_5_00


def show_metrics(y_true, y_pred, y_score, miou, mae, logger):
    logger.info(metrics.confusion_matrix(y_true, y_pred))
    # print(metrics.classification_report(y_true, y_pred))
    # Accuracy
    acc = metrics.accuracy_score(y_true, y_pred)
    # AP
    ap = metrics.average_precision_score(y_true, y_score)
    # ROC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=False)
    # print(fpr, tpr)
    fnr = 1 - tpr
    # Equal error rate
    eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # tpr_result = calc_tpr(fpr, tpr)
    roc_auc = metrics.auc(fpr, tpr)
    # metrics_template = "ACC: {:f} AUC: {:f} EER: {:f} TPR@0.01: {:f} TPR@0.10: {:f} TPR@1.00: {:f}"
    # print(metrics_template.format(acc, roc_auc, eer, tpr_result[0], tpr_result[3], tpr_result[6]))
    metrics_template = "ACC: {:f} AP: {:f} AUC: {:f} EER: {:f} mIoU: {:f} MAE: {:f}"
    logger.info(metrics_template.format(acc, ap, roc_auc, eer, miou, mae))
