import os
import re
import time

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from training import models
from training.datasets.transform import create_val_test_transform
from training.tools.metrics import AverageMeter, ProgressMeter, accuracy
from training.tools.train_utils import parse_args


class TestDataset(Dataset):

    def __init__(self, data_root, data, prefix, transform: A.Compose):
        self.root_path = os.path.join(data_root, prefix)
        self.transform = transform
        self.data = data

    def __getitem__(self, index):
        img_file, label = self.data[index]
        img_path = os.path.join(self.root_path, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image = transformed["image"]

        return {'images': image, 'labels': label}

    def __len__(self):
        return len(self.data)


def _prepare_data(args):
    root_path = os.path.join(args.data_dir, args.prefix)
    data = []
    lines = open(os.path.join(root_path, 'real_fake_pairs.dat')).readlines()
    for line in lines:
        line_split = line.split(',')
        data.append((line_split[0], 0))
        data.append((line_split[1].strip(), 1))
    return data


def get_test_dataloader(model_cfg, args):
    file_list = _prepare_data(args)
    test_transform = create_val_test_transform(model_cfg)
    test_data = TestDataset(data_root=args.data_dir, data=file_list, prefix=args.prefix, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True, drop_last=False)
    return test_loader, file_list


def crop_and_resize(mask_output, image, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    size = mask_output.shape[0]
    h, w = image.shape[:2]
    if w > h:
        scale = size / w
        h_scale = h * scale
        # w_scale = size
        start, end = int((size - h_scale) / 2), int((size - h_scale) / 2 + h_scale)
        mask_output = mask_output[start: end, :]
    else:
        scale = size / h
        w_scale = w * scale
        # h_scale = size
        start, end = int((size - w_scale) / 2), int((size - w_scale) / 2 + w_scale)
        mask_output = mask_output[:, start: end]
    interpolation = interpolation_up if scale > 1 else interpolation_down
    mask_output = cv2.resize(mask_output, (int(w), int(h)), interpolation=interpolation)
    return mask_output


def generate_predict_mask(images: torch.Tensor, masks_pred: torch.Tensor, file_list: list, batch_idx, args):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    masks_pred = masks_pred.squeeze().cpu().numpy()
    results_fold_path = os.path.join(args.data_dir, f'{args.prefix}_{args.arch}_results')
    image_fold_path = os.path.join(args.data_dir, args.prefix)
    for i, sample in enumerate(zip(images, masks_pred)):
        image_input, mask_output = sample
        image_data = file_list[batch_idx * args.batch_size + i]
        image_path = os.path.join(image_fold_path, image_data[0])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_output = crop_and_resize(mask_output, image)
        # image_input = image_input * std * 255.0
        # image_input = image_input + mean * 255.0
        # image = Image.fromarray(image_input.astype(np.uint8))
        # image_path = os.path.join(results_fold_path, image_data['file'])
        # image.save(image_path)
        mask_output[mask_output >= 0.2] = 1.
        mask = Image.fromarray((mask_output * 255).astype(np.uint8))
        mask_path = os.path.join(results_fold_path, '{}_pred.png'.format(image_data[0][:-4]))
        mask.save(mask_path)


def predict(test_loader, file_list: list, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    pw_acc = AverageMeter('Pixel-wise Acc', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, top1, pw_acc], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, sample in enumerate(test_loader):
            images, labels = sample['images'].cuda(), sample['labels'].cuda()

            # compute output
            outputs = model(images)
            if isinstance(outputs, tuple):
                if len(outputs) == 4:
                    labels_pred, masks1, masks2, masks_pred = outputs
                    # masks_pred = torch.argmax(masks_pred, dim=1)
                elif len(outputs) == 2:
                    labels_pred, masks_pred = outputs
            else:
                labels_pred = outputs

            # measure accuracy and record loss
            acc1, = accuracy(labels_pred, labels)
            top1.update(acc1[0], images.size(0))
            # if isinstance(outputs, tuple):
            #     overall_acc = eval_metrics(masks.cpu(), masks_pred.cpu(), 256)
            #     pw_acc.update(overall_acc, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % args.print_freq == 0:
                progress.display(batch_idx + 1)

            generate_predict_mask(images, masks_pred, file_list, batch_idx, args)


def main():
    args = parse_args()
    print(args)
    os.makedirs(os.path.join(args.data_dir, f'{args.prefix}_{args.arch}_results'), exist_ok=True)
    if args.resume:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        print("Loading checkpoint '{}'".format(args.resume))
        model = models.__dict__[args.arch](pretrained=False)
        model.cuda()
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)

        print("Initializing Data Loader")
        test_loader, file_list = get_test_dataloader(model.default_cfg, args)

        print("Starting generation")
        predict(test_loader, file_list, model, args)


if __name__ == '__main__':
    """
    PYTHONPATH=. python temp/predict.py --data-dir G:\\Celeb-DF-v2 --arch deeplab_v3_plus --workers 1 \
    --prefix celeb-df --batch-size 10 --print-freq 1 --resume weights/deeplab_v3_plus_celeb-df_2.pt
    """
    main()
