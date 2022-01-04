import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from config import Config
from constants import DEEPER_FORENSICS
from training.datasets.transform import create_train_transform, create_val_test_transform

CONFIG = Config()
ORI_ROOT = CONFIG['ORI_ROOT']


class DeeperForensicsDataset(Dataset):

    def __init__(self, data_root, df: DataFrame, mode, transform: A.Compose):
        self.original_path = os.path.join(ORI_ROOT, 'original_sequences', 'youtube', 'c23')
        self.manipulated_path = os.path.join(data_root, 'manipulated_videos')
        self.df = df
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        video, img_file, label, ori_video, frame = self.df.iloc[index].values
        if label == 1:
            img_path = os.path.join(self.manipulated_path, 'end_to_end_crops', video, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_path = os.path.join(self.manipulated_path, 'end_to_end_diffs', video,
                                     '{}_diff.png'.format(img_file[:-4]))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            img_path = os.path.join(self.original_path, 'crops', video, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # data augmentation
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        mask = mask.unsqueeze(0) / 255.

        return {'images': image, 'labels': label, 'masks': mask}

    def __len__(self):
        r = self.df.shape[0]
        return r


def get_deeper_forensics_dataloader(model_cfg, args):
    train_df = pd.read_csv(f'data/{DEEPER_FORENSICS}/data_{DEEPER_FORENSICS}_end_to_end_train.csv')
    train_transform = create_train_transform(model_cfg)
    train_data = DeeperForensicsDataset(data_root=args.data_dir, df=train_df, mode='train', transform=train_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    val_df = pd.read_csv(f'data/{DEEPER_FORENSICS}/data_{DEEPER_FORENSICS}_end_to_end_val.csv')
    val_transform = create_val_test_transform(model_cfg)
    val_data = DeeperForensicsDataset(data_root=args.data_dir, df=val_df, mode='validation', transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=False)

    return train_sampler, train_loader, val_loader


def get_deeper_forensics_test_dataloader(model_cfg, args):
    test_df = pd.read_csv(f'data/{DEEPER_FORENSICS}/data_{DEEPER_FORENSICS}_end_to_end_test.csv')
    # test_df = test_df.iloc[:57265]
    test_transform = create_val_test_transform(model_cfg)
    test_data = DeeperForensicsDataset(data_root=args.data_dir, df=test_df, mode='test', transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True, drop_last=True)
    return test_loader
