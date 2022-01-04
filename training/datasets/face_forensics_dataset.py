import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from constants import FACE_FORENSICS
from training.datasets.transform import create_generalization_transform, generalization_preprocessing
from training.datasets.transform import create_train_transform, create_val_test_transform


class FaceForensicsDataset(Dataset):

    def __init__(self, data_root, df: DataFrame, mode: str, transform: A.Compose, fake_type: str,
                 compression_version: str, use_generalization=False, label_smoothing=0.01):
        """
        Args:
            fake_type(str): Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, All
        """
        self.original_path = os.path.join(data_root, 'original_sequences', 'youtube', compression_version)
        self.manipulated_path = os.path.join(data_root, 'manipulated_sequences')
        self.df = df
        self.mode = mode
        self.transform = transform
        self.generalization_transform = create_generalization_transform()
        self.fake_type = fake_type
        self.compression_version = compression_version
        self.use_generalization = use_generalization
        self.label_smoothing = label_smoothing

    def __getitem__(self, index):
        if self.fake_type == 'All':
            video, img_file, label, ori_video, frame, fake_type = self.df.iloc[index].values
            fake_type_path = os.path.join(self.manipulated_path, fake_type)
        else:
            video, img_file, label, ori_video, frame = self.df.iloc[index].values
            fake_type_path = os.path.join(self.manipulated_path, self.fake_type)

        if label == 1:
            # image
            img_path = os.path.join(fake_type_path, self.compression_version, 'crops', video, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # mask
            mask_path = os.path.join(fake_type_path, 'c23', 'diffs', video, f'{img_file[:-4]}_diff.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            img_path = os.path.join(self.original_path, 'crops', video, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.mode == 'train' and self.use_generalization:
            landmark_path = os.path.join(self.original_path, "landmarks", str(ori_video), img_file[:-4] + ".npy")
            image, mask = generalization_preprocessing(landmark_path, image, label, mask, self.generalization_transform)

        # data augmentation
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        mask = mask.unsqueeze(0) / 255.

        return {'images': image, 'labels': label, 'masks': mask}

    def __len__(self):
        r = self.df.shape[0]
        return r


def get_face_forensics_dataloader(model_cfg, args, fake_type="Deepfakes"):
    train_df = pd.read_csv(f'data/{FACE_FORENSICS}/{args.compression_version}/data_{FACE_FORENSICS}_{fake_type}_train.csv')
    # row_num = train_df.shape[0]
    # start_index = (row_num - 9600) // 2
    # train_df = train_df.iloc[start_index:start_index + 9600]
    # train_df = train_df.iloc[-1000:]
    train_transform = create_train_transform(model_cfg)
    train_data = FaceForensicsDataset(data_root=args.data_dir, df=train_df, mode='train', transform=train_transform,
                                      fake_type=fake_type, compression_version=args.compression_version,
                                      use_generalization=False)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    val_df = pd.read_csv(f'data/{FACE_FORENSICS}/{args.compression_version}/data_{FACE_FORENSICS}_{fake_type}_val.csv')
    # val_df = val_df.iloc[-1000:]
    val_transform = create_val_test_transform(model_cfg)
    val_data = FaceForensicsDataset(data_root=args.data_dir, df=val_df, mode='validation', transform=val_transform,
                                    fake_type=fake_type, compression_version=args.compression_version)
    val_loader = DataLoader(val_data, batch_size=30, shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=False)

    return train_sampler, train_loader, val_loader


def get_face_forensics_test_dataloader(model_cfg, args, fake_type="Deepfakes"):
    test_df = pd.read_csv(f'data/{FACE_FORENSICS}/{args.compression_version}/data_{FACE_FORENSICS}_{fake_type}_test.csv')
    # test_df = test_df.iloc[-1000:]
    test_transform = create_val_test_transform(model_cfg)
    test_data = FaceForensicsDataset(data_root=args.data_dir, df=test_df, mode='test', transform=test_transform,
                                     fake_type=fake_type, compression_version=args.compression_version)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True, drop_last=False)
    return test_loader
