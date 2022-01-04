import math
import os
import random

import albumentations as A
import cv2
import dlib
import numpy as np
import skimage
from albumentations import DualTransform
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation
from skimage import measure, draw

from config import BASE_DIR

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(BASE_DIR, 'libs', 'shape_predictor_68_face_landmarks.dat'))


def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks


def blackout_convex_hull(img, mask):
    try:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1

        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)
        first = random.random() > 0.5
        if random.random() > 0.5:
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        else:
            if first:
                cropped_img[:, :x] = 0
            else:
                cropped_img[:, x:] = 0

        img[cropped_img > 0] = 0
        mask[cropped_img > 0] = 0
    except Exception as e:
        pass


def drop_background(img, mask):
    try:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1
        img[cropped_img == 0] = 0
        mask[cropped_img == 0] = 0
    except Exception as e:
        pass


def blend_back(img_ori, img, mask_ori, mask):
    img_ori[img > 50] = img[img > 50]
    mask_ori[mask > 0] = mask[mask > 0]
    return img_ori, mask_ori


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_eyes(image, mask, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    shadow = np.zeros_like(image[..., 0])
    line = cv2.line(shadow, (x1, y1), (x2, y2), color=1, thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    mask[line] = 0
    return image, mask


def remove_nose(image, mask, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    shadow = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(shadow, (x3, y3), (x4, y4), color=1, thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    mask[line] = 0
    return image, mask


def remove_mouth(image, mask, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    shadow = np.zeros_like(image[..., 0])
    line = cv2.line(shadow, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    mask[line] = 0
    return image, mask


def remove_background(image, mask, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    shadow = np.zeros_like(image[..., 0])
    line = cv2.line(shadow, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    mask[line] = 0
    return image, mask


def remove_landmark(image, mask, landmarks):
    if random.random() > 0.5:
        image, mask = remove_eyes(image, mask, landmarks)
    elif random.random() > 0.5:
        image, mask = remove_mouth(image, mask, landmarks)
    elif random.random() > 0.5:
        image, mask = remove_nose(image, mask, landmarks)
    return image, mask


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):

    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=self.interpolation_down,
                                          interpolation_up=self.interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return "max_side", "interpolation_down", "interpolation_up"

    def apply_to_bbox(self, bbox, **params):
        pass

    def apply_to_keypoint(self, keypoint, **params):
        pass

    def get_params_dependent_on_targets(self, params):
        pass


def generalization_preprocessing(landmark_path, image, label, mask, generalization_transform=None):
    # if os.path.exists(landmark_path) and random.random() < 0.3:
    #     landmarks = np.load(landmark_path)
    #     image, mask = remove_landmark(image, mask, landmarks)
    # elif random.random() < 0.3:
    #     blackout_convex_hull(image, mask)
    if random.random() < 0.3:
        bitmap_masks = prepare_bit_masks(mask)
        bitmap_mask = random.choice(bitmap_masks)
        image = np.multiply(image, np.expand_dims(bitmap_mask, axis=2))
        mask = np.multiply(mask, bitmap_mask)

    # elif generalization_transform is not None and label == 1 and random.random() < 0.5:
    #     image_tmp, mask_tmp = np.copy(image), np.copy(mask)
    #     drop_background(image, mask)
    #     g_transformed = generalization_transform(image=image, mask=mask)
    #     image = g_transformed["image"]
    #     mask = g_transformed["mask"]
    #     if random.random() < 0.5:
    #         image, mask = blend_back(image_tmp, image, mask_tmp, mask)
    return image, mask


def create_generalization_transform():
    return A.Compose([
        A.Blur(blur_limit=(5, 10), p=0.7),
        A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
        A.OpticalDistortion(distort_limit=(1., 2.), border_mode=cv2.BORDER_CONSTANT, p=0.5)
    ])


def create_train_transform(model_cfg):
    size = model_cfg['input_size'][1]
    mean = model_cfg['mean']
    std = model_cfg['std']
    return A.Compose([
        A.OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def create_val_test_transform(model_cfg):
    size = model_cfg['input_size'][1]
    mean = model_cfg['mean']
    std = model_cfg['std']
    return A.Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
