import os
from functools import partial
from multiprocessing.pool import Pool

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm

from preprocessing.option import parse_args

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_original_with_fakes(root_dir):
    pairs = []
    for video_fold in os.listdir(root_dir):
        if 'synthesis' not in video_fold:
            continue
        for video_path in os.listdir(os.path.join(root_dir, video_fold)):
            video_path_split = video_path.split('_')
            assert len(video_path_split) == 3
            pairs.append(('{}_{}'.format(video_path_split[0], video_path_split[2][:-4]), video_path[:-4]))
    return pairs


def save_diffs(pair, root_dir):
    ori_id, fake_id = pair
    ori_dir = os.path.join(root_dir, "crops", ori_id)
    fake_dir = os.path.join(root_dir, "crops", fake_id)
    diff_dir = os.path.join(root_dir, "diffs", fake_id)

    if os.path.exists(diff_dir):
        return
    os.makedirs(diff_dir, exist_ok=True)

    for frame in os.listdir(ori_dir):
        image_id = frame[:-4]
        diff_image_id = "{}_diff.png".format(image_id)
        ori_path = os.path.join(ori_dir, frame)
        fake_path = os.path.join(fake_dir, frame)
        diff_path = os.path.join(diff_dir, diff_image_id)

        if os.path.exists(fake_path):
            img1 = cv2.imread(ori_path, cv2.IMREAD_COLOR)
            img2 = cv2.imread(fake_path, cv2.IMREAD_COLOR)
            try:
                d, a = structural_similarity(img1, img2, multichannel=True, full=True)
                a = 1 - a
                diff = (a * 255).astype(np.uint8)
                diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(diff_path, diff)
            except:
                pass


def main():
    args = parse_args()
    pairs = get_original_with_fakes(args.root_dir)
    os.makedirs(os.path.join(args.root_dir, "diffs"), exist_ok=True)
    with Pool(processes=1) as p:
        with tqdm(total=len(pairs)) as pbar:
            func = partial(save_diffs, root_dir=args.root_dir)
            for v in p.imap_unordered(func, pairs):
                pbar.update()


if __name__ == '__main__':
    main()
