import os
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocessing.constants import CELEB_DF
from preprocessing.option import parse_args

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

LIST_OF_TESTING_VIDEOS = 'List_of_testing_videos.txt'


def get_test_videos(root_dir) -> list:
    test_videos = []
    for line in open(os.path.join(root_dir, LIST_OF_TESTING_VIDEOS), 'r'):
        line_split = line.split()
        assert len(line_split) == 2
        test_videos.append(line_split[1])
    return test_videos


def get_real_fake_pairs_for_test(root_dir):
    pairs = []
    test_videos = get_test_videos(root_dir)
    for test_video in test_videos:
        video_fold = Path(test_video).parent.name
        video_path = Path(test_video).name
        if 'real' in video_fold:
            pairs.append((video_path[:-4], video_path[:-4]))
        else:
            video_path_split = video_path.split('_')
            assert len(video_path_split) == 3
            pairs.append(('{}_{}'.format(video_path_split[0], video_path_split[2][:-4]), video_path[:-4]))

    return pairs


def get_real_fake_pairs(root_dir):
    pairs = []
    pairs_for_test = set(get_real_fake_pairs_for_test(root_dir))
    for video_fold in os.listdir(root_dir):
        if 'real' not in video_fold and 'synthesis' not in video_fold:
            continue
        for video_path in os.listdir(os.path.join(root_dir, video_fold)):
            if 'real' in video_fold:
                pair = (video_path[:-4], video_path[:-4])
                if pair not in pairs_for_test:
                    pairs.append(pair)
            else:
                video_path_split = video_path.split('_')
                assert len(video_path_split) == 3
                pair = ('{}_{}'.format(video_path_split[0], video_path_split[2][:-4]), video_path[:-4])
                if pair not in pairs_for_test:
                    pairs.append(pair)
    return pairs


def get_paths(vid, label, root_dir):
    ori_vid, fake_vid = vid
    ori_dir = os.path.join(root_dir, 'crops', ori_vid)
    fake_dir = os.path.join(root_dir, 'crops', fake_vid)
    mask_dir = os.path.join(root_dir, 'diffs', fake_vid)
    data = []
    # Some unaligned videos have been removed
    if not os.path.exists(fake_dir):
        return data
    # Some masks may have been removed
    if label == 1 and not os.path.exists(mask_dir):
        return data
    for frame in os.listdir(fake_dir):
        ori_img_path = os.path.join(ori_dir, frame)
        fake_img_path = os.path.join(fake_dir, frame)
        if label == 0 and os.path.exists(ori_img_path):
            data.append([ori_img_path, label, ori_vid])
        elif label == 1 and os.path.exists(fake_img_path) and os.path.exists(ori_img_path):
            mask_path = os.path.join(mask_dir, "{}_diff.png".format(frame[:-4]))
            if os.path.exists(mask_path):
                data.append([fake_img_path, label, ori_vid])

    return data


def save_folds(args, ori_fake_pairs, mode='train'):
    data = []
    if mode == 'train_fake':
        ori_ori_pairs = set()
    else:
        ori_ori_pairs = set([(ori, ori) for ori, fake in ori_fake_pairs if ori == fake])
    ori_fake_pairs = set(ori_fake_pairs) - ori_ori_pairs
    with Pool(processes=1) as p:
        # original label=0
        with tqdm(total=len(ori_ori_pairs)) as pbar:
            func = partial(get_paths, label=0, root_dir=args.root_dir)
            for v in p.imap_unordered(func, ori_ori_pairs):
                pbar.update()
                data.extend(v)
        # fake label=1
        with tqdm(total=len(ori_fake_pairs)) as pbar:
            func = partial(get_paths, label=1, root_dir=args.root_dir)
            for v in p.imap_unordered(func, ori_fake_pairs):
                pbar.update()
                data.extend(v)
    fold_data = []
    for img_path, label, ori_vid in data:
        path = Path(img_path)
        video = path.parent.name
        file = path.name
        file_split = file.split("_")
        if len(file_split) == 2:
            frame = int(file_split[0])
        else:
            frame = 0
        fold_data.append([video, file, label, ori_vid, frame])
    # random.shuffle(fold_data)
    columns = ["video", "file", "label", "original", "frame"]
    save_file = '../data/{}/data_{}_{}.csv'.format(CELEB_DF, CELEB_DF, mode)
    pd.DataFrame(fold_data, columns=columns).to_csv(save_file, index=False)


def real_fake_split(ori_fake_pairs):
    ori_fake_pairs_real = []
    ori_fake_pairs_fake = []
    for pair in ori_fake_pairs:
        if pair[0] == pair[1]:
            ori_fake_pairs_real.append(pair)
        else:
            ori_fake_pairs_fake.append(pair)
    return ori_fake_pairs_real, ori_fake_pairs_fake


def main():
    args = parse_args()
    os.makedirs('../data/{}'.format(CELEB_DF), exist_ok=True)
    ori_fake_pairs_test = get_real_fake_pairs_for_test(args.root_dir)
    ori_fake_pairs = get_real_fake_pairs(args.root_dir)
    assert set(ori_fake_pairs).isdisjoint(set(ori_fake_pairs_test))
    print(len(ori_fake_pairs))

    def cmp(pair):
        pair_fake_split = pair[1].split('_')
        if len(pair_fake_split) == 1:
            return pair[1]
        elif len(pair_fake_split) == 2:
            return '{}{}'.format(pair_fake_split[0], pair_fake_split[1])
        else:
            return '{}{}'.format(pair_fake_split[0], pair_fake_split[2])

    ori_fake_pairs_test.sort(key=cmp)
    ori_fake_pairs_train, ori_fake_pairs_val = train_test_split(ori_fake_pairs, test_size=0.1, random_state=111)
    ori_fake_pairs_train.append(('flickrface', 'flickrface'))
    save_folds(args, ori_fake_pairs_train, 'train')
    save_folds(args, ori_fake_pairs_val, 'val')
    save_folds(args, ori_fake_pairs_test, 'test')

    ori_fake_pairs_train.sort(key=cmp)
    ori_fake_pairs_real, ori_fake_pairs_fake = real_fake_split(ori_fake_pairs_train)
    save_folds(args, ori_fake_pairs_real, 'train_real')
    save_folds(args, ori_fake_pairs_fake, 'train_fake')


if __name__ == '__main__':
    # Total: 6529
    # Train: 5409
    # Validation: 602
    # Test: 518
    main()
