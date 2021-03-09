import argparse
import os
from functools import partial
from multiprocessing.pool import Pool

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from tqdm import tqdm

from preprocessing.detect_original_faces import get_original_video_paths

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cuda:0")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract image landmarks")
    parser.add_argument("--root-dir", help="root directory", default="")
    args = parser.parse_args()
    return args


def save_landmarks(ori_path, root_dir):
    ori_id = os.path.basename(ori_path)[:-4]
    ori_dir = os.path.join(root_dir, "crops", ori_id)
    landmark_dir = os.path.join(root_dir, "landmarks", ori_id)
    if os.path.exists(landmark_dir):
        return
    os.makedirs(landmark_dir, exist_ok=True)

    for frame in os.listdir(ori_dir):
        image_id = frame[:-4]
        landmarks_id = image_id
        ori_path = os.path.join(ori_dir, frame)
        landmark_path = os.path.join(landmark_dir, landmarks_id)

        if os.path.exists(ori_path):
            try:
                image_ori = cv2.imread(ori_path, cv2.IMREAD_COLOR)[..., ::-1]
                frame_img = Image.fromarray(image_ori)
                batch_boxes, conf, landmarks = detector.detect(frame_img, landmarks=True)
                if landmarks is not None:
                    landmarks = np.around(landmarks[0]).astype(np.int16)
                    np.save(landmark_path, landmarks)
            except Exception as e:
                print(e)
                pass


def main():
    args = parse_args()
    originals = get_original_video_paths(args.root_dir)
    print(len(originals))
    os.makedirs(os.path.join(args.root_dir, "landmarks"), exist_ok=True)
    with Pool(processes=1) as p:
        with tqdm(total=len(originals)) as pbar:
            func = partial(save_landmarks, root_dir=args.root_dir)
            for v in p.imap_unordered(func, originals):
                pbar.update()


if __name__ == '__main__':
    main()
