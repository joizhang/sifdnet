import argparse
import json
import os
from glob import glob
from typing import Type

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from preprocessing import face_detector
from preprocessing.face_detector import VideoDataset
from preprocessing.face_detector import VideoFaceDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Process a original videos with face detector")
    parser.add_argument("--root-dir", help="root directory", default="/home/xinlin/data2/Celeb-DF-v2")
    parser.add_argument("--detector-type", default="FacenetDetector", choices=["FacenetDetector"],
                        help="type of the detector")
    args = parser.parse_args()
    return args


def get_original_video_paths(root_dir):
    paths = []
    for video_fold in os.listdir(root_dir):
        if 'real' in video_fold:
            paths.extend(glob(os.path.join(root_dir, video_fold, "*.mp4")))
    return paths


def temp_func(x):
    return x


def process_videos(videos, root_dir, detector_cls: Type[VideoFaceDetector]):
    detector = face_detector.__dict__[detector_cls]()
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=temp_func)
    for item in tqdm(loader):
        video, indices, frames = item[0]
        video_id = os.path.splitext(os.path.basename(video))[0]
        out_dir = os.path.join(root_dir, "boxes")
        output_json = os.path.join(out_dir, "{}.json".format(video_id))
        if os.path.exists(output_json):
            continue
        batches = [frames[i:i + detector.batch_size] for i in range(0, len(frames), detector.batch_size)]
        result = {}
        for j, frames in enumerate(batches):
            result.update({int(j * detector.batch_size) + i: b for i, b in zip(indices, detector.detect_faces(frames))})
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}.json".format(video_id)), "w") as f:
            json.dump(result, f)


def main():
    args = parse_args()
    originals = get_original_video_paths(args.root_dir)
    print(len(originals))
    process_videos(originals, args.root_dir, args.detector_type)


if __name__ == "__main__":
    main()
