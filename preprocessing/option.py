import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters of preprocessing and statistical program")
    parser.add_argument("--root-dir", default="", help="Root directory")
    parser.add_argument("--detector-type", default="FacenetDetector", choices=["FacenetDetector"],
                        help="type of the detector")
    parser.add_argument("--crops-dir", default="crops", help="crops directory")
    parser.add_argument("--save", default=False, help="Save to file")
    parser.add_argument("--folds-csv", type=str, default="folds.csv")
    parser.add_argument("--fake-type", type=str, default="Deepfakes", help="Manipulation type")
    args = parser.parse_args()
    return args
