"""
Drive 
"""

import argparse
import logging

from sklearn.datasets import fetch_olivetti_faces


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_dir', 
    type=str,
    default="./figures/",
    help="Optional. Denotes the location of the saving dir")
parser.add_argument(
    '--run_demo_with_faces', 
    type=str2bool,
    default=True,
    help="Optional. Specifies whether to run the faces plotting demo")


def run_olivetti_faces_demo():
	# Get the face data images
	face_imgs = fetch_olivetti_faces().data



if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)

    # Load the parameters passed
    args = parser.parse_args()

    logging.info("Arguments passed are: %s", args)
    