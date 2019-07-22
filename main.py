"""
Driver program which fetches information from command line. Can be used to
run a sample demo as well.
Arguments:
    save_dir: Optional, Denotes the location of the saving dir. All figures
    		  will be saved in the specified dir. Default value is './figures/'
	run_demo_with_faces: Optional, denotes whether to run the face demo upon
						 invocation of the script.
"""

import argparse
import logging
import os
import sklearn

import visualizer as vs

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA


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
    '--should_save_fig', 
    type=str2bool,
    default=True,
    help="Optional. Specifies whether to save the figures or show it on display")
parser.add_argument(
    '--run_demo_with_faces', 
    type=str2bool,
    default=True,
    help="Optional. Specifies whether to run the faces plotting demo")


def run_olivetti_faces_demo():
	# Get the face data images
	face_imgs = fetch_olivetti_faces().data
	# Denotes the resized shape if the features are considered as images
	resize_shape = (64, 64)

	# We are only operating one class in order to get better visualizations
	# Note that the first 10 faces correspond to the same person. Making it
	# easier to capture all variance in fewer components
	face_imgs = face_imgs[:10]

	# Fit the face samples with two components
	pca = PCA(n_components=2, svd_solver='randomized')
	pca.fit(face_imgs)

	vs.plot_pca_components_as_img(pca, resize_shape)

	vs.plot_pca_feature_dist(pca, face_imgs)

	vs.gen_and_plot_imgs_from_pca_coords(pca, resize_shape, face_imgs)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # Load the parameters passed
    args = parser.parse_args()

    # Set the global constants
    vs.FIG_SAVE_DIR = args.save_dir
    vs.SHOULD_SAVE_FIG = args.should_save_fig

    # Create the dir if not exists
    if not os.path.exists(vs.FIG_SAVE_DIR):
	    os.makedirs(vs.FIG_SAVE_DIR)

    logging.info("Arguments passed are: %s", args)
    
    # Execute the demo in case the args is passed
    if args.run_demo_with_faces:
    	run_olivetti_faces_demo()
