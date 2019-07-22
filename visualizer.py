import itertools
import logging
import os
import sklearn

import matplotlib
# In order to handle the OSX backend issue
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

# Constants and params for the file
IMG_DTYPE = 'img'
HIST_DTYPE = 'hist'

SHOULD_SAVE_FIG = True
FIG_SAVE_DIR = "./figures/"


def plot_subplots(objs, main_title, plot_titles, path, dtype=IMG_DTYPE):
    '''
        Plots objs (could represent imgs or values) as subplots.
        Args:
            objs: Numpy array of size (num_instance, h, w [, 1/3]) in
                  case of images. Otherwise numpy array of size 
                  (num_instance, v).
            plot_titles: List of plot titles for the objs. The length
                        must be the same as the number of objs passed
            path: Path to save the resultant image
            dtype: Represents whether to plot image or histogram. Legal
                   values: IMG_DTYPE, HIST_DTYPE
    '''

    assert \
        objs.shape[0] == len(plot_titles), \
        "Objs shape and plot titles do not match. Objs: {}, Titles: {}".format(objs.shape, len(plot_titles))

    # Cosntruct the grid lengths
    nrows = round(len(plot_titles) ** (0.5))
    ncols = round(len(plot_titles) / nrows)

    fig, axs = plt.subplots(nrows, ncols, squeeze=False)
    fig.subplots_adjust(hspace=.5)
    fig.suptitle(main_title)

    for i in range(objs.shape[0]):

        # Get the coordinates
        ni, nj = i // ncols, i % ncols
        ax = axs[ni][nj]

        if dtype == IMG_DTYPE:
            ax.imshow(objs[i], cmap=plt.cm.bone, interpolation='nearest')
        elif dtype == HIST_DTYPE:
            ax.hist(objs[i])
        else:
            logging.error("Invalid dtype provided for plotting")
            return

        ax.set_title(plot_titles[i])
        ax.grid()

    if SHOULD_SAVE_FIG:
        # Save figure to the specified path
        plt.savefig(path)
    else:
        # Plot the plots otherwise
        plt.show()


def plot_pca_components_as_img(pca, resize_shape, filename='pca_components.png'):
    '''
        Plots pca components as images of specified size.
        Consider a face dataset of images with size 64x64. PCA operates
        on a flattened representation. Before plotting we resize the
        flattened rep to an image of the specified shape.
        Args:
            pca: The sklearn PCA object which has been fit with the dataset
            resize_shape: Tuple representing the shape of the img. Note that
                          it should either be (h, w), (h, w, 1) or (h, w, 3)
                          in order to allow ax.imshow() to plot it.
    '''

    logging.info("Plotting the pca components...")

    # Add the mean to the components in order to plot
    components = np.concatenate((pca.mean_.reshape((1, -1)), pca.components_), axis=0)
    component_objs = components.reshape((-1, *resize_shape))
    # Generate the corresponding titles of the subplots
    plot_titles = ['mean'] + ['comp_' + str(i) for i in range(len(pca.components_))]

    # Plot and save the resultant images
    plot_subplots(
        component_objs, 'pca_components', plot_titles, path=os.path.join(FIG_SAVE_DIR, filename), dtype=IMG_DTYPE
    )

    logging.info("Finsihed plotting the pca components.")


def plot_pca_feature_dist(pca, x, filename='pca_feat_dist.png'):
    '''
        Plots distribution of the pca features as histograms.
        Args:
            pca: The sklearn PCA object which has been fit with the dataset
            x: The data on which to compute the features. Shape: (n, f)
            filename: Destination to save the resultant plot
    '''

    logging.info("Plotting the distribution of the pca components...")

    fets = np.transpose(pca.transform(x))
    plot_titles = ['fet_' + str(i) for i in range(fets.shape[0])]

    # Plot and save the resultant images
    plot_subplots(
        fets, 'pca_features', plot_titles, path=os.path.join(FIG_SAVE_DIR, filename), dtype=HIST_DTYPE
    )

    logging.info("Finished plotting the distribution.")


def gen_and_plot_imgs_from_pca_coords(pca, resize_shape, x=None, num_comps_to_plot=1, num_steps=3, filename='gen_pca_imgs.png'):
    '''
        Generates and reconstructs features from a compressed representation, then plots 
        the resultant images with the specified params.
        Args:
            pca: The sklearn PCA object which has been fit with the dataset
            resize_shape: Tuple representing the shape of the img. Note that it should 
                          either be (h, w), (h, w, 1) or (h, w, 3) in order to allow 
                          ax.imshow() to plot it.
            x: Sample of the dataset in order to infer the ideal ranges to plot
            num_comps_to_plot: The number of components varied in order to generate the
                               new images
            num_steps: Number of datapoints in each sample
            filename: The file to save the final result in
    '''

    logging.info("Generating and plotting images...")

    num_pca_comp = pca.components_.shape[0]

    # Verifications for the arguments
    assert \
        num_comps_to_plot <= num_pca_comp, \
        "Number of components to plot is larger than actual number of components: {}, {}".format(num_comps_to_plot, num_pca_comp)

    if num_comps_to_plot ** num_steps > 200:
        logging.error(
            "Not proceeding forward since the specified arguments would lead to a large number of subplots. " \
            "Please remove this line from the code manually to proceed after checking the argument values." \
            "num_comps_to_plot: {}, num_steps: {}".format(num_comps_to_plot, num_steps)
        )
        return

    if num_comps_to_plot ** num_steps > 50:
        logging.warning(
            "Note that the specified arguments would lead to a large number of subplots." \
            "num_comps_to_plot: {}, num_steps: {}".format(num_comps_to_plot, num_steps)
        )

    # Infer the range to vary the components if possible
    if x is None:
        # Default range in case we can't infer the range
        ls, rs = [-5] * num_pca_comp, [-5] * num_pca_comp
    else:
        fets = pca.transform(x)
        ls, rs = np.amin(fets, axis=0), np.amax(fets, axis=0)
        # Scale the ranges in order to get more extreme results
        scale = 1.5
        ls, rs = ls * scale, rs * scale

    # Generate the coordinates for each component to be varied
    coords = [np.arange(ls[i], rs[i], (rs[i] - ls[i]) / num_steps) for i in range(num_comps_to_plot)]
    # Assume the coordinates for the rest to be 0
    coords = coords + [[0.0] for i in range(num_comps_to_plot, num_pca_comp)]

    # Generate all permutations of all the coordinates
    all_coords = np.array(list(itertools.product(*coords)))
    # Generate plot titles, compressing the coords to a string
    plot_titles = \
        ['gen_img_' + np.array2string(all_coords[i], formatter={'all': lambda x: "%.1f" % x}) for i in range(all_coords.shape[0])]

    # Get the inverse generated features. This will be resized to the desired image
    gen_inv_imgs = pca.inverse_transform(all_coords.reshape((-1, num_pca_comp))).reshape((-1, *resize_shape))

    logging.info("Generated the images from our representation. Now plotting the images...")

    # Plot and save the resultant images
    plot_subplots(
        gen_inv_imgs, 'generated_pca_imgs', plot_titles, path=os.path.join(FIG_SAVE_DIR, filename), dtype=IMG_DTYPE
    )

    logging.info("Finished plotting the images.")
