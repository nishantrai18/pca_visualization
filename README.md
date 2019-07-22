# Utilities for PCA Visualization

This repo contains helper functions and demos in order to generate visualizations for PCA. Specifically, it adds helpers in order to generate new sample from reconstructing instances from the compressed representation.

The functions present in the scripts can be used as utilities to power your own analysis and visualizations.

## Getting Started

The scripts `main.py` and `visualizer.py` contain all the logic of the program. main.py is the driver program responsible for taking cmd arguments and running the demo if required. `visualization.py` contain helper functions and is the brains behind the repo.

Use the following commands to get started,

	> python main.py --save_dir ./figures/ --run_demo_with_faces True --should_save_fig True

This would run the demo instance which works on a face dataset. It saves the resulting images in the specified directory.

## Results

We use the Olivetti face dataset in order to power are demonstration. This section gives an overview of the section and the resultant figures.

We first fetch the data from sklearn.datasets. This dataset consists of 40 people with 10 instances of each of their faces.
```
from sklearn.datasets import fetch_olivetti_faces

# Get the face data images
face_imgs = fetch_olivetti_faces().data

# Denotes the resized shape if the features are considered as images
resize_shape = (64, 64)
```

Note that `resize_shape` would be **different** based on your setup. This is used to render the flattened representation to an image later on.

We then proceed to perform PCA on a subset of the data. We only choose one class in order to get better visualizations.

```
from sklearn.decomposition import PCA

# We are only operating one class in order to get better visualizations
# Note that the first 10 faces correspond to the same person. Making it
# easier to capture all variance in fewer components
face_imgs = face_imgs[:10]

# Fit the face samples with two components
pca = PCA(n_components=3, svd_solver='randomized')
pca.fit(face_imgs)
```

Now that we've performed PCA, we can proceed to visualize the results. We call some utility functions defined in `visualization.py`

```
vs.plot_pca_components_as_img(pca, resize_shape)
```

This plots the PCA components, the result is as shown below,
![alt text][pca_comps]

We can also plot the distribution of the compressed features.

```
vs.plot_pca_feature_dist(pca, face_imgs)
```

In our case, this gives us,
![alt text][pca_fets]

Finally, we can generate **NEW** instances by sampling values in the compressed feature space and reconstructing it. We have a utility which encapsulates all this logic.

```
vs.gen_and_plot_imgs_from_pca_coords(pca, resize_shape, face_imgs, num_comps_to_plot=2)
```

This results in,
![alt text][pca_gen_imgs]

More details can be found in the comments present in the code.

## Driver script

`main.py` acts as the driver script and supports multiple flags to aid usage. A rough description of the flags is given below,

- **save_dir**:  Optional. Denotes the location where we save the resultant figures
- **should_save_fig**: Optional. Specifies whether to save the figures or show it on display. Setting this to false opens the plots instead of saving them.
- **run_demo_with_faces**: Optional. Specifies whether to run the faces plotting demo. If false, then you can choose to write your own script and put it into main to execute.

## Requirements

- Python3
- numpy
- scikit-learn
- matplotlib

[pca_fets]:  https://github.com/nishantrai18/pca_visualization/blob/master/figures/pca_feat_dist.png "PCA feature distribution"
[pca_comps]:  https://github.com/nishantrai18/pca_visualization/blob/master/figures/pca_components.png "PCA components"
[pca_gen_imgs]:  https://github.com/nishantrai18/pca_visualization/blob/master/figures/gen_pca_imgs.png "Generated images"
