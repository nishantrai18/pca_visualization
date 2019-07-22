# Utilities for PCA Visualization

This repo contains helper functions and demos in order to generate visualizations for PCA. Specifically, it adds helpers in order to generate new sample from reconstructing instances from the compressed representation.

## Getting Started

The scripts `main.py` and `visualizer.py` contain all the logic of the program. main.py is the driver program responsible for taking cmd arguments and running the demo if required. `visualization.py` contain helper functions and is the brains behind the repo.

Use the following commands to get started,

	> python main.py --save_dir ./figures/ --run_demo_with_faces True --should_save_fig True

This would run the demo instance which works on a face dataset. It saves the resulting images in the specified directory.

## Requirements

- Python3
- numpy
- scikit-learn
- matplotlib
