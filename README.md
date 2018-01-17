# deep-margins
A collection of scripts for investigating decision margins in neural networks.
* input_pipe.py: We read in binary data (cat and dog images) using the new Tensorflow Dataset api
* crop.py: Helper script to crop images. Going to be included in input_pipe.py eventually
* closest_pair.py: A script to find the pair of images with the least euclidean distance seperating them. However, this script is inefficient with large (5k+) datasets, and needs to be rewritten using CUDA
* conv_net.py: Simple convolutional network model and training for testing.
* lin_reg.py: For testing.
