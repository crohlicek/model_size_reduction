# Effects of Matrix Decompositions in Approaches to Model Size Reduction
Files related to 18.065 (MIT: Matrix Methods in Data Analysis, Signal Processing, and Machine Learning; Professor: Gilbert Strang) term project exploring the impacts of matrix decompositions on neural network size reduction. Implementation of matrix methods and experiments are done with respect to a baseline Lenet-5 architecture applied to the classification task of recognizing the images of handwritten letters found in the EMNIST dataset. Models and related methods implemened in Pytorch.

## Files:
- **emnist_baseline.py**: 
    - Implentation of baseline model constructor
    - Training methods (with and withtout data logging)
    - Testing method
    - Method to import and load EMNIST data.
- **conv_pruning.py**: 
    - Pruning method to operate on a convolution layer's tensor of weights
    - Constructor to generate network with convolutional layers pruned to set percentage
    - Training method with data logging (modified to maintain zero-valued weights as in order to simulate the removal of pruned connections)
    - Data collection method that runs sequence of data collection experiments for given list of pruning percentages
