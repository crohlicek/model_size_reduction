# Effects of Matrix Decompositions in Approaches to Model Size Reduction
Files related to 18.065 (MIT: Matrix Methods in Data Analysis, Signal Processing, and Machine Learning; Professor: Gilbert Strang) term project exploring the impacts of matrix decompositions on neural network size reduction. Implementation of matrix methods and experiments are done with respect to a baseline Lenet-5 architecture applied to the classification task of recognizing the images of handwritten letters found in the EMNIST dataset. Models and related methods implemened in Pytorch.

## Files and Contents:
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
- **fc_pruning.py**:
    - Pruning method to operate on a fully connected layer's matrix of weights
	- Constructor to generate network with fully connected layers pruned to set percentage
	- Training method with data logging (modified to maintain zero-valued weights as in order to simulate the removal of pruned connections)
	- Data collection method that runs sequence of data collection experiments for given list of pruning percentages
- **cp_decomp_conv.py**:
	- Method to take in a convolutional tensor and rank parameter and output a the sequence of reduced-rank tensors found through CP-decomposition
	- Constructor to generate network with convolutional layers' tensors replaced with the tensors found through CP-decomposition
	- Data collection method to automate experiments for networks constructed to a set of specified CP-decomposition ranks
- **fc_svd.py**:
	- Method to get the SVD matrices used in dividing each fully conected layer
	- Constructor to generate network with fully connected layers' weight matrices decomposed into US and V^T matrices through insertion of intermediate layer, for S the diagonal matrix of a reduced number of singular values.
	- Data collection method to automate experiments for networks constructed to a set of specified percentages of remaining singular values
- **18_065_Final_Project_Report.pdf**:
	- Project report summarizing the experiments and theory

