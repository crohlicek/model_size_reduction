import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

from emnist_baseline import get_data_loaders, training_with_data_collection, get_test_accuracy, Net

# Define Hyper-parameters 
num_classes = 26
num_epochs = 100
batch_size = 100
learning_rate = 0.001
patience = 10

# from https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
# implementation from paper https://arxiv.org/pdf/1412.6553.pdf
def cp_decomposition_conv_layer(layer, rank):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensorly.
        #parafac returns a tuple of (weights, factors)
    weights, factors = parafac(tensor=tl.tensor(layer.weight.data), rank=rank, init='svd')

    #parafac spits out numpy arrays but the later transpose methods want tensors:
        #will use torch.tensor to cast them to tensors
    last_array, first_array, vertical_array, horizontal_array = factors
    last = torch.tensor(last_array)
    first = torch.tensor(first_array)
    vertical = torch.tensor(vertical_array)
    horizontal = torch.tensor(horizontal_array)


    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, 
            dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1], 
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
            groups=vertical.shape[1], bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1], \
            out_channels=horizontal.shape[1], 
            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
            padding=(0, layer.padding[0]), 
            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
            out_channels=last.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=True)

    pointwise_r_to_t_layer.bias.data = layer.bias.data

    depthwise_horizontal_layer.weight.data = \
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]
    
    return nn.Sequential(*new_layers)


#METHOD THAT TAKES IN LENET 5 AND RAKN, RETURNS CP'd LENET5

def generate_conv_cp_network(net, rank):
    #GET NEW LAYERS BY BREAKING UP THE ORIGINAL TWO CONV LAYERS
    c1, c2, c3, c4 = cp_decomposition_conv_layer(net.conv1, rank)
    c5, c6, c7, c8 = cp_decomposition_conv_layer(net.conv2, rank)
    
    
    #NOW TO PLUG THESE INTO A NEW NETWORK:
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 3x3 square convolution
            # kernel
            self.conv1 = c1
            self.conv2 = c2
            self.conv3 = c3
            self.conv4 = c4
            self.conv5 = c5
            self.conv6 = c6
            self.conv7 = c7
            self.conv8 = c8

            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 26)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv4(self.conv3(self.conv2(self.conv1(x))))), (2, 2))
            # If the size is a square you can only specify a single number
            x = F.max_pool2d(F.relu(self.conv8(self.conv7(self.conv6(self.conv5(x))))), (2, 2))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    #generate new lenet5 to work out this example
    cp_method_example = Net()
    print(cp_method_example)

    return cp_method_example


# method to iterate through a list of cp-decomposition ranks and return the data
    # input: original lenet5, list of ranks
def collect_cp_conv_data(net, r_list):
    #dict to output experiment data
        # keys will be the percentages and values will be lists of output logs values
    data_dict = {}

    #get dataloaders for each network
    data_loaders, train_loader, val_loader, test_loader = get_data_loaders()

    for rank in r_list:
        #ordered contents of output list will be:
            # number of parameters
            # average training loss log - a
            # average validation loss log - b
            # average training accuracy log - c
            # average validation accuracy log - d
            # stopping epoch - e
            # test accuracy

        output_list = []
        cp_decomp_net = generate_conv_cp_network(net, rank)

        #add number of parameters to the list
        output_list.append(sum([param.nelement() for param in cp_decomp_net.parameters()]))

        #call to the training method:
        a,b,c,d,e = training_with_data_collection(net=cp_decomp_net,
                                                data_loaders=data_loaders,
                                                num_epochs=num_epochs,
                                                patience=patience)
        output_list.append(a)
        output_list.append(b)
        output_list.append(c)
        output_list.append(d)
        output_list.append(e)

        #get test accuracy
        test_acc = get_test_accuracy(cp_decomp_net, test_loader)
        output_list.append(test_acc)

        #assign this output list to the value for this entry of the data dict
        data_dict[rank] = output_list

    return data_dict

def main():
	net = Net()
	print(net)
	#here the preloaded model can be loaded from path

	#data object to be saved to flie
	experiment_data = collect_cp_conv_data(net, [1, 5, 10, 15])



if __name__ == "__main__":
    main()    





