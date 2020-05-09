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

#METHOD TO SOLVE FOR VT AND US MATRICES FOR THREE FC LAYERS
    #takes as input, the three FC weight matrices
def get_svd_weights(m1, #net.fc1.weight.data
                    m2, #net.fc2.weight.data
                    m3, #net.fc3.weight.data
                    percent): #percent of the SVs to keep
    
    #decomposes the three input weight matrics
    u_1,s_1,v_1 = m1.svd()
    u_2,s_2,v_2 = m2.svd()
    u_3,s_3,v_3 = m3.svd()

    #get the number of SVs to keep in each layer:
    keep1 = int(percent*len(s_1))
    keep2 = int(percent*len(s_2))
    keep3 = int(percent*len(s_3))

    #cut trim s for each layer
    s1_new = torch.diag(s_1[:keep1])
    s2_new = torch.diag(s_2[:keep2])
    s3_new = torch.diag(s_3[:keep3])

    #readjust dimensions of U and V to match the reduced size of S
    u1_reduced = u_1[:,:keep1]
    v1_reduced = v_1[:,:keep1]
    u2_reduced = u_2[:,:keep2]
    v2_reduced = v_2[:,:keep2]
    u3_reduced = u_3[:,:keep3]
    v3_reduced = v_3[:,:keep3]

    # define the matrices of weights for the new fc layers:
    us1_red = torch.mm(u1_reduced, s1_new)
    vt1_red = torch.transpose(v1_reduced, 0, 1)
    us2_red = torch.mm(u2_reduced, s2_new)
    vt2_red = torch.transpose(v2_reduced, 0, 1)
    us3_red = torch.mm(u3_reduced, s3_new)
    vt3_red = torch.transpose(v3_reduced, 0, 1)

    #RETURNS THE SVD MATRICES IN TUPLES CORRESPONDING TO LAYER
        #ALONG WITH THE INTERMEDIATE LAYER DIMENSION FOR THE CONSTRUCTION OF THE REDUCED NETWORK
    return (us1_red, vt1_red, keep1), (us2_red, vt2_red, keep2), (us3_red, vt3_red, keep3)

#METHOD THAT TAKES IN LENET 5 AND PERCENT, RETURNS SVD'd LENET5
def generate_fc_svd_network(net, percent):
    a, b, c = get_svd_weights(net.fc1.weight.data,
                                net.fc2.weight.data,
                                net.fc3.weight.data,
                                percent)
    
    (us1_red, vt1_red, keep1) = a
    (us2_red, vt2_red, keep2) = b
    (us3_red, vt3_red, keep3) = c
    
    
    #NOW TO PLUG THESE INTO A NEW NETWORK:
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 3x3 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            # an affine operation: y = Wx + b

            #FC1 IS DECOMPOSED INTO VT1 AND US1 -- KEEP PARAMETER GOES HERE 
            self.fc1_vt = nn.Linear(16 * 6 * 6, keep1)  # 6*6 from image dimension
            self.fc1_us = nn.Linear(keep1, 120)

            #FC2 IS DECOMPOSED INTO VT2 AND US2 -- KEEP PARAMETER GOES HERE 
            self.fc2_vt = nn.Linear(120, keep2)
            self.fc2_us = nn.Linear(keep2, 84)

            #FC2 IS DECOMPOSED INTO VT2 AND US2 -- KEEP PARAMETER GOES HERE 
            self.fc3_vt = nn.Linear(84, keep3)
            self.fc3_us = nn.Linear(keep3, 26)

            #STATEMENT TO PLUG IN MATRIX PRODUCT FROM ABOVE IN FOR WEIGHTS:
            #using this to assign the weights from the previously trained network
            with torch.no_grad():
                self.fc1_vt.weight.data = vt1_red
                self.fc1_us.weight.data = us1_red
                self.fc2_vt.weight.data = vt2_red
                self.fc2_us.weight.data = us2_red
                self.fc3_vt.weight.data = vt3_red
                self.fc3_us.weight.data = us3_red

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # If the size is a square you can only specify a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            #ADD THE NEW LAYER IN THE FORWARD PASS
            x = F.relu(self.fc1_us(self.fc1_vt(x)))
            x = F.relu(self.fc2_us(self.fc2_vt(x)))
            x = self.fc3_us(self.fc3_vt(x))
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    #generate new lenet5 to work out this example
    svd_method_example = Net()
    print(svd_method_example)

    return svd_method_example

# method to iterate through a list of pruning percentages and return the data
    # input: original lenet5, list of percentages
def collect_SVD_FC_data(net, p_list):
    #dict to output experiment data
        # keys will be the percentages and values will be lists of output logs values
    data_dict = {}

    #get dataloaders for each network
    data_loaders, train_loader, val_loader, test_loader = get_data_loaders()

    for percent in p_list:
        #ordered contents of output list will be:
            # number of parameters
            # average training loss log - a
            # average validation loss log - b
            # average training accuracy log - c
            # average validation accuracy log - d
            # stopping epoch - e
            # test accuracy

        output_list = []
        svd_net = generate_fc_svd_network(net, percent)

        #add number of parameters to the list
        output_list.append(sum([param.nelement() for param in svd_net.parameters()]))

        #call to the training method:
        a,b,c,d,e = training_with_data_collection(net=svd_net,
                                                data_loaders=data_loaders,
                                                num_epochs=num_epochs,
                                                patience=patience)
        output_list.append(a)
        output_list.append(b)
        output_list.append(c)
        output_list.append(d)
        output_list.append(e)

        model_save_name = 'Lenet5_SVD_FC_{}.pt'.format(percent)
        path = F"/content/gdrive/My Drive/18.065 Final Project/{model_save_name}" 
        torch.save(svd_net.state_dict(), path)

        #get test accuracy
        test_acc = get_test_accuracy(svd_net, test_loader)
        output_list.append(test_acc)

        #assign this output list to the value for this entry of the data dict
        data_dict[percent] = output_list

    return data_dict


def main():
	net = Net()
	print(net)
	#here the preloaded model can be loaded from path

	#data object to be saved to flie
	experiment_data = collect_SVD_FC_data(net, [0.05, 0.10, 0.20, 0.50, 0.75])



if __name__ == "__main__":
    main()    


