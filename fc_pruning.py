import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np

from emnist_baseline import get_data_loaders, get_test_accuracy, Net

# Define Hyper-parameters 
num_classes = 26
num_epochs = 100
batch_size = 100
learning_rate = 0.001
patience = 10

#defining in-place pruning to maintain weight.grad pointer
# input: weight matrix mtx, percent to remove pct; output: pruned tensor, mask
def magnitude_prune_fc(mtx, pct):
    # sorted list of (magnitudes of) elements in tensor
    l = [np.abs(element.item()) for element in mtx.flatten()]
    l.sort()
    # cut off the pct% smallest elements
    threshold = l[int(pct*len(l))]
    for i in range(len(mtx)):
        for j in range(len(mtx[i])):
            if np.abs(mtx[i][j]) < threshold:
                mtx[i][j]=0.0

    # generate mask for multiplication against gradient mtx
    mask = (mtx != 0.0)
    mask = mask.float()

    return mtx, mask

#METHOD THAT TAKES IN LENET 5 AND PERCENT, REASSIGNS CONV LAYERS, RETURNS MASKS FOR TRAINING METHOD
def generate_fc_prune_network(net, percent):
    fc1 = net.fc1.weight.data
    fc2 = net.fc2.weight.data
    fc3 = net.fc3.weight.data

    fc1_pruned, fc1_mask = magnitude_prune_fc(fc1, percent)
    fc2_pruned, fc2_mask = magnitude_prune_fc(fc2, percent)
    fc3_pruned, fc3_mask = magnitude_prune_fc(fc3, percent)

    net.fc1.weight.data = fc1_pruned
    net.fc2.weight.data = fc2_pruned
    net.fc3.weight.data = fc3_pruned

    return fc1_mask, fc2_mask, fc3_mask

#TRAINING METHOD WITH OUTPUT OF TRAIN AND VAL LOSS AND ACC. LOGS, AND NUM. EPOCHS TRAINED

#training method, takes: model, dictionary of train and loaders, num epochs, and early stopping tolerance
def fc_training_with_data_collection(net, data_loaders, num_epochs, patience, fc1_mask, fc2_mask, fc3_mask):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) 

    #VARIABLE TO RETURN THE STOPPPING EPOCH
    stopping_epoch = 0
    
    #LOSS AND ACCURACY LOGS TO BE RETURNED AFTER TRAINING
    avg_training_loss_log = []
    # avg_valid_losses is already created and used below, return that
    avg_training_acc_log = []
    avg_val_acc_log = []


    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    #each epoch store the average validation loss 
    
    # Train the model
    total_step = len(data_loaders['train'])
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # optimizer = scheduler(optimizer, epoch)
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode

            #RUNNING LOSS INITIALIZED AT THE BEGINNING OF EACH PHASE:
            running_loss = 0.0
            #WILL ALSO INITIALIZE A RUNNING ACCURACY COUNTER:
            running_accuracy = 0.0

            #------ITERATE THROUGH PHASE DATALOADER------

            #iterate over data for corresponding phase
            for i, (images, labels) in enumerate(data_loaders[phase]):
                #correct EMNIST labeling issue:
                labels = labels-1
                
                # Forward pass
                outputs = net(images)
                loss = criterion(outputs, labels)

                #-------BATCH ACCURACY CALCULATION------
                with torch.no_grad():
                    correct = 0
                    total = 0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    batch_accuracy = (100 * correct / total)
                    running_accuracy += batch_accuracy
                #---------------------------------------

                # Backprpagation and optimization
                optimizer.zero_grad()

                running_loss += loss.item()
                # backward + optimize only if in training phase
                if phase == 'train':
                    #backward step
                    loss.backward()
                    #------APPLY PRUNING MASKS TO GRADIENT------
                    # sparsify_grad()
                    net.fc1.weight.grad.data.mul_(fc1_mask)
                    net.fc2.weight.grad.data.mul_(fc2_mask)
                    net.fc3.weight.grad.data.mul_(fc3_mask)
                    #-------------------------------------------
                    # update the weights
                    optimizer.step()

                if (i+1) % 100 == 0:
                    print ('{}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(phase, epoch+1, num_epochs, i+1, total_step, loss.item()))


            #------UPDATE STEP FOR LOSS AND ACC LOGS------
            #UPDATE OF LOSS LOGS
            if running_loss > 0.0:
                if phase == 'val':
                    #UPDATE VAL LOSS LOG
                    num_samples = len(val_loader.dataset)/batch_size
                    # print('{} samples validation loop'.format(num_samples))
                    avg_val_loss = running_loss / num_samples
                    print('avg. val. loss of {} for this epoch'.format(avg_val_loss))
                    avg_valid_losses.append(avg_val_loss)
                    print('Running list of avg. val. losses: {}'.format(avg_valid_losses))

                else:
                    #UPDATE TRAIN LOSS LOG
                    num_samples = len(train_loader.dataset)/batch_size
                    avg_trn_loss = running_loss / num_samples
                    print('avg. train. loss of {} for this epoch'.format(avg_trn_loss))
                    avg_training_loss_log.append(avg_trn_loss)
                    print('Running list of avg. train losses: {}'.format(avg_training_loss_log))

            #UPDATE OF ACCURACY LOGS (avg_training_acc_log, avg_val_acc_log)
            if running_accuracy > 0.0:
                if phase == 'val':
                    #UPDATE VAL ACC LOG
                    num_samples = len(val_loader.dataset)/batch_size
                    avg_val_acc = running_accuracy / num_samples
                    print('avg. val. accuracy of {} for this epoch'.format(avg_val_acc))
                    avg_val_acc_log.append(avg_val_acc)
                    print('Running list of avg. val. accuracies: {}'.format(avg_val_acc_log))
                
                else:
                    #UPDATE TRAIN ACC LOG
                    num_samples = len(train_loader.dataset)/batch_size
                    avg_trn_acc = running_accuracy / num_samples
                    print('avg. training accuracy of {} for this epoch'.format(avg_trn_acc))
                    avg_training_acc_log.append(avg_trn_acc)
                    print('Running list of avg. training accuracies: {}'.format(avg_training_acc_log))



            #------EARLY STOPPING CHECK STEP------

            #stop criterion: val loss hasn't decreased in (patience) many epochs
                #so for length l we consider (avg_val_losses[l-5], ..., avg_val_losses[l-1])
                #-> so stop if the most recent loss is greater than all preceeding elements in the considered window:

            #considered test list:
            l = len(avg_valid_losses)
            #have to wait for at least (patience) epochs to pass
            if l >= patience:
                early_stopping_window = avg_valid_losses[(l - patience):l]
                print('Early stopping window: {}'.format(early_stopping_window))
                #check:
                if avg_valid_losses[l-1] == max(early_stopping_window):
                    print('EARLY STOPPING CRITERION MET, STOP TRAINING')
                    stopping_epoch = epoch+1
                    break
            #breaks to get out of training loop
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break


    return avg_training_loss_log, avg_valid_losses, avg_training_acc_log, avg_val_acc_log, stopping_epoch
    
# method to iterate through a list of pruning percentages and return the data
    # input: original lenet5, list of percentages
def collect_prune_fc_data(net, p_list):
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
        #THE MAGNITUDE PRUNING METHOD PRUNES THE WEIGHTS IN PLACE AND RETURNS THE MASKS
            #NEED ACCOUNT FOR THAT AND FEED THE MASKS INTO THE CORRECT TRAINING METHOD  
        fc1_mask, fc2_mask, fc3_mask = generate_fc_prune_network(net, percent)

        #add number of parameters to the list
        output_list.append(sum([param.nelement() for param in net.parameters()]))

        #call to the training method:
        a,b,c,d,e = fc_training_with_data_collection(net,
                                                    data_loaders,
                                                    num_epochs,
                                                    patience,
                                                    fc1_mask,
                                                    fc2_mask,
                                                    fc3_mask)
        output_list.append(a)
        output_list.append(b)
        output_list.append(c)
        output_list.append(d)
        output_list.append(e)

        #get test accuracy
        test_acc = get_test_accuracy(net, test_loader)
        output_list.append(test_acc)

        #assign this output list to the value for this entry of the data dict
        data_dict[percent] = output_list

    return data_dict

def main():
	net = Net()
	print(net)
	#here the preloaded model can be loaded from path

	#data object to be saved to flie
	experiment_data = collect_prune_fc_data(net, [0.95, 0.9, 0.8, 0.5, 0.25])



if __name__ == "__main__":
    main()    





