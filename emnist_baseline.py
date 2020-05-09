import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np

# Define Hyper-parameters 
num_classes = 26
num_epochs = 100
batch_size = 100
learning_rate = 0.001
patience = 10

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
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

#method to download EMNIST and spit out the dataloaders
def get_data_loaders():
    transform = transforms.Compose(
        [transforms.Resize((32,32)),
        transforms.ToTensor()])

    # EMNIST dataset  -- trying on the 'letters' split (26 classes of upper and lower case letters)
    train_dataset = torchvision.datasets.EMNIST(root='../../data',
                                            split='letters', 
                                            train=True, 
                                            transform=transform,  
                                            download=True)

    #Break up training data into train and val
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, (104000, 20800))


    test_dataset = torchvision.datasets.EMNIST(root='../../data',
                                            split='letters', 
                                            train=False, 
                                            transform=transform)

    # Create Data loaders

    #feed train and val subsets into data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_subset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_subset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    #create dictionary for validation step in training
    data_loaders = {"train": train_loader, "val": val_loader}

    return data_loaders, train_loader, val_loader, test_loader


#training method, takes: model, dictionary of train and loaders, num epochs, and early stopping tolerance
def early_stop_training(net, data_loaders, num_epochs, patience):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) 

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # Train the model
    total_step = len(data_loaders['train'])
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            #iterate over data for corresponding phase
            for i, (images, labels) in enumerate(data_loaders[phase]):
                #correct EMNIST labeling issue:
                labels = labels-1
                
                # Forward pass
                outputs = net(images)
                loss = criterion(outputs, labels)
                
                # Backprpagation and optimization
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()
                else:
                    #if we're in a validation step then store the avg valid. loss
                        #accumulate in the running_loss variable
                    running_loss  = running_loss + loss.item()
                    #and then at the end of the epoch, divide off by the number of samples (208):
                        #divide total epoch loss by --> len(val_subset)/batch_size

                if (i+1) % 100 == 0:
                    print ('{}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(phase, epoch+1, num_epochs, i+1, total_step, loss.item()))


            #UPDATE STEP FOR AVG_VALID_LOSSES LIST
            #at the end of the loop, if this was a validation loop of accumulating val. loss
                # --> this will be the case if running_loss>0.0
            if running_loss > 0.0:
                num_samples = len(val_loader.dataset)/batch_size
                avg_val_loss = running_loss / num_samples
                avg_valid_losses.append(avg_val_loss)
                print('Running list of avg. val. losses: {}'.format(avg_valid_losses))


            #EARLY STOPPING CHECK STEP
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

#TRAINING METHOD WITH OUTPUT OF TRAIN AND VAL LOSS AND ACC. LOGS, AND NUM. EPOCHS TRAINED

#training method, takes: model, dictionary of train and loaders, num epochs, and early stopping tolerance
def training_with_data_collection(net, data_loaders, num_epochs, patience):
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

                # Backpropagation and optimization
                optimizer.zero_grad()

                running_loss += loss.item()
                # backward + optimize only if in training phase
                if phase == 'train':
                    #ACCUMULATE RUNNING LOSS FOR TRAIN PHASE
                    # running_loss = running_loss + loss.item()

                    #backward step
                    loss.backward()
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
    

def get_test_accuracy(net, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            labels = labels-1
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return(100 * correct / total) 


def main():
	net = Net()
	print(net)

	#get dataloaders
	data_loaders, train_loader, val_loader, test_loader = get_data_loaders()

	#call to the training method:
	early_stop_training(net=net, data_loaders=data_loaders, num_epochs=num_epochs, patience=patience)

	#call to test method:
	test_acc = get_test_accuracy(net=net, test_loader=test_loader)



if __name__ == "__main__":
    main()







