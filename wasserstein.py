#it is based on load and train protein in data

import torch

import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data

import torch.nn.functional as f

import matplotlib.pyplot as plt
import numpy as np

import csv


#################################
# Building a network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.fc1=nn.Linear(9,50)
        self.fc2=nn.Linear(50,1)

    def forward(self, x):

        output =self.fc1(x)
        output=self.fc2(output)
        return output

#####################################
# Making a dataset
# importing data from CSV


y=[]; x=[]
with open('../data/protein/CASP.csv') as file:
    next(file)
    for row in file:
        data_point=row.strip().split(',')
        data_point_num=[float(i) for i in data_point]

        y.append(np.array(data_point_num[0]))

        x.append(data_point_num[1:])
            #print(row['F1'])


tensor_x=torch.stack([torch.Tensor(i) for i in x])
tensor_y=torch.stack([torch.Tensor(i) for i in y])
dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

#train_length=int(0.8*len(dataset))
#test_length=1-train_length

# dividing the whole dataset into train and test
# we compute the distribution of the whole dataset (trainval+test)
# we divide trainval into train and val where the fraction for train datatset varies
# training is used with train dataset (loss1) and wasserstein (comparing batch distribution with population distribution)

trainval_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])

train_percentages=[1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
#train_percentages=[0.001, 0.005]

accuracies=[]

for perc in train_percentages:
    print("percentage of training dataset: %f\n" % perc)

    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [int(perc*len(trainval_dataset)), len(trainval_dataset)-int(perc*len(trainval_dataset))])

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


    ##################################

    net=Net()
    optimizer=optim.Adam(net.parameters(), lr=0.005)
    criterion=nn.MSELoss()
    #####################################

    def calculate_quantiles(data, quantile_number):

        data_sorted=torch.sort(data)    
        data_quantile_share=torch.linspace(0,(len(data_sorted[0])-1), quantile_number+1)
        #data_quantile_share=data_quantile_share/

        data_quantile_share_floor=torch.floor(data_quantile_share)
        data_quantile_share_floor_int=data_quantile_share_floor.detach().numpy().astype(int) #turn to ints to be read is index
        data_quantile_share_ceil=torch.ceil(data_quantile_share)
        data_quantile_share_ceil_int=data_quantile_share_ceil.detach().numpy().astype(int)
        
        data_qunatile_share_fractional=data_quantile_share-data_quantile_share_floor

        earlier_elem=data_sorted[0][data_quantile_share_floor_int]
        later_elem=data_sorted[0][data_quantile_share_ceil_int]

        quantiles=earlier_elem+data_qunatile_share_fractional*(later_elem-earlier_elem)
        return quantiles


    def wasserstein_loss(y_pred):
        ypred_quantiles=calculate_quantiles(y_pred, quantile_num)
        wass_loss=torch.mean((ytruepop_quantiles- ypred_quantiles)**2)
        return wass_loss

    ######################################################


    quantile_num=400
    ytruepop_quantiles=calculate_quantiles(tensor_y, quantile_num)

    best_overall=1000000000000000
    for iter in range(10):
        print("-------------------------------Run: %d" % iter)
        best_testmseloss=100000000000000
        not_improved=0
        epoch=0
        while (not_improved<51):
        #for epoch in range(30):
            epoch+=1
            print("Epoch: %d" % epoch)
            for ind, (x,y) in enumerate(train_dataset_loader):
                optimizer.zero_grad()
                forw=net(x)
                loss1=criterion(forw, y)
                #loss_wass=wasserstein_loss(forw)
                #loss=loss1+loss_wass
                loss=loss1

                loss.backward()
                optimizer.step()

            i=0
            test_mseloss=0
            for ind, (x,y) in enumerate(test_dataset_loader):
                i+=1
                forw=net(x)
                mse=criterion(forw, y)
                test_mseloss+=mse
            if ((test_mseloss/i)<best_testmseloss):
                best_testmseloss=test_mseloss/i
                print("test mse loss: %.3f" % (test_mseloss/i))
                not_improved=0
            else:
                not_improved+=1
        best_sofar=float("%.2f" % best_testmseloss.detach().numpy())
        best_overall=np.minimum(best_overall, best_sofar)

    accuracies.append(best_overall)
    print(accuracies)
    
        

        