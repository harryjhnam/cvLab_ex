import torch
import torch.nn.init as init
from torch.autograd import Variable

import torchvision.utils as utils
import torchvision.transforms as transforms
torch.manual_seed(8)

import numpy as np

import glog
import time
from tensorboardX import SummaryWriter

writer = SummaryWriter()

"""download the dataset and set dataloader"""
import torchvision.datasets as dsets

batch_size = 100

trainset = dsets.MNIST(root = 'data/', train = True,
                       transform = transforms.ToTensor(), download = True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                           shuffle=True, num_workers=2)

testset = dsets.MNIST(root = 'data/', train = False,
                      transform = transforms.ToTensor(), download = True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                          shuffle=True, num_workers=2)



"""Building Model"""

linear1 = torch.nn.Linear(784,512,bias=True)
linear2 = torch.nn.Linear(512,10,bias=True)
relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1,relu,linear2)
    
print(model)

loss_func = torch.nn.CrossEntropyLoss() #including softmax


"""hyperparameters - take arguments (lr, epochs)"""
import sys
learning_rate = sys.argv[1]
training_epochs = sys.argv[2]
print("hyperparams : lr = {}, train_epochs = {}".format(learning_rate, training_epochs))

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


"""training"""
for epoch in range(training_epochs):
    avg_loss = 0
    total_batch = len(trainset)//batch_size
    start_time = time.time()
    
    for i, (batch_images,batch_labels) in enumerate(train_loader):
        X = Variable(batch_images.view(-1,28*28))
        Y = Variable(batch_labels)
        
        optimizer.zero_grad()
        Y_prediction = model(X)
        
        #calculating loss
        loss = loss_func(Y_prediction,Y)
        
        #calculating accuracy
        _, predicted = torch.max(Y_prediction.data, 1)
        acc = 100*(predicted==Y).sum()/batch_size
        
        loss.backward()
        optimizer.step()
        
        avg_loss += loss/total_batch
        writer.add_scalar('Train/Loss', loss, epoch*total_batch+i)
        writer.add_scalar('Train/Acc', acc, epoch*total_batch+i)
        
    print("[Epoch: {}] loss = {}, acc = {}%".format(epoch+1, avg_loss.data[0].item(), acc))
    print("         training takes {} secs".format(time.time()-start_time))
    
print("Learning finished!")

#Saving model

print("Saving model...")
torch.save(model.state_dict(),'model.pkl')
print("Model is saved!")

writer.close()

#Measuring performance
correct = 0
total = 0
for images, labels in testset:
    images  = Variable(images.view(-1, 28 * 28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += 1
    correct += (predicted == labels).sum()
    
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))