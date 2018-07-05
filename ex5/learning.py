import torch
import torch.nn.init as init
from torch.autograd import Variable

import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
torch.manual_seed(8)

import matplotlib.pyplot as plt
import numpy as np

#MNIST dataset
mnist_train = dsets.MNIST(root = 'data/',
                          train = True,
                          transform = transforms.ToTensor(),
                          download = True)
mnist_test = dsets.MNIST(root = 'data/',
                         train = False,
                         transform = transforms.ToTensor(),
                         download = True)
"""
print(mnist_train.train_data.size())
print(mnist_train.train_labels.size())

idx = 0
plt.imshow(mnist_train.train_data[idx,:,:].numpy(),cmap='gray')
plt.title('%i' %mnist_train.train_labels[idx])
plt.show()
"""

#Data loader
batch_size = 100

data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers = 1)

def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

batch_images, batch_labels = next(iter(data_loader))
"""
print(batch_images.size())
print(batch_labels.size())

imshow(utils.make_grid(batch_images))
print(batch_labels.numpy())
"""
#Define Neural Network
linear1 = torch.nn.Linear(784,512,bias=True)
linear2 = torch.nn.Linear(512,10,bias=True)
relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1,relu,linear2)
    
print(model)

cost_func = torch.nn.CrossEntropyLoss() #including softmax

if __name__=='__main__':
    #Model training
    learning_rate = 0.001
    training_epochs = 5

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = len(mnist_train)//batch_size
    
        for i, (batch_images,batch_labels) in enumerate(data_loader):
            X = Variable(batch_images.view(-1,28*28))
            Y = Variable(batch_labels)
    
            optimizer.zero_grad()
            Y_prediction = model(X)
            cost = cost_func(Y_prediction,Y)
            cost.backward()
            optimizer.step()
        
            avg_cost += cost/total_batch

        print("[Epoch: {}] cost = {}".format(epoch+1, avg_cost.data[0].item()))

    print("Learning finished!")

    print("Saving model...")
    torch.save(model.state_dict(),'model.pkl')
    print("Model is saved!")

    #Measuring performance
    correct = 0
    total = 0
    for images, labels in mnist_test:
        images  = Variable(images.view(-1, 28 * 28))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == labels).sum()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

