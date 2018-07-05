import learning

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random

#Loading model
model = learning.model
model.load_state_dict(torch.load('model.pkl'))

#Measure performance
correct = 0
total = 0
for images, labels in learning.mnist_test:
    images = Variable(images.view(-1,28*28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data,1)
    total += 1
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images : %d %%' %(100*correct/total))

#Random test
r = random.randint(0,len(learning.mnist_test)-1)
X_single_data = Variable(learning.mnist_test.test_data[r:r + 1].view(-1,28*28).float())
Y_single_data = Variable(learning.mnist_test.test_labels[r:r + 1])

single_prediction = model(X_single_data)

print('Label : ', Y_single_data.data.view(1).numpy())
print('Prediction : ', torch.max(single_prediction.data, 1)[1].numpy())

plt.imshow(X_single_data.data.view(28,28).numpy(), cmap='gray')
plt.show()
