import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(1234) # for reproducibility

# The output of torchvision datasets are PILImage images with range [0,1]
# We transform them to Tensors of normalized range [-1,1].
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5))]
)

# Data will be downloaded in .data directory
trainset = torchvision.datasets.CIFAR10(
    root=".data", train=True, download=False, transform=transform
)
print("trainset = ")
print(trainset)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='.data', train=False, download=False, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 + 0.5 # transform back
    npimg = img.numpy()
    plt.imshow( np.transpose(npimg, (1,2,0)) )
    plt.savefig("IMG_some_images.png", dpi=150)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


import torch.nn as nn
import torch.nn.functional as F
from my_net import Net

PATH = './cifar_net_trained.pth' # retrain
net = Net()
net.load_state_dict(torch.load(PATH))

# Use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


