import torch
import numpy as np

device = "cpu"
print("device = ", device)

# Loading the Fashion-MNIST dataset
from torchvision import datasets, transforms
data_path = "DATASET"

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# Download and load the training data
trainset = datasets.FashionMNIST(data_path, download=False, train=True, transform=transform)
testset = datasets.FashionMNIST(data_path, download=False, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

print(len(trainset))
print(len(testset))

# Examine a sample
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

# Define the network architecture
from torch import nn, optim

model = nn.Sequential(
    nn.Linear(28*28, 300),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Linear(100,10),
    nn.LogSoftmax(dim=1)
).to(device)

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#loss_fn = nn.NLLLoss()
loss_fn = nn.CrossEntropyLoss()

n_epochs = 5 # set to zero
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        inp = imgs.reshape(imgs.shape[0], -1).to(device)
        labels = labels.to(device)
        outs = model(inp)
        loss = loss_fn(outs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

#torch.save(model.state_dict(), "MLP_fashion.ckpt")
