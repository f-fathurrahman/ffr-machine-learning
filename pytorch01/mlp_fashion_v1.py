import torch
import numpy as np

print("PyTorch version = ", torch.__version__)

device = "cpu"

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
# Not using dataloader for simplicity

# Examine a sample
#images, labels = trainset[0]
#print(type(images))
#print(images.shape)
#print(labels)

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
loss_fn = nn.NLLLoss()

n_epochs = 30
for epoch in range(n_epochs):
    acc_loss = 0
    for img, label in trainset:
        inp = img.reshape(img.shape[0], -1).to(device)
        label = torch.tensor([label]).to(device)
        out = model(inp)
        loss = loss_fn(out, label)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        acc_loss = acc_loss + float(loss)
    print("Epoch: %d, Loss: %f" % (epoch, acc_loss/len(trainset)))

#torch.save(model.state_dict(), "MLP_fashion.ckpt")
