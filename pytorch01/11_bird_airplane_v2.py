from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

#device = "cpu"
device = "cuda"

torch.manual_seed(123)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

from torchvision import datasets, transforms
data_path = 'DATASET'

cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4915, 0.4823, 0.4468),
            (0.2470, 0.2435, 0.2616)
        )
    ])
)

cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4915, 0.4823, 0.4468),
            (0.2470, 0.2435, 0.2616)
        )
    ])
)

label_map = {0: 0, 2: 1} # airplane = 0 -> 0, bird = 2 -> 1 
class_names = ["airplane", "bird"]
cifar2 = [
    (img, label_map[label])
    for img, label in cifar10
    if label in [0, 2]
]
cifar2_val = [
    (img, label_map[label])
    for img, label in cifar10_val
    if label in [0, 2]
]


train_loader = torch.utils.data.DataLoader(
    cifar2, batch_size=64, shuffle=True
)

model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
).to(device)
#model.load_state_dict(torch.load("11_bird_airplane_v2.ckpt"))

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

n_epochs = 100 # set to zero
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

torch.save(model.state_dict(), "11_bird_airplane_v2.ckpt")

# Loader, not shuffled, for
train_loader = torch.utils.data.DataLoader(
    cifar2, batch_size=64, shuffle=False
)
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in train_loader:
        inp = imgs.reshape(imgs.shape[0], -1).to(device)
        labels = labels.to(device)
        outputs = model(inp).to(device)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        #print(predicted)
        #print(labels.to)
        correct += int((predicted == labels).sum())

print("Accuracy (train): %f" % (correct / total))




val_loader = torch.utils.data.DataLoader(
    cifar2_val, batch_size=64, shuffle=False
)
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        inp = imgs.reshape(imgs.shape[0], -1).to(device)
        labels = labels.to(device)
        outputs = model(inp).to(device)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy (validation): %f" % (correct / total))
