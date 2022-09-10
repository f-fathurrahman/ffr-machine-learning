import numpy as np
import d2l_torch as d2l

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("default")
plt.rcParams.update({"text.usetex": True})

import torch
from torch.utils import data

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True): #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))


from torch import nn
net = nn.Sequential(nn.Linear(2,1))

# Initialize the parameter manually
net[0].weight.data.normal_(0.0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward() # calculate derivatives
        trainer.step() # update parameters
    l = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {l:f}")

w = net[0].weight.data
print("error in estimating w:", true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print("error in estimating b:", true_b - b)