import torch
import torch.nn as nn

torch.manual_seed(1234) # for reproducibility

#
# Prepare data
#
x = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
# Append squeeze
x = x.unsqueeze_(1)
t = t.unsqueeze_(1)

# Split into training and validation dataset
Nsamples = x.shape[0]
Nval = int(0.2*Nsamples) # for validation data

shuffled_indices = torch.randperm(Nsamples)
train_indices = shuffled_indices[:-Nval]
val_indices = shuffled_indices[-Nval:]

x_train = x[train_indices]
t_train = t[train_indices]

x_val = x[val_indices]
t_val = t[val_indices]

# scale or normalize feature
x_train_n = 0.1*x_train
x_val_n = 0.1*x_val

#
# Prepare model
#

#seq_model = torch.nn.Sequential(
#    nn.Linear(1,8),
#    nn.Tanh(),
#    nn.Linear(8,1)
#)

from collections import OrderedDict
seq_model = nn.Sequential( OrderedDict([
    ('hidden_linear', nn.Linear(1, 2)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(2, 1))
]) )

optimizer = torch.optim.Adam(
    seq_model.parameters(),
    lr=1e-2
)

def training_loop(Nepochs, optimizer, model, loss_fn,
                   x_train, t_train, x_val, t_val):
    for epoch in range(1,Nepochs+1):
        
        t_train_m = model(x_train)
        loss_train = loss_fn(t_train_m, t_train)

        t_val_m = model(x_val)
        loss_val = loss_fn(t_val_m, t_val)

        optimizer.zero_grad()
        loss_train.backward() # backprop on training data
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch:8d}, Training loss {loss_train.item():10.4f},"
                  f" Validation loss {loss_val.item():10.4f}")


training_loop(
    Nepochs=5000, 
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(),
    x_train=x_train_n,
    t_train=t_train,
    x_val=x_val_n,
    t_val=t_val
)

# Model
t_m = seq_model(0.1*x)

#print("Loss on data: ", nn.MSELoss(t_m, t))

print('output', seq_model(x_val_n))
print('answer', t_val)
#print('hidden', seq_model.hidden_linear.weight.grad)

import matplotlib.pyplot as plt
import numpy as np

x_plt = x.numpy().squeeze()
t_plt = t.numpy().squeeze()
t_m_plt = t_m.detach().numpy().squeeze()

print(x_plt.shape)
print(t_plt.shape)
print(t_m_plt.shape)

plt.clf()
idx_sorted = np.argsort(x_plt)
x_plt = x_plt[idx_sorted]
t_plt = t_plt[idx_sorted]
t_m_plt = t_m_plt[idx_sorted]

plt.plot(x_plt, t_plt, marker="x", label="data", linewidth=0)
plt.plot(x_plt, t_m_plt, marker="o", label="model")
plt.xlabel("Temperature (°Celsius)")
plt.ylabel("Temperature (°Fahrenheit)")
plt.legend()
plt.savefig("IMG_10_simple_nn.pdf")

