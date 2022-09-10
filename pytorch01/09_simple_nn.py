import torch

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

# Linear system:
#
# Args:
#     in_features: size of each input sample
#     out_features: size of each output sample
#     bias: If set to ``False``, the layer will not learn an additive bias.
#         Default: ``True``

LinearModel = torch.nn.Linear(1,1)
print("Test evaluating")
print( LinearModel(x_val_n) )

print("\nLinearModel weight and bias:")
print(LinearModel.weight)
print(LinearModel.bias)

print("\nUsing manual calculation:")
print(LinearModel.weight*x_val_n[0] + LinearModel.bias)
print(LinearModel.weight*x_val_n[1] + LinearModel.bias)


optimizer = torch.optim.SGD(
    LinearModel.parameters(),
    lr=1e-3
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

# 
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

training_loop(
    3000, 
    optimizer,
    LinearModel,
    loss_fn,
    x_train,
    t_train,
    x_val,
    t_val
)

# Model
t_m = LinearModel(x)

print("Loss on data: ", loss_fn(t_m, t))

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
plt.savefig("IMG_09_simple_nn.pdf")

