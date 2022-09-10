# Quadratic equation

import torch
import torch.optim as optim

torch.manual_seed(1234) # for reproducibility

#def calc_model(x, w1, w2, b):
#    return w2 * x**2 + w1*x + b

def calc_model(x, w1, w2, w3, b):
    return w3*x**3 + w2 * x**2 + w1*x + b

def calc_loss(t_p, t_c):
    s = (t_p - t_c)**2
    return s.mean()

# Data
x = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]) # feature
t = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]) # target

# Split into training and validation dataset
Nsamples = x.shape[0]
Nval = int(0.2*Nsamples) # for validation data

shuffled_indices = torch.randperm(Nsamples)
train_indices = shuffled_indices[:-Nval]
val_indices = shuffled_indices[-Nval:]

# Training data
x_train = x[train_indices]
t_train = t[train_indices]

# Validation data
x_val = x[val_indices]
t_val = t[val_indices]

def training_loop( Nepochs, optimizer, params, x_train, t_train, x_val, t_val):
    #
    for epoch in range(1,Nepochs+1):
        #
        t_train_model = calc_model(x_train, *params)
        loss_train = calc_loss(t_train_model, t_train)
        #
        t_val_model = calc_model(x_val, *params)
        loss_val = calc_loss(t_val_model, t_val)
        #
        # No val_loss.backward() here, since we don’t want to train the validation
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        #        
        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch:8d}, Training loss {loss_train.item():18.10f},"
                  f" Validation loss {loss_val.item():18.10f}")
    return params


# Initial parameters
#params = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
params = torch.tensor([1.0, 1.0, 0.0, 0.0], requires_grad=True)

learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate)
params = training_loop(
    Nepochs=5000,
    optimizer=optimizer,
    params=params,
    x_train=x_train,
    t_train=t_train,
    x_val=x_val,
    t_val=t_val
)
print("params = ", params)

#t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
#t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_model = calc_model(x, *params)
loss_model = calc_loss(t_model, t)
print("loss = ", loss_model)

print(x)
print("t = "); print(t)
print("t_model = "); print(t_model)


import matplotlib.pyplot as plt
import numpy as np

plt.clf()
idx_sorted = np.argsort(x)
plt.plot( x.numpy()[idx_sorted], t.detach().numpy()[idx_sorted],
  marker="x", label="data", linewidth=0) # no marker
plt.plot( x.numpy()[idx_sorted], t_model.detach().numpy()[idx_sorted],
  marker="o", label="model")
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.legend()
plt.savefig("IMG_08_simple_model.pdf")
