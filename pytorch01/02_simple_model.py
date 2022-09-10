# Handwritten-gradient

import torch

def model(t_u, w, b):
    return w*t_u + b

def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1.0

def loss_fn(t_p, t_c):
    s = (t_p - t_c)**2
    return s.mean()

def dloss_fn(t_p, t_c):
    dsq_diffs = 2*(t_p - t_c)/t_p.size(0)
    return dsq_diffs

# Gradient of the loss function wrt w and b
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

# Data
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

# Initial model parameters
w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
loss = loss_fn(t_p, t_c)

Δ = 0.01
loss_rate_of_change_w = \
    (loss_fn(model(t_u, w + Δ, b), t_c) -
    loss_fn(model(t_u, w - Δ, b), t_c)) / (2.0*Δ)
loss_rate_of_change_b = \
    (loss_fn(model(t_u, w, b + Δ), t_c) -
    loss_fn(model(t_u, w, b - Δ), t_c)) / (2.0*Δ)

print("Using finite-difference: ")
print("loss_rate_of_change_w = ", loss_rate_of_change_w)
print("loss_rate_of_change_b = ", loss_rate_of_change_b)

grad = grad_fn(t_u, t_c, t_p, w, b)

print("Using grad:")
print("grad = ", grad)

#learning_rate = 1e-2
#w = w - learning_rate * loss_rate_of_change_w
#b = b - learning_rate * loss_rate_of_change_b


