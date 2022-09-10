import torch

def model(t_u, w, b):
    return w*t_u + b

def loss_fn(t_p, t_c):
    s = (t_p - t_c)**2
    return s.mean()

# Data
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

# Initial model parameters
w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
print(t_p)

loss = loss_fn(t_p, t_c)
print(loss)

Δ = 0.1

loss_rate_of_change_w = \
    (loss_fn(model(t_u, w + Δ, b), t_c) -
    loss_fn(model(t_u, w - Δ, b), t_c)) / (2.0*Δ)

loss_rate_of_change_b = \
    (loss_fn(model(t_u, w, b + Δ), t_c) -
    loss_fn(model(t_u, w, b - Δ), t_c)) / (2.0*Δ)

learning_rate = 1e-2
w = w - learning_rate * loss_rate_of_change_w
b = b - learning_rate * loss_rate_of_change_b

print("w = ", w)
print("b = ", b)