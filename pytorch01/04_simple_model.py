# Using autograd

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

def training_loop(Nepochs, learning_rate, params, t_u, t_c):
    for epoch in range(1,Nepochs+1):
        #
        if params.grad is not None:
            params.grad.zero_()
        #
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        # Update parameters
        with torch.no_grad():
            params -= learning_rate*params.grad
        #        
        if epoch % 500 == 0:
            print("Epoch %8d, Loss %18.10f " % (epoch, float(loss)))
            print("updated params = ", params)
            print("grad           = ", params.grad)
    return params

# Using normalized/scaled input
print("\nUsing scaled input\n")
t_un = 0.1*t_u
params = training_loop(
    Nepochs=5000,
    learning_rate=1e-2,
    params=torch.tensor([1.0, 0.0], requires_grad=True),
    t_u=t_un,
    t_c=t_c
)
print("params = ", params)
